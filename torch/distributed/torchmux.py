"""
torchmux: Simulate N-GPU distributed training on M GPUs (M ≤ N).

Usage:
    python -m torch.distributed.torchmux --nproc 8 train.py [args...]
    python -m torch.distributed.torchmux --nproc 8 --ngpus 2 train.py [args...]

Launches N worker processes mapped round-robin onto M physical GPUs.
Workers execute cooperatively: only one runs per GPU at a time, yielding
at collective boundaries. When a worker yields it checkpoints its entire
CUDA context via the driver API (VRAM returned to the driver) and
restores it when rescheduled. Collectives are resolved by bookkeeping
each rank's tensor contribution on disk — no NCCL is needed.

Produces two chrome://tracing traces (set TORCHMUX_TRACE_DIR, default
/tmp): a natural trace showing actual serial execution with
snapshot/restore overhead, and a synthetic trace reconstructing what
parallel execution would look like.
"""

import argparse
import json
import mmap
import os
import runpy
import shutil
import socket
import struct
import sys
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ---- torch.device remapping ----

_OrigDevice = torch.device
_ngpus = 1


class _DeviceMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, _OrigDevice)

    def __subclasscheck__(cls, subclass):
        if subclass is _OrigDevice:
            return True
        return type.__subclasscheck__(cls, subclass)


class _MuxDevice(metaclass=_DeviceMeta):
    def __new__(cls, *args, **kwargs):
        d = _OrigDevice(*args, **kwargs)
        if d.type == "cuda" and d.index is not None:
            return _OrigDevice("cuda", d.index % _ngpus)
        return d


# ---- Shared int via mmap (survives mp.spawn pickling) ----


class _SharedInt:
    def __init__(self):
        self._fd = tempfile.NamedTemporaryFile(delete=False)
        self._path = self._fd.name
        self._fd.write(struct.pack("i", 0))
        self._fd.flush()
        self._mm = mmap.mmap(self._fd.fileno(), 4)

    @property
    def value(self):
        self._mm.seek(0)
        return struct.unpack("i", self._mm.read(4))[0]

    @value.setter
    def value(self, v):
        self._mm.seek(0)
        self._mm.write(struct.pack("i", v))
        self._mm.flush()

    def cleanup(self):
        self._mm.close()
        self._fd.close()
        os.unlink(self._path)

    def __getstate__(self):
        return self._path

    def __setstate__(self, path):
        self._path = path
        self._fd = open(path, "r+b")
        self._mm = mmap.mmap(self._fd.fileno(), 4)


# ---- Cooperative scheduling ----

_sched = None  # _SharedInt — which rank may run
_rank = None
_ws = None
_held = False


def _acquire(*, restore=True):
    global _held
    while _sched.value != _rank:
        time.sleep(1e-5)
    _held = True
    if restore:
        _restore_gpu()


def _release(*, snapshot=True):
    global _held
    if snapshot:
        _snapshot_gpu()
    _sched.value = (_rank + 1) % _ws
    _held = False


# ---- Per-process trace recording ----

_trace_events = []  # list of (cat, name, start_us, dur_us)
_compute_start = None  # monotonic us when current compute phase began


def _us():
    return time.monotonic() * 1e6


def _trace(cat, name, start, dur):
    _trace_events.append((cat, name, start, dur))


def _begin_compute():
    global _compute_start
    _compute_start = _us()


def _end_compute():
    global _compute_start
    if _compute_start is not None:
        dur = _us() - _compute_start
        if dur > 0:
            _trace("compute", "compute", _compute_start, dur)
        _compute_start = None


# ---- GPU state checkpoint/restore via CUDA driver ----

_baton = None  # CudaBaton instance
_checkpointed = False  # has this process ever been checkpointed?


def _snapshot_gpu():
    global _checkpointed
    if not torch.cuda.is_initialized():
        return
    torch.cuda.synchronize()
    t0 = _us()
    _baton.checkpoint(os.getpid())
    _trace("mux", "snapshot", t0, _us() - t0)
    _checkpointed = True


def _restore_gpu():
    global _checkpointed
    if _checkpointed:
        t0 = _us()
        _baton.restore_and_unlock(os.getpid())
        _trace("mux", "restore", t0, _us() - t0)
        _checkpointed = False


def _yield_and_wait(check_fn):
    """Release token (snapshots GPU), wait, reacquire (restores GPU)."""
    if _held:
        _release()
    while not check_fn():
        time.sleep(1e-5)
    _acquire()


# ---- Helpers ----


def _completed_work():
    from torch._C._distributed_c10d import _create_work_from_future
    from torch.futures import Future

    fut = Future()
    fut.set_result(None)
    return _create_work_from_future(fut)


_coll_dir = None  # shared tempdir for collective data


def _coll_path(pg_id, coll_id, filename):
    return os.path.join(_coll_dir, f"pg{pg_id}", f"c{coll_id:08d}", filename)


def _save_tensor(path, tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(tensor.detach().cpu(), tmp)
    os.rename(tmp, path)


def _load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True)


# ---- File-based ProcessGroup ----

from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    ReduceOp,
    ReduceScatterOptions,
    ScatterOptions,
)


class _MuxPG(dist.ProcessGroup):
    """Resolves collectives by bookkeeping tensors on disk.

    Each collective: save contribution → check if all ranks contributed →
    if yes, resolve on CPU and continue; if no, snapshot GPU to disk,
    yield the GPU, and wait. When rescheduled, restore GPU state and
    load the collective result.
    """

    _next_pg_id = 0

    def __init__(self, rank, world_size):
        super().__init__(rank, world_size)
        self._rank = rank
        self._ws = world_size
        self._pg_id = _MuxPG._next_pg_id
        _MuxPG._next_pg_id += 1
        self._coll_id = 0

    def _next_coll(self):
        cid = self._coll_id
        self._coll_id += 1
        return cid

    def _ready_path(self, cid, rank):
        return _coll_path(self._pg_id, cid, f"ready_{rank}")

    def _resolved_path(self, cid):
        return _coll_path(self._pg_id, cid, "resolved")

    def _mark_ready(self, cid):
        p = self._ready_path(cid, self._rank)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").close()

    def _all_ready(self, cid):
        return all(os.path.exists(self._ready_path(cid, r)) for r in range(self._ws))

    def _is_resolved(self, cid):
        return os.path.exists(self._resolved_path(cid))

    def _mark_resolved(self, cid):
        open(self._resolved_path(cid), "w").close()

    def _wait_for_resolved(self, cid):
        _yield_and_wait(lambda: self._is_resolved(cid))

    def _in_path(self, cid, rank, idx):
        return _coll_path(self._pg_id, cid, f"in_r{rank}_t{idx}.pt")

    def _out_path(self, cid, rank, idx):
        return _coll_path(self._pg_id, cid, f"out_r{rank}_t{idx}.pt")

    def _do_collective(self, name, fn):
        """End compute span, run collective fn, begin new compute span."""
        _end_compute()
        t0 = _us()
        with torch.profiler.record_function(f"torchmux::collective::{name}"):
            result = fn()
        _trace("collective", name, t0, _us() - t0)
        _begin_compute()
        return result

    # ---- collectives ----

    @torch.no_grad()
    def allreduce(self, tensor_list, opts=AllreduceOptions()):
        def _run():
            cid = self._next_coll()
            op = opts.reduceOp if hasattr(opts, "reduceOp") else ReduceOp.SUM
            for i, t in enumerate(tensor_list):
                _save_tensor(self._in_path(cid, self._rank, i), t)
            self._mark_ready(cid)
            if self._all_ready(cid):
                for i in range(len(tensor_list)):
                    acc = _load_tensor(self._in_path(cid, 0, i))
                    for r in range(1, self._ws):
                        acc = acc + _load_tensor(self._in_path(cid, r, i))
                    if op == ReduceOp.AVG:
                        acc = acc / self._ws
                    for r in range(self._ws):
                        _save_tensor(self._out_path(cid, r, i), acc)
                self._mark_resolved(cid)
            else:
                self._wait_for_resolved(cid)
            for i, t in enumerate(tensor_list):
                t.copy_(_load_tensor(self._out_path(cid, self._rank, i)).to(t.device))
            return _completed_work()

        return self._do_collective("allreduce", _run)

    @torch.no_grad()
    def allreduce_coalesced(self, tensor_list, opts=AllreduceOptions()):
        return self.allreduce(tensor_list, opts)

    @torch.no_grad()
    def broadcast(self, tensor_list, opts=BroadcastOptions()):
        def _run():
            cid = self._next_coll()
            root = opts.rootRank
            if self._rank == root:
                for i, t in enumerate(tensor_list):
                    _save_tensor(self._in_path(cid, root, i), t)
            self._mark_ready(cid)
            if self._all_ready(cid):
                self._mark_resolved(cid)
            else:
                self._wait_for_resolved(cid)
            if self._rank != root:
                for i, t in enumerate(tensor_list):
                    t.copy_(_load_tensor(self._in_path(cid, root, i)).to(t.device))
            return _completed_work()

        return self._do_collective("broadcast", _run)

    @torch.no_grad()
    def allgather(self, output_tensors, input_tensor, opts=AllgatherOptions()):
        def _run():
            cid = self._next_coll()
            for i, t in enumerate(input_tensor):
                _save_tensor(self._in_path(cid, self._rank, i), t)
            self._mark_ready(cid)
            if self._all_ready(cid):
                self._mark_resolved(cid)
            else:
                self._wait_for_resolved(cid)
            for i, out_list in enumerate(output_tensors):
                for r in range(self._ws):
                    out_list[r].copy_(
                        _load_tensor(self._in_path(cid, r, i)).to(out_list[r].device)
                    )
            return _completed_work()

        return self._do_collective("allgather", _run)

    @torch.no_grad()
    def _allgather_base(self, output, input, opts=AllgatherOptions()):
        def _run():
            cid = self._next_coll()
            _save_tensor(self._in_path(cid, self._rank, 0), input)
            self._mark_ready(cid)
            if self._all_ready(cid):
                chunks = [
                    _load_tensor(self._in_path(cid, r, 0)) for r in range(self._ws)
                ]
                _save_tensor(self._out_path(cid, 0, 0), torch.cat(chunks, dim=0))
                self._mark_resolved(cid)
            else:
                self._wait_for_resolved(cid)
            output.copy_(_load_tensor(self._out_path(cid, 0, 0)).to(output.device))
            return _completed_work()

        return self._do_collective("allgather", _run)

    @torch.no_grad()
    def allgather_into_tensor_coalesced(self, outputs, inputs, opts=AllgatherOptions()):
        for o, i in zip(outputs, inputs):
            self._allgather_base(o, i, opts)
        return _completed_work()

    @torch.no_grad()
    def _reduce_scatter_base(self, output, input, opts=ReduceScatterOptions()):
        def _run():
            cid = self._next_coll()
            op = opts.reduceOp if hasattr(opts, "reduceOp") else ReduceOp.SUM
            _save_tensor(self._in_path(cid, self._rank, 0), input)
            self._mark_ready(cid)
            if self._all_ready(cid):
                acc = _load_tensor(self._in_path(cid, 0, 0))
                for r in range(1, self._ws):
                    acc = acc + _load_tensor(self._in_path(cid, r, 0))
                if op == ReduceOp.AVG:
                    acc = acc / self._ws
                chunk_size = acc.size(0) // self._ws
                for r in range(self._ws):
                    _save_tensor(
                        self._out_path(cid, r, 0),
                        acc[r * chunk_size : (r + 1) * chunk_size].contiguous(),
                    )
                self._mark_resolved(cid)
            else:
                self._wait_for_resolved(cid)
            output.copy_(
                _load_tensor(self._out_path(cid, self._rank, 0)).to(output.device)
            )
            return _completed_work()

        return self._do_collective("reduce_scatter", _run)

    @torch.no_grad()
    def reduce_scatter(self, output_tensor, scatter_list, opts=ReduceScatterOptions()):
        def _run():
            cid = self._next_coll()
            for i, chunks in enumerate(scatter_list):
                for r, chunk in enumerate(chunks):
                    _save_tensor(
                        _coll_path(self._pg_id, cid, f"in_r{self._rank}_s{i}_c{r}.pt"),
                        chunk,
                    )
            self._mark_ready(cid)
            if self._all_ready(cid):
                for dst in range(self._ws):
                    for i in range(len(output_tensor)):
                        acc = _load_tensor(
                            _coll_path(self._pg_id, cid, f"in_r0_s{i}_c{dst}.pt")
                        ).clone()
                        for src in range(1, self._ws):
                            acc.add_(
                                _load_tensor(
                                    _coll_path(
                                        self._pg_id, cid, f"in_r{src}_s{i}_c{dst}.pt"
                                    )
                                )
                            )
                        _save_tensor(self._out_path(cid, dst, i), acc)
                self._mark_resolved(cid)
            else:
                self._wait_for_resolved(cid)
            for i, t in enumerate(output_tensor):
                t.copy_(_load_tensor(self._out_path(cid, self._rank, i)).to(t.device))
            return _completed_work()

        return self._do_collective("reduce_scatter", _run)

    @torch.no_grad()
    def reduce_scatter_tensor_coalesced(
        self, outputs, inputs, opts=ReduceScatterOptions()
    ):
        for o, i in zip(outputs, inputs):
            self._reduce_scatter_base(o, i, opts)
        return _completed_work()

    @torch.no_grad()
    def scatter(self, output, input, opts=ScatterOptions()):
        raise NotImplementedError("_MuxPG.scatter")

    @torch.no_grad()
    def gather(self, output, input, opts=ScatterOptions()):
        raise NotImplementedError("_MuxPG.gather")

    @torch.no_grad()
    def alltoall(self, output, input, opts=AllToAllOptions()):
        raise NotImplementedError("_MuxPG.alltoall")

    @torch.no_grad()
    def barrier(self, opts=BarrierOptions()):
        def _run():
            cid = self._next_coll()
            self._mark_ready(cid)
            if not self._all_ready(cid):
                self._wait_for_resolved(cid)
            else:
                self._mark_resolved(cid)
            return _completed_work()

        return self._do_collective("barrier", _run)

    # ---- metadata ----

    def size(self):
        return self._ws

    def getBackendName(self):
        return "mux_files"

    @property
    def pg_name(self):
        world = dist.distributed_c10d._world
        from torch.testing._internal.distributed.multi_threaded_pg import (
            ThreadLocalWorld,
        )

        if isinstance(world, ThreadLocalWorld):
            world = world._get_world()
        return world.pg_names.get(self, f"mux_pg_{self._pg_id}")

    @property
    def group_name(self):
        return self.pg_name

    def __repr__(self):
        return f"MuxFiles(rank={self._rank}, size={self._ws})"


# ---- Backend factory ----


def _create_mux_pg(store, rank, world_size, timeout):
    from torch.distributed.distributed_c10d import _store_based_barrier

    pg = _MuxPG(rank, world_size)
    if _held:
        _release(snapshot=False)
    _store_based_barrier(rank, store, "", world_size, timeout)
    _acquire(restore=False)
    return pg


# ---- Worker process ----


def _worker(
    rank,
    world_size,
    ngpus,
    sched_next,
    master_port,
    coll_dir_path,
    script,
    script_args,
    run_as_module,
):
    global _ngpus, _sched, _rank, _ws, _held, _coll_dir, _baton
    _ngpus = ngpus
    _sched = sched_next
    _rank = rank
    _ws = world_size
    _coll_dir = coll_dir_path

    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank % ngpus),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "GROUP_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(ngpus)),
        }
    )

    torch.device = _MuxDevice
    _orig_cuda_set_device = torch.cuda.set_device
    torch.cuda.set_device = lambda *a, **kw: _orig_cuda_set_device(rank % ngpus)

    import ctypes

    ctypes.CDLL("libcuda.so.1").cuInit(0)

    from torch.distributed.checkpoint.baton import CudaBaton

    _baton = CudaBaton()

    dist.Backend.register_backend("mux_files", _create_mux_pg, devices=["cpu", "cuda"])

    _orig_init = dist.init_process_group
    _orig_destroy = dist.destroy_process_group
    _orig_new_group = dist.new_group

    def _mux_init(backend=None, **kwargs):
        _end_compute()
        t0 = _us()
        if _held:
            _release(snapshot=False)
        _orig_init(backend="mux_files", **kwargs)
        _acquire(restore=False)
        _trace("collective", "init", t0, _us() - t0)
        _begin_compute()

    def _mux_destroy():
        _end_compute()
        t0 = _us()
        if _held:
            _release(snapshot=False)
        try:
            _orig_destroy()
        finally:
            _acquire(restore=False)
            _trace("collective", "destroy", t0, _us() - t0)
            _begin_compute()

    def _mux_new_group(*args, **kwargs):
        _end_compute()
        t0 = _us()
        if _held:
            _release(snapshot=False)
        try:
            return _orig_new_group(*args, **kwargs)
        finally:
            _acquire(restore=False)
            _trace("collective", "new_group", t0, _us() - t0)
            _begin_compute()

    dist.init_process_group = _mux_init
    dist.destroy_process_group = _mux_destroy
    dist.new_group = _mux_new_group

    trace_dir = os.environ.get("TORCHMUX_TRACE_DIR", "/tmp")
    os.makedirs(trace_dir, exist_ok=True)

    _acquire()
    _begin_compute()
    try:
        sys.argv = [script] + list(script_args)
        if run_as_module:
            runpy.run_module(script, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(script, run_name="__main__")
    finally:
        _end_compute()
        if _held:
            _release()

        synthetic_path = os.path.join(coll_dir_path, f"trace_rank{rank}.json")
        with open(synthetic_path, "w") as f:
            json.dump(_trace_events, f)


# ---- Entry point ----


def main():
    parser = argparse.ArgumentParser(
        prog="torchmux",
        description="Simulate N-GPU distributed training on M GPUs",
    )
    parser.add_argument(
        "--nproc", type=int, required=True, help="Number of simulated workers"
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="Number of physical GPUs (default 1)",
    )
    parser.add_argument(
        "-m", action="store_true", dest="module", help="Run script as a Python module"
    )
    parser.add_argument("script", help="Training script or module name")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    nproc = args.nproc
    ngpus = args.ngpus
    assert nproc >= 1
    assert ngpus >= 1

    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    sched = _SharedInt()
    coll_dir = tempfile.mkdtemp(prefix="torchmux_colls_")

    gpu_desc = "GPU 0" if ngpus == 1 else f"{ngpus} GPUs"
    print(
        f"torchmux: {nproc} workers on {gpu_desc}, script={args.script}",
        flush=True,
    )

    try:
        mp.spawn(
            _worker,
            args=(
                nproc,
                ngpus,
                sched,
                port,
                coll_dir,
                args.script,
                args.script_args,
                args.module,
            ),
            nprocs=nproc,
            join=True,
        )
    finally:
        from torch.distributed import torchmux_trace

        trace_dir = os.environ.get("TORCHMUX_TRACE_DIR", "/tmp")

        events_by_rank = {}
        for r in range(nproc):
            p = os.path.join(coll_dir, f"trace_rank{r}.json")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                try:
                    with open(p) as f:
                        events_by_rank[r] = [tuple(e) for e in json.load(f)]
                except (json.JSONDecodeError, ValueError):
                    pass

        if events_by_rank:
            natural_path = os.path.join(trace_dir, "torchmux_natural.json")
            synthetic_path = os.path.join(trace_dir, "torchmux_synthetic.json")

            torchmux_trace.export_natural(events_by_rank, natural_path, nproc)
            torchmux_trace.export_synthetic(events_by_rank, synthetic_path, nproc)

            print(
                f"torchmux: traces written to {natural_path} and {synthetic_path}",
                flush=True,
            )

        sched.cleanup()
        shutil.rmtree(coll_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
