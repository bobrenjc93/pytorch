"""
torchmux: Simulate N-GPU distributed training on M GPUs (M < N).

Usage:
    python -m torch.distributed.torchmux --nproc 8 train.py [args...]
    python -m torch.distributed.torchmux --nproc 8 --ngpus 2 train.py [args...]

Takes a standard torchrun-compatible training script and runs N workers
as cooperatively-scheduled threads in a single process. Workers are
mapped round-robin onto M physical GPUs (default M=1).

Scripts can use device=f"cuda:{rank}" and backend="nccl" — torchmux
remaps CUDA device indices to physical GPUs (cuda:{rank % ngpus}) and
redirects any backend to vnccl.

Only one worker executes at a time. Workers yield to each other at
collective boundaries (all_reduce, all_gather, broadcast, etc.). This
cooperative scheduling means:
  - The global RNG is never corrupted by thread interleaving
  - Collective numerics are bitwise identical to real NCCL + torchrun
"""

import argparse
import os
import runpy
import sys
import threading
import traceback

import torch
import torch.distributed as dist


_tls = threading.local()

_exec_lock = None
_ngpus = 1

# ---- torch.device remapping ----

_OrigDevice = torch.device


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


class _ThreadLocalEnv:
    def __init__(self, real_environ):
        object.__setattr__(self, "_real", real_environ)

    def _override(self, key):
        overrides = getattr(_tls, "env_overrides", None)
        if overrides is not None and key in overrides:
            return overrides[key]
        return None

    def __getitem__(self, key):
        v = self._override(key)
        return v if v is not None else self._real[key]

    def get(self, key, default=None):
        v = self._override(key)
        return v if v is not None else self._real.get(key, default)

    def __contains__(self, key):
        return self._override(key) is not None or key in self._real

    def __setitem__(self, key, value):
        self._real[key] = value

    def __delitem__(self, key):
        del self._real[key]

    def __iter__(self):
        return iter(self._real)

    def __len__(self):
        return len(self._real)

    def __repr__(self):
        return repr(self._real)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _yield_at_collective(fn):
    def wrapper(*args, **kwargs):
        _exec_lock.release()
        try:
            return fn(*args, **kwargs)
        finally:
            _exec_lock.acquire()

    wrapper.__name__ = getattr(fn, "__name__", "")
    wrapper.__qualname__ = getattr(fn, "__qualname__", "")
    return wrapper


_COLLECTIVES = [
    "broadcast",
    "all_reduce",
    "all_reduce_coalesced",
    "reduce",
    "all_gather",
    "all_gather_into_tensor",
    "all_gather_coalesced",
    "gather",
    "scatter",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "all_to_all",
    "all_to_all_single",
    "barrier",
    "batch_isend_irecv",
]


_run_as_module = False
_nproc_for_trace = 1


_traces_written = False


def _export_traces():
    global _traces_written
    if _traces_written:
        return
    _traces_written = True

    from torch.distributed import torchmux_trace

    trace_dir = os.environ.get("TORCHMUX_TRACE_DIR", "/tmp")
    natural_path = os.path.join(trace_dir, "torchmux_natural.json")
    synthetic_path = os.path.join(trace_dir, "torchmux_synthetic.json")
    torchmux_trace.export_natural(natural_path, _nproc_for_trace)
    torchmux_trace.export_synthetic(synthetic_path, _nproc_for_trace)
    print(f"torchmux: traces written to {natural_path} and {synthetic_path}", flush=True)


def _worker(rank, world_size, script, script_args, store, ready_barrier, errors):
    try:
        _tls.env_overrides = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank % _ngpus),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "GROUP_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
        _tls.rank = rank
        _tls.world_size = world_size
        _tls.store = store

        ready_barrier.wait()

        from torch.distributed.vnccl import _acquire_exec_lock_ordered

        _acquire_exec_lock_ordered(rank, world_size)
        try:
            from torch.distributed import torchmux_trace as _trace

            _tls._exec_start = _trace._us()
            sys.argv = [script] + list(script_args)
            if _run_as_module:
                runpy.run_module(script, run_name="__main__", alter_sys=True)
            else:
                runpy.run_path(script, run_name="__main__")
        finally:
            from torch.distributed.vnccl import _release_exec_lock_ordered

            _release_exec_lock_ordered(rank, world_size)
            if rank == 0:
                _export_traces()
    except SystemExit:
        pass
    except Exception:
        errors[rank] = traceback.format_exc()


def main():
    parser = argparse.ArgumentParser(
        prog="torchmux",
        description="Simulate N-GPU distributed training on M GPUs",
    )
    parser.add_argument("--nproc", type=int, required=True, help="Number of simulated workers (N)")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of physical GPUs (M, default 1)")
    parser.add_argument("-m", action="store_true", dest="module", help="Run script as a Python module (like python -m)")
    parser.add_argument("script", help="Training script or module name")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    nproc = args.nproc
    assert nproc >= 1
    assert args.ngpus >= 1 and args.ngpus <= nproc

    global _ngpus, _run_as_module
    _ngpus = args.ngpus
    _run_as_module = args.module

    from torch.testing._internal.distributed.multi_threaded_pg import (
        _install_threaded_pg,
    )

    _install_threaded_pg()
    torch._C._distributed_c10d._set_thread_isolation_mode(True)

    from torch.distributed import vnccl as _vnccl  # noqa: F401,F811

    global _exec_lock
    _exec_lock = _vnccl.exec_lock

    dist.Backend.default_device_backend_map["cuda"] = "vnccl"

    store = dist.HashStore()

    _orig_init_pg = dist.init_process_group
    _orig_destroy_pg = dist.destroy_process_group
    _orig_cuda_set_device = torch.cuda.set_device

    def _patched_init_pg(backend=None, init_method=None, **kwargs):
        kwargs.pop("timeout", None)
        from torch.distributed.vnccl import (
            _rng_states, _save_rng, _restore_rng,
            _release_exec_lock_ordered, _acquire_exec_lock_ordered,
        )

        rank = _tls.rank
        ws = _tls.world_size
        _rng_states[rank] = _save_rng()
        _release_exec_lock_ordered(rank, ws)
        try:
            _orig_init_pg(
                backend="vnccl",
                rank=rank,
                world_size=ws,
                store=_tls.store,
            )
        finally:
            _acquire_exec_lock_ordered(rank, ws)
            if rank in _rng_states:
                _restore_rng(_rng_states[rank])

    _traces_exported = [False]

    def _patched_destroy_pg():
        if not _traces_exported[0]:
            _traces_exported[0] = True
            _export_traces()

        from torch.distributed.vnccl import _rng_states, _save_rng, _restore_rng

        rank = getattr(_tls, "rank", None)
        if rank is not None:
            _rng_states[rank] = _save_rng()
        _exec_lock.release()
        try:
            _orig_destroy_pg()
        finally:
            _exec_lock.acquire()
            if rank is not None and rank in _rng_states:
                _restore_rng(_rng_states[rank])

    dist.init_process_group = _patched_init_pg
    dist.destroy_process_group = _patched_destroy_pg
    torch.cuda.set_device = lambda *a, **kw: _orig_cuda_set_device(
        getattr(_tls, "rank", 0) % _ngpus
    )
    torch.device = _MuxDevice
    os.environ = _ThreadLocalEnv(os.environ)

    # Cooperative scheduling is handled inside vnccl._do() — every
    # collective (whether called via dist.all_reduce or directly via
    # ProcessGroup.allreduce) releases the exec_lock before blocking
    # on the barrier and reacquires it after. No wrapping needed here.

    global _nproc_for_trace
    _nproc_for_trace = nproc

    gpu_desc = "GPU 0" if _ngpus == 1 else f"{_ngpus} GPUs"
    print(f"torchmux: {nproc} workers on {gpu_desc}, script={args.script}", flush=True)

    errors = [None] * nproc
    ready_barrier = threading.Barrier(nproc)
    threads = []
    for rank in range(nproc):
        t = threading.Thread(
            target=_worker,
            args=(rank, nproc, args.script, args.script_args, store, ready_barrier, errors),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    _export_traces()

    dist.init_process_group = _orig_init_pg
    dist.destroy_process_group = _orig_destroy_pg
    torch.cuda.set_device = _orig_cuda_set_device
    torch.device = _OrigDevice
    torch._C._distributed_c10d._set_thread_isolation_mode(False)

    failed = False
    for rank, err in enumerate(errors):
        if err is not None:
            print(f"\n[rank {rank}] FAILED:\n{err}", file=sys.stderr)
            failed = True
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
