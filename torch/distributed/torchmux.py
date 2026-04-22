"""
torchmux: Run N distributed workers on 1 GPU in a single process.

Usage:
    python -m torch.distributed.torchmux --nproc 8 train.py [args...]

Takes a standard torchrun-compatible training script and runs it with
N workers as threads, all sharing GPU 0. Three things are monkey-patched
so existing scripts work without modification:

  1. os.environ — RANK, LOCAL_RANK, WORLD_SIZE return per-thread values
  2. dist.init_process_group — uses the vnccl backend (local collectives)
  3. torch.cuda.set_device — pins all workers to GPU 0
  4. dist.all_reduce et al. — yield the execution lock at collective
     boundaries so the next worker can run

Only one worker executes at a time. Workers yield to each other at
collective boundaries (all_reduce, all_gather, broadcast, etc.). This
cooperative scheduling means:
  - N workers share 1 GPU's worth of memory
  - The global RNG is never corrupted by thread interleaving
  - Collective numerics are bitwise identical to real NCCL + torchrun

Requires scripts to use LOCAL_RANK (not RANK) for device selection.
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

_exec_lock = threading.Lock()


class _ThreadLocalEnv:
    """
    Proxy for os.environ that returns per-thread values for
    RANK, LOCAL_RANK, WORLD_SIZE, and related keys.
    """

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
    """
    Wrap a dist collective so the calling worker releases the execution
    lock before entering and reacquires it after. This is how workers
    yield to each other: worker A runs alone until it hits a collective,
    releases the lock, blocks inside the collective waiting for the
    others, and worker B picks up the lock and runs until it also hits
    the collective — at which point the collective resolves and one
    worker reacquires the lock.
    """

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


def _worker(rank, world_size, script, script_args, store, ready_barrier, errors):
    try:
        _tls.env_overrides = {
            "RANK": str(rank),
            "LOCAL_RANK": "0",
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

        _exec_lock.acquire()
        try:
            sys.argv = [script] + list(script_args)
            runpy.run_path(script, run_name="__main__")
        finally:
            _exec_lock.release()
    except SystemExit:
        pass
    except Exception:
        errors[rank] = traceback.format_exc()


def main():
    parser = argparse.ArgumentParser(
        prog="torchmux",
        description="Run N distributed workers on 1 GPU in a single process",
    )
    parser.add_argument("--nproc", type=int, required=True, help="Number of workers")
    parser.add_argument("script", help="Training script (torchrun-compatible)")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    nproc = args.nproc
    assert nproc >= 1

    # -- set up thread-local distributed world --

    from torch.testing._internal.distributed.multi_threaded_pg import (
        _install_threaded_pg,
    )

    _install_threaded_pg()
    torch._C._distributed_c10d._set_thread_isolation_mode(True)

    from torch.distributed import vnccl as _vnccl  # noqa: F401,F811

    store = dist.HashStore()

    # -- monkey-patches --

    _orig_init_pg = dist.init_process_group
    _orig_destroy_pg = dist.destroy_process_group
    _orig_cuda_set_device = torch.cuda.set_device

    def _patched_init_pg(backend=None, init_method=None, **kwargs):
        kwargs.pop("timeout", None)
        _exec_lock.release()
        try:
            _orig_init_pg(
                backend="vnccl",
                rank=_tls.rank,
                world_size=_tls.world_size,
                store=_tls.store,
            )
        finally:
            _exec_lock.acquire()

    def _patched_destroy_pg():
        _exec_lock.release()
        try:
            _orig_destroy_pg()
        finally:
            _exec_lock.acquire()

    dist.init_process_group = _patched_init_pg
    dist.destroy_process_group = _patched_destroy_pg
    torch.cuda.set_device = lambda *a, **kw: _orig_cuda_set_device(0)
    os.environ = _ThreadLocalEnv(os.environ)

    for name in _COLLECTIVES:
        if hasattr(dist, name):
            setattr(dist, name, _yield_at_collective(getattr(dist, name)))

    # -- launch workers --

    print(f"torchmux: {nproc} workers on GPU 0, script={args.script}", flush=True)

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

    # -- restore --

    dist.init_process_group = _orig_init_pg
    dist.destroy_process_group = _orig_destroy_pg
    torch.cuda.set_device = _orig_cuda_set_device
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
