"""
GPU baton: checkpoint and restore a process's entire CUDA context.

Wraps cuCheckpointProcess{Lock,Checkpoint,Restore,Unlock} via ctypes
so a process can release its entire CUDA context (VRAM returned to the
driver) and later restore it at the same virtual addresses.

Usage from a coordinator process::

    baton = CudaBaton()
    baton.checkpoint(worker_pid)  # worker's VRAM freed
    baton.restore(worker_pid)  # worker's VRAM restored

The target process must have an active CUDA context. After checkpoint,
the process is in CHECKPOINTED state and cannot make CUDA calls.
After restore, it is back in LOCKED state; call unlock to resume.
"""

import ctypes
import logging
import threading

log = logging.getLogger(__name__)

_cuda = None
_cuda_lock = threading.Lock()
_cuda_init_lock = threading.Lock()
_cuda_initialized = False


# Checkpoint process state constants returned by get_state().
CHECKPOINT_STATE_INVALID = 0
CHECKPOINT_STATE_ACTIVE = 1
CHECKPOINT_STATE_LOCKED = 2
CHECKPOINT_STATE_CHECKPOINTED = 3


class _LockArgs(ctypes.Structure):
    _fields_ = [
        ("timeoutMs", ctypes.c_uint),
        ("reserved0", ctypes.c_uint),
        ("reserved1", ctypes.c_uint64 * 7),
    ]


class _CheckpointArgs(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_uint64 * 8)]


class _RestoreArgs(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_uint64 * 8)]


class _UnlockArgs(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_uint64 * 8)]


def _bind_cuda_symbol(lib, name: str, argtypes):
    try:
        fn = getattr(lib, name)
    except AttributeError as e:
        raise RuntimeError(
            "torchmux requires CUDA driver checkpoint/restore support. "
            f"Your libcuda.so.1 is missing `{name}`."
        ) from e
    fn.restype = ctypes.c_int
    fn.argtypes = argtypes
    return fn


def _get_cuda():
    global _cuda
    if _cuda is not None:
        return _cuda
    with _cuda_lock:
        if _cuda is not None:
            return _cuda
        try:
            lib = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise RuntimeError(
                "torchmux requires the CUDA driver library (libcuda.so.1). "
                "Ensure CUDA drivers are installed and libcuda.so.1 is on "
                "the library search path."
            ) from e

        _ptr = ctypes.POINTER
        _c_int = ctypes.c_int

        _bind_cuda_symbol(lib, "cuInit", [ctypes.c_uint])
        _bind_cuda_symbol(lib, "cuGetErrorString", [_c_int, _ptr(ctypes.c_char_p)])
        _bind_cuda_symbol(
            lib, "cuCheckpointProcessLock", [_c_int, _ptr(_LockArgs)]
        )
        _bind_cuda_symbol(
            lib,
            "cuCheckpointProcessCheckpoint",
            [_c_int, _ptr(_CheckpointArgs)],
        )
        _bind_cuda_symbol(
            lib, "cuCheckpointProcessRestore", [_c_int, _ptr(_RestoreArgs)]
        )
        _bind_cuda_symbol(
            lib, "cuCheckpointProcessUnlock", [_c_int, _ptr(_UnlockArgs)]
        )
        _bind_cuda_symbol(
            lib, "cuCheckpointProcessGetState", [_c_int, _ptr(_c_int)]
        )

        _cuda = lib
    return _cuda


def _maybe_init_cuda():
    global _cuda_initialized
    if _cuda_initialized:
        return
    import torch

    if not torch.cuda.is_initialized():
        return
    cuda = _get_cuda()
    with _cuda_init_lock:
        if _cuda_initialized:
            return
        _check(cuda.cuInit(0), "cuInit")
        _cuda_initialized = True


def _check(result: int, name: str) -> None:
    if result != 0:
        cuda = _get_cuda()
        err = ctypes.c_char_p()
        cuda.cuGetErrorString(result, ctypes.byref(err))
        msg = err.value.decode() if err.value else f"code {result}"
        raise RuntimeError(f"{name} failed: {msg}")


class CudaBaton:
    """Checkpoint/restore a process's GPU state via the CUDA driver API.

    All methods target a remote process by PID. The caller must be in
    the same CUDA MPS or checkpoint-capable environment.
    """

    def lock(self, pid: int, timeout_ms: int = 30000) -> None:
        _maybe_init_cuda()
        cuda = _get_cuda()
        args = _LockArgs(timeoutMs=timeout_ms)
        _check(cuda.cuCheckpointProcessLock(pid, ctypes.byref(args)), "Lock")

    def checkpoint(self, pid: int, timeout_ms: int = 30000) -> None:
        """Lock + checkpoint: VRAM is freed, process cannot use CUDA."""
        self.lock(pid, timeout_ms)
        try:
            cuda = _get_cuda()
            args = _CheckpointArgs()
            _check(
                cuda.cuCheckpointProcessCheckpoint(pid, ctypes.byref(args)),
                "Checkpoint",
            )
        except Exception:
            try:
                self.unlock(pid)
            except Exception:
                log.error(
                    "CudaBaton: unlock failed after checkpoint failure for "
                    "pid %d; process may be stuck in LOCKED state",
                    pid,
                )
            raise

    def restore(self, pid: int) -> None:
        """Restore from last checkpoint (process enters LOCKED state)."""
        _maybe_init_cuda()
        cuda = _get_cuda()
        args = _RestoreArgs()
        _check(
            cuda.cuCheckpointProcessRestore(pid, ctypes.byref(args)),
            "Restore",
        )

    def unlock(self, pid: int) -> None:
        _maybe_init_cuda()
        cuda = _get_cuda()
        args = _UnlockArgs()
        _check(
            cuda.cuCheckpointProcessUnlock(pid, ctypes.byref(args)),
            "Unlock",
        )

    def restore_and_unlock(self, pid: int) -> None:
        """Restore + unlock: process can resume CUDA calls."""
        self.restore(pid)
        try:
            self.unlock(pid)
        except Exception:
            # Process is restored but stuck in LOCKED state. Log and
            # re-raise — the caller must handle this, otherwise the
            # process will deadlock on its next CUDA call.
            log.error(
                "CudaBaton: unlock failed after successful restore for pid %d; "
                "process is in LOCKED state and cannot make CUDA calls",
                pid,
            )
            raise

    def get_state(self, pid: int) -> int:
        _maybe_init_cuda()
        cuda = _get_cuda()
        state = ctypes.c_int()
        _check(
            cuda.cuCheckpointProcessGetState(pid, ctypes.byref(state)),
            "GetState",
        )
        return state.value
