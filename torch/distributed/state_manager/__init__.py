"""
Per-worker GPU state manager for torchmux.

Automatically discovers all live CUDA tensors via the garbage collector
and provides per-worker snapshot/restore. No explicit registration is
needed — every CUDA tensor is captured.
"""

from torch.distributed.state_manager._manager import WorkerStateManager

__all__ = ["WorkerStateManager"]
