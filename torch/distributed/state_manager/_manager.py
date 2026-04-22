"""
WorkerStateManager: snapshot/restore all GPU tensor state per worker.

No explicit tensor registration is needed. snapshot() walks the garbage
collector to find every live CUDA tensor, copies its data to CPU, and
keys the snapshot by the tensor's GPU address (data_ptr). restore()
walks gc again and writes data back to tensors at matching addresses.

This works because the cooperative scheduler ensures only one worker
runs between snapshot and restore, and GPU allocations are never freed
across workers (so addresses stay stable and unique per worker).

Not yet memory-efficient: all workers' allocations coexist on GPU.
True memory savings (freeing inactive workers' GPU memory) requires
cuMemAddressReserve/cuMemMap for address-stable reallocation — future
work.
"""

import gc

import torch


class WorkerStateManager:
    def __init__(self):
        self._snapshots = {}

    def snapshot(self, rank):
        saved = []
        for obj in gc.get_objects():
            if not isinstance(obj, torch.Tensor):
                continue
            if not obj.is_cuda or obj.numel() == 0:
                continue
            try:
                saved.append((obj.data_ptr(), obj.shape, obj.data.cpu().clone()))
            except Exception:
                pass
        self._snapshots[rank] = saved

    def restore(self, rank):
        saved = self._snapshots.get(rank)
        if not saved:
            return
        ptr_to_data = {ptr: data for ptr, _, data in saved}
        for obj in gc.get_objects():
            if not isinstance(obj, torch.Tensor):
                continue
            if not obj.is_cuda or obj.numel() == 0:
                continue
            try:
                ptr = obj.data_ptr()
                if ptr in ptr_to_data:
                    obj.data.copy_(ptr_to_data[ptr])
            except Exception:
                pass

    def has_snapshot(self, rank):
        return rank in self._snapshots

    def clear(self, rank=None):
        if rank is not None:
            self._snapshots.pop(rank, None)
        else:
            self._snapshots.clear()
