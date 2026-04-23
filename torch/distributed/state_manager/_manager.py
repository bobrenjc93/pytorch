"""
WorkerStateManager: snapshot/restore all GPU tensor state per worker.

Walks the garbage collector to find every live CUDA tensor, groups them
by underlying storage (to handle views correctly), serializes raw storage
bytes to disk, and frees GPU memory via storage.resize_(0). On restore,
bytes are loaded from disk into fresh GPU allocations, and each tensor
is reconnected to its storage with the original offset, shape, and stride.

Tensor objects are tracked via weak references so the state manager does
not prevent garbage collection of tensors the worker has discarded.
"""

import ctypes
import gc
import os
import tempfile
import weakref
from dataclasses import dataclass, field

import torch


@dataclass
class _TensorRecord:
    ref: weakref.ref
    storage_key: int
    storage_offset: int
    shape: tuple
    stride: tuple
    dtype: torch.dtype


@dataclass
class _StorageRecord:
    filepath: str
    nbytes: int
    device_index: int


@dataclass
class _WorkerSnapshot:
    tensors: list[_TensorRecord] = field(default_factory=list)
    storages: dict[int, _StorageRecord] = field(default_factory=dict)


class WorkerStateManager:
    def __init__(self, snapshot_dir: str | None = None):
        self._snapshot_dir = snapshot_dir or tempfile.mkdtemp(
            prefix="torchmux_snapshots_"
        )
        self._snapshots: dict[int, _WorkerSnapshot] = {}

    def _rank_dir(self, rank: int) -> str:
        d = os.path.join(self._snapshot_dir, f"rank_{rank}")
        os.makedirs(d, exist_ok=True)
        return d

    def _save_storage(self, storage: torch.UntypedStorage, filepath: str) -> None:
        cpu_storage = storage.cpu()
        nbytes = cpu_storage.nbytes()
        buf = (ctypes.c_char * nbytes).from_address(cpu_storage.data_ptr())
        with open(filepath, "wb") as f:
            f.write(bytes(buf))

    def _load_storage(
        self, filepath: str, nbytes: int, device_index: int
    ) -> torch.UntypedStorage:
        with open(filepath, "rb") as f:
            raw = f.read()
        cpu_storage = torch.UntypedStorage(nbytes)
        ctypes.memmove(cpu_storage.data_ptr(), raw, nbytes)
        return cpu_storage.cuda(device_index)

    def snapshot(self, rank: int) -> None:
        self._clear_rank_files(rank)
        snap = _WorkerSnapshot()
        rank_dir = self._rank_dir(rank)

        seen_storages: dict[int, torch.UntypedStorage] = {}

        for obj in gc.get_objects():
            if not isinstance(obj, torch.Tensor):
                continue
            if not obj.is_cuda or obj.numel() == 0:
                continue
            try:
                s = obj.untyped_storage()
                if s.nbytes() == 0:
                    continue
                skey = s.data_ptr()
                if skey not in seen_storages:
                    idx = len(seen_storages)
                    snap.storages[skey] = _StorageRecord(
                        filepath=os.path.join(rank_dir, f"storage_{idx}.bin"),
                        nbytes=s.nbytes(),
                        device_index=s.device.index or 0,
                    )
                    seen_storages[skey] = s

                snap.tensors.append(
                    _TensorRecord(
                        ref=weakref.ref(obj),
                        storage_key=skey,
                        storage_offset=obj.storage_offset(),
                        shape=tuple(obj.shape),
                        stride=tuple(obj.stride()),
                        dtype=obj.dtype,
                    )
                )
            except Exception:
                pass

        for skey, storage in seen_storages.items():
            self._save_storage(storage, snap.storages[skey].filepath)

        for storage in seen_storages.values():
            storage.resize_(0)

        self._snapshots[rank] = snap

    def restore(self, rank: int) -> None:
        snap = self._snapshots.get(rank)
        if not snap:
            return

        gpu_storages: dict[int, torch.UntypedStorage] = {}
        for skey, record in snap.storages.items():
            gpu_storages[skey] = self._load_storage(
                record.filepath, record.nbytes, record.device_index
            )

        with torch.no_grad():
            for tr in snap.tensors:
                t = tr.ref()
                if t is None:
                    continue
                storage = gpu_storages.get(tr.storage_key)
                if storage is None:
                    continue
                t.set_(storage, tr.storage_offset, tr.shape, tr.stride)

    def has_snapshot(self, rank: int) -> bool:
        return rank in self._snapshots

    def _clear_rank_files(self, rank: int) -> None:
        snap = self._snapshots.pop(rank, None)
        if snap:
            for record in snap.storages.values():
                try:
                    os.remove(record.filepath)
                except OSError:
                    pass
            try:
                os.rmdir(self._rank_dir(rank))
            except OSError:
                pass

    def clear(self, rank: int | None = None) -> None:
        if rank is not None:
            self._clear_rank_files(rank)
        else:
            for r in list(self._snapshots.keys()):
                self._clear_rank_files(r)

    def cleanup(self) -> None:
        self.clear()
        try:
            os.rmdir(self._snapshot_dir)
        except OSError:
            pass
