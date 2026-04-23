"""
WorkerStateManager: snapshot/restore all GPU tensor state per worker.

Walks the garbage collector to find every live CUDA tensor, groups them
by underlying storage (to handle views correctly), serializes raw storage
bytes to disk, and frees GPU memory via storage.resize_(0). On restore,
the SAME storage objects are resized back and refilled from disk so that
all tensors sharing them — including those only referenced from C++ and
invisible to Python's GC — automatically see the restored data.
"""

import ctypes
import gc
import os
import tempfile
import weakref
from dataclasses import dataclass, field

import torch


@dataclass
class _StorageRecord:
    filepath: str
    nbytes: int
    device_index: int
    storage_ref: object  # strong ref to the original UntypedStorage


@dataclass
class _WorkerSnapshot:
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

    def _load_into_storage(
        self, storage: torch.UntypedStorage, filepath: str, nbytes: int
    ) -> None:
        """Resize the storage back and copy saved data into it."""
        storage.resize_(nbytes)
        with open(filepath, "rb") as f:
            raw = f.read()
        cpu_storage = torch.UntypedStorage(nbytes)
        ctypes.memmove(cpu_storage.data_ptr(), raw, nbytes)
        # Copy from CPU storage into the resized GPU storage via
        # uint8 tensor views so we get a proper D2H copy.
        dev = storage.device
        src = torch.empty(0, dtype=torch.uint8)
        src.set_(cpu_storage, 0, (nbytes,), (1,))
        dst = torch.empty(0, dtype=torch.uint8, device=dev)
        dst.set_(storage, 0, (nbytes,), (1,))
        dst.copy_(src)

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
                    seen_storages[skey] = s
            except Exception:
                pass

        for idx, (skey, s) in enumerate(seen_storages.items()):
            filepath = os.path.join(rank_dir, f"storage_{idx}.bin")
            self._save_storage(s, filepath)
            snap.storages[skey] = _StorageRecord(
                filepath=filepath,
                nbytes=s.nbytes(),
                device_index=s.device.index or 0,
                storage_ref=s,
            )

        for s in seen_storages.values():
            s.resize_(0)

        self._snapshots[rank] = snap

    def restore(self, rank: int) -> None:
        snap = self._snapshots.get(rank)
        if not snap:
            return

        for record in snap.storages.values():
            s = record.storage_ref
            self._load_into_storage(s, record.filepath, record.nbytes)

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
