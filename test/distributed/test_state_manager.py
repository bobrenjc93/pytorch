"""Tests for torch.distributed.state_manager.WorkerStateManager."""

import gc
import os
import tempfile

import torch
from torch.distributed.state_manager import WorkerStateManager
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestWorkerStateManager(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.manager = WorkerStateManager()

    def tearDown(self):
        self.manager.cleanup()
        super().tearDown()

    def test_basic_snapshot_restore(self):
        t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        expected = t.cpu().clone()

        self.manager.snapshot(0)
        self.manager.restore(0)

        self.assertEqual(t.cpu(), expected)

    def test_gpu_memory_freed_after_snapshot(self):
        """storage.resize_(0) returns GPU memory to the caching allocator."""
        t = torch.randn(1000, 1000, device="cuda")
        torch.cuda.synchronize()
        alloc_before = torch.cuda.memory_allocated()

        self.manager.snapshot(0)
        torch.cuda.synchronize()
        alloc_after = torch.cuda.memory_allocated()

        self.assertLess(alloc_after, alloc_before)

        self.manager.restore(0)

    def test_snapshot_writes_to_disk(self):
        t = torch.tensor([42.0], device="cuda")
        self.manager.snapshot(0)

        snap = self.manager._snapshots[0]
        self.assertGreater(len(snap.storages), 0)
        for record in snap.storages.values():
            self.assertTrue(os.path.isfile(record.filepath))

        self.manager.restore(0)

    def test_multiple_tensors(self):
        t1 = torch.tensor([1.0, 2.0], device="cuda")
        t2 = torch.tensor([3.0, 4.0, 5.0], device="cuda")
        t3 = torch.tensor([[6.0, 7.0], [8.0, 9.0]], device="cuda")

        e1, e2, e3 = t1.cpu().clone(), t2.cpu().clone(), t3.cpu().clone()

        self.manager.snapshot(0)
        self.manager.restore(0)

        self.assertEqual(t1.cpu(), e1)
        self.assertEqual(t2.cpu(), e2)
        self.assertEqual(t3.cpu(), e3)

    def test_multiple_ranks_separate_tensors(self):
        """Each rank's tensors are independently snapshotted and restored."""
        t0 = torch.tensor([1.0], device="cuda")
        e0 = t0.cpu().clone()
        self.manager.snapshot(0)

        t1 = torch.tensor([10.0], device="cuda")
        e1 = t1.cpu().clone()
        self.manager.snapshot(1)

        self.manager.restore(0)
        self.assertEqual(t0.cpu(), e0)

        self.manager.restore(1)
        self.assertEqual(t1.cpu(), e1)

    def test_view_preservation(self):
        """Views sharing storage should still share storage after restore."""
        base = torch.randn(4, 4, device="cuda")
        view = base[1:3, :]

        original_base = base.cpu().clone()
        original_view = view.cpu().clone()

        self.manager.snapshot(0)
        self.manager.restore(0)

        self.assertEqual(base.cpu(), original_base)
        self.assertEqual(view.cpu(), original_view)

        base[1, 0] = 999.0
        self.assertEqual(view[0, 0].item(), 999.0)

    def test_non_contiguous_tensor(self):
        """Transposed tensors preserve their stride after restore."""
        t = torch.randn(3, 4, device="cuda")
        t_transposed = t.t()
        original = t_transposed.cpu().clone()
        original_stride = t_transposed.stride()

        self.manager.snapshot(0)
        self.manager.restore(0)

        self.assertEqual(t_transposed.cpu(), original)
        self.assertEqual(t_transposed.stride(), original_stride)

    def test_has_snapshot(self):
        self.assertFalse(self.manager.has_snapshot(0))
        t = torch.tensor([1.0], device="cuda")
        self.manager.snapshot(0)
        self.assertTrue(self.manager.has_snapshot(0))
        self.assertFalse(self.manager.has_snapshot(1))
        self.manager.restore(0)

    def test_clear_single_rank(self):
        t = torch.tensor([1.0], device="cuda")
        self.manager.snapshot(0)

        t2 = torch.tensor([2.0], device="cuda")
        self.manager.snapshot(1)

        self.manager.clear(0)
        self.assertFalse(self.manager.has_snapshot(0))
        self.assertTrue(self.manager.has_snapshot(1))
        self.manager.restore(1)

    def test_clear_all(self):
        t = torch.tensor([1.0], device="cuda")
        self.manager.snapshot(0)

        t2 = torch.tensor([2.0], device="cuda")
        self.manager.snapshot(1)

        self.manager.clear()
        self.assertFalse(self.manager.has_snapshot(0))
        self.assertFalse(self.manager.has_snapshot(1))

    def test_clear_removes_files(self):
        t = torch.tensor([1.0], device="cuda")
        self.manager.snapshot(0)

        snap = self.manager._snapshots[0]
        files = [r.filepath for r in snap.storages.values()]
        for f in files:
            self.assertTrue(os.path.isfile(f))

        self.manager.clear(0)
        for f in files:
            self.assertFalse(os.path.isfile(f))

    def test_restore_nonexistent_rank_is_noop(self):
        t = torch.tensor([42.0], device="cuda")
        self.manager.restore(99)
        self.assertEqual(t, torch.tensor([42.0], device="cuda"))

    def test_snapshot_overwrites_previous(self):
        t = torch.tensor([1.0], device="cuda")
        self.manager.snapshot(0)
        self.manager.restore(0)

        t.fill_(2.0)
        self.manager.snapshot(0)
        self.assertTrue(self.manager.has_snapshot(0))
        self.manager.restore(0)
        self.assertEqual(t, torch.tensor([2.0], device="cuda"))

    @parametrize(
        "dtype",
        [torch.float32, torch.float64, torch.float16, torch.int32, torch.int64],
    )
    def test_various_dtypes(self, dtype):
        if dtype.is_floating_point:
            t = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=dtype)
        else:
            t = torch.tensor([1, 2, 3], device="cuda", dtype=dtype)

        expected = t.cpu().clone()
        self.manager.snapshot(0)
        self.manager.restore(0)
        self.assertEqual(t.cpu(), expected)

    def test_multidimensional_tensors(self):
        t = torch.randn(3, 4, 5, device="cuda")
        expected = t.cpu().clone()

        self.manager.snapshot(0)
        self.manager.restore(0)

        self.assertEqual(t.cpu(), expected)

    def test_large_tensor(self):
        t = torch.randn(1000, 1000, device="cuda")
        expected = t.cpu().clone()

        self.manager.snapshot(0)
        self.manager.restore(0)

        self.assertEqual(t.cpu(), expected)

    def test_custom_snapshot_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = WorkerStateManager(snapshot_dir=tmpdir)
            t = torch.tensor([1.0, 2.0], device="cuda")
            expected = t.cpu().clone()

            mgr.snapshot(0)

            snap = mgr._snapshots[0]
            for record in snap.storages.values():
                self.assertTrue(record.filepath.startswith(tmpdir))
                self.assertTrue(os.path.isfile(record.filepath))

            mgr.restore(0)
            self.assertEqual(t.cpu(), expected)
            mgr.cleanup()

    def test_cleanup_removes_directory(self):
        mgr = WorkerStateManager()
        snapshot_dir = mgr._snapshot_dir
        t = torch.tensor([1.0], device="cuda")
        mgr.snapshot(0)
        self.assertTrue(os.path.isdir(snapshot_dir))

        mgr.cleanup()
        self.assertFalse(os.path.isdir(snapshot_dir))

    def test_repeated_snapshot_restore_cycles(self):
        t = torch.tensor([1.0], device="cuda")

        for i in range(5):
            t.fill_(float(i))
            expected = t.cpu().clone()
            self.manager.snapshot(0)
            self.manager.restore(0)
            self.assertEqual(t.cpu(), expected)

    def test_zero_numel_tensor_ignored(self):
        t_empty = torch.empty(0, device="cuda")
        t_real = torch.tensor([1.0], device="cuda")
        expected = t_real.cpu().clone()

        self.manager.snapshot(0)
        self.manager.restore(0)
        self.assertEqual(t_real.cpu(), expected)

    def test_cpu_tensor_ignored(self):
        t_cpu = torch.tensor([1.0])
        t_gpu = torch.tensor([2.0], device="cuda")
        expected = t_gpu.cpu().clone()

        self.manager.snapshot(0)
        self.manager.restore(0)
        self.assertEqual(t_gpu.cpu(), expected)

    def test_weakref_dead_tensor_skipped(self):
        """If a tensor is GC'd before restore, it's safely skipped."""
        t1 = torch.tensor([1.0], device="cuda")
        t2 = torch.tensor([2.0], device="cuda")
        expected = t2.cpu().clone()

        self.manager.snapshot(0)

        del t1
        gc.collect()

        self.manager.restore(0)
        self.assertEqual(t2.cpu(), expected)

    def test_storage_deduplication(self):
        """Views of the same storage are deduplicated and share storage after restore."""
        base = torch.randn(100, device="cuda")
        view1 = base[:50]
        view2 = base[50:]
        expected_base = base.cpu().clone()

        self.manager.snapshot(0)
        self.manager.restore(0)

        self.assertEqual(base.cpu(), expected_base)
        self.assertEqual(view1.cpu(), expected_base[:50])
        self.assertEqual(view2.cpu(), expected_base[50:])

        base[0] = 777.0
        self.assertEqual(view1[0].item(), 777.0)
        base[50] = 888.0
        self.assertEqual(view2[0].item(), 888.0)

    def test_tensor_usable_after_snapshot_restore(self):
        """Tensor is fully functional after restore — can compute, grad, etc."""
        t = torch.randn(3, 3, device="cuda", requires_grad=True)
        expected = t.detach().cpu().clone()

        self.manager.snapshot(0)
        self.manager.restore(0)

        self.assertEqual(t.detach().cpu(), expected)
        loss = t.sum()
        loss.backward()
        self.assertIsNotNone(t.grad)


if __name__ == "__main__":
    run_tests()
