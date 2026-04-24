"""Integration tests for torchmux and vnccl.

Tests cooperative scheduling trace ordering, collective correctness
under torchmux (file-based PG), and vnccl (thread-based PG).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests, TestCase


TRAIN_SCRIPT = """
import os
import torch
import torch.distributed as dist

dist.init_process_group()
rank = dist.get_rank()
device = f"cuda:{rank % int(os.environ.get('TORCHMUX_NGPUS', '1'))}"
torch.cuda.set_device(device)

# Two allreduces with compute in between
x = torch.randn(4, 4, device=device)
dist.all_reduce(x)
y = x @ x
dist.all_reduce(y)

dist.destroy_process_group()
"""


CORRECTNESS_SCRIPT = """
import json
import os
import sys
import torch
import torch.distributed as dist

dist.init_process_group()
rank = dist.get_rank()
ws = dist.get_world_size()
device = f"cuda:{rank % int(os.environ.get('TORCHMUX_NGPUS', '1'))}"
torch.cuda.set_device(device)

results = {}

# allreduce: each rank contributes (rank+1), expect sum = ws*(ws+1)/2
x = torch.full((4,), float(rank + 1), device=device)
dist.all_reduce(x)
expected_sum = ws * (ws + 1) / 2
results["allreduce"] = torch.allclose(x, torch.full_like(x, expected_sum))

# broadcast: root=0 sends tensor, others receive
y = torch.full((4,), 42.0, device=device) if rank == 0 else torch.zeros(4, device=device)
dist.broadcast(y, src=0)
results["broadcast"] = torch.allclose(y, torch.full_like(y, 42.0))

# allgather: each rank contributes rank, gather all
inp = torch.full((2,), float(rank), device=device)
gathered = [torch.zeros(2, device=device) for _ in range(ws)]
dist.all_gather(gathered, inp)
allgather_ok = all(
    torch.allclose(gathered[r], torch.full((2,), float(r), device=device))
    for r in range(ws)
)
results["allgather"] = allgather_ok

# reduce_scatter: each rank contributes full tensor, gets a chunk of the sum
full = torch.arange(ws * 2, dtype=torch.float32, device=device)
out = torch.zeros(2, device=device)
dist.reduce_scatter_tensor(out, full)
chunk = full[rank * 2 : (rank + 1) * 2] * ws
results["reduce_scatter"] = torch.allclose(out, chunk)

dist.destroy_process_group()

output_path = os.path.join(os.environ["TORCHMUX_RESULTS_DIR"], f"rank{rank}.json")
with open(output_path, "w") as f:
    json.dump({k: bool(v) for k, v in results.items()}, f)
"""


class TestTorchmuxTraceOrdering(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self._tmpdir = tempfile.mkdtemp()
        self._script = os.path.join(self._tmpdir, "train.py")
        with open(self._script, "w") as f:
            f.write(TRAIN_SCRIPT)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        super().tearDown()

    def _run_torchmux(self, nproc, script, env_extra=None):
        trace_dir = os.path.join(self._tmpdir, "traces")
        os.makedirs(trace_dir, exist_ok=True)
        env = os.environ.copy()
        env["TORCHMUX_TRACE_DIR"] = trace_dir
        env["TORCHMUX_NGPUS"] = "1"
        if env_extra:
            env.update(env_extra)
        result = subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.torchmux",
                "--nproc-per-node", str(nproc), script,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode, 0,
            f"torchmux failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )
        return trace_dir

    def test_3worker_trace_ordering(self):
        """Assert trace events follow cooperative scheduling invariants.

        Expected pattern per collective (3 workers):
          W0: compute -> collective_start -> snapshot
          W1: compute -> collective_start -> snapshot
          W2: compute -> collective_start -> collective_finish (resolves)
          W2: continues to next compute...
          W2: next_collective_start -> snapshot (yields for first time)
          W0: restore -> collective_finish -> compute
          ...
        """
        trace_dir = self._run_torchmux(3, self._script)

        natural_path = os.path.join(trace_dir, "torchmux_natural.json")
        self.assertTrue(os.path.exists(natural_path))

        with open(natural_path) as f:
            trace = json.load(f)

        events = [
            e for e in trace["traceEvents"] if e.get("ph") == "X"
        ]
        self.assertGreater(len(events), 0, "no trace events")

        per_worker = {}
        for e in events:
            per_worker.setdefault(e["pid"], []).append(e)
        for pid in per_worker:
            per_worker[pid].sort(key=lambda e: e["ts"])

        self.assertEqual(len(per_worker), 3, "expected 3 workers")

        # Global timeline: sort all events by start time
        timeline = sorted(events, key=lambda e: e["ts"])

        # -- Invariant 1: compute spans never overlap across workers --
        compute_spans = [
            (e["pid"], e["ts"], e["ts"] + e["dur"])
            for e in timeline
            if e["cat"] == "compute"
        ]
        for i in range(len(compute_spans)):
            for j in range(i + 1, len(compute_spans)):
                pi, si, ei = compute_spans[i]
                pj, sj, ej = compute_spans[j]
                if pi != pj:
                    self.assertFalse(
                        si < ej and sj < ei,
                        f"compute overlap: worker {pi} [{si}-{ei}] "
                        f"vs worker {pj} [{sj}-{ej}]",
                    )

        # -- Invariant 2: every snapshot is preceded by either a
        #    collective or a compute (never another snapshot) --
        for pid, worker_events in per_worker.items():
            for i, e in enumerate(worker_events):
                if e["cat"] == "mux" and e["name"] == "snapshot":
                    if i > 0:
                        prev = worker_events[i - 1]["cat"]
                        self.assertIn(
                            prev,
                            ("collective", "compute"),
                            f"worker {pid}: snapshot preceded by "
                            f"{prev} (expected collective or compute)",
                        )

        # -- Invariant 3: every restore is followed by either a
        #    collective or a compute (never another restore) --
        for pid, worker_events in per_worker.items():
            for i, e in enumerate(worker_events):
                if e["cat"] == "mux" and e["name"] == "restore":
                    if i < len(worker_events) - 1:
                        nxt = worker_events[i + 1]["cat"]
                        self.assertIn(
                            nxt,
                            ("collective", "compute"),
                            f"worker {pid}: restore followed by "
                            f"{nxt} (expected collective or compute)",
                        )

        # -- Invariant 4: the last worker to arrive at a collective
        #    resolves without snapshotting. At least one worker must
        #    have at least one collective NOT followed by a snapshot. --
        any_immediate = False
        for pid, worker_events in per_worker.items():
            for i, e in enumerate(worker_events):
                if e["cat"] == "collective":
                    next_is_snapshot = (
                        i + 1 < len(worker_events)
                        and worker_events[i + 1]["cat"] == "mux"
                        and worker_events[i + 1]["name"] == "snapshot"
                    )
                    if not next_is_snapshot:
                        any_immediate = True
        self.assertTrue(
            any_immediate,
            "every collective across all workers is followed by a "
            "snapshot (expected the last-to-arrive worker to resolve "
            "at least one collective immediately)",
        )

        # -- Invariant 5: snapshot-restore pairs bracket gaps where
        #    other workers run. No worker should have snapshot without
        #    a later restore (except the final cleanup). --
        for pid, worker_events in per_worker.items():
            snapshots = sum(
                1
                for e in worker_events
                if e["cat"] == "mux" and e["name"] == "snapshot"
            )
            restores = sum(
                1
                for e in worker_events
                if e["cat"] == "mux" and e["name"] == "restore"
            )
            self.assertIn(
                snapshots - restores,
                (0, 1),
                f"worker {pid}: {snapshots} snapshots vs {restores} "
                f"restores (expected equal or off by one)",
            )


class TestTorchmuxCollectiveCorrectness(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        super().tearDown()

    def test_collective_correctness_2workers(self):
        script_path = os.path.join(self._tmpdir, "correctness.py")
        with open(script_path, "w") as f:
            f.write(CORRECTNESS_SCRIPT)

        results_dir = os.path.join(self._tmpdir, "results")
        os.makedirs(results_dir)

        env = os.environ.copy()
        env["TORCHMUX_NGPUS"] = "1"
        env["TORCHMUX_RESULTS_DIR"] = results_dir
        env["TORCHMUX_TRACE_DIR"] = self._tmpdir

        result = subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.torchmux",
                "--nproc-per-node", "2", script_path,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode, 0,
            f"torchmux failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

        for rank in range(2):
            rpath = os.path.join(results_dir, f"rank{rank}.json")
            self.assertTrue(os.path.exists(rpath), f"missing results for rank {rank}")
            with open(rpath) as f:
                results = json.load(f)
            for op_name, passed in results.items():
                self.assertTrue(passed, f"rank {rank}: {op_name} failed")


class TestVNCCL(TestCase):
    """Tests vnccl collective correctness using direct PG method calls.

    Each test creates VNCCLProcessGroup instances directly and runs
    collective operations from separate threads, bypassing
    dist.init_process_group to avoid global state conflicts.
    """

    def setUp(self):
        super().setUp()
        self._world_size = 3
        self._pgs = []

    def tearDown(self):
        from torch.distributed.vnccl import VNCCLProcessGroup

        VNCCLProcessGroup._active.clear()
        for pg in self._pgs:
            dist.distributed_c10d._world.pg_names.pop(pg, None)
        self._pgs = []
        super().tearDown()

    def _make_pgs(self):
        from torch.distributed.vnccl import VNCCLProcessGroup

        pgs = []
        for r in range(self._world_size):
            pg = VNCCLProcessGroup(r, self._world_size)
            dist.distributed_c10d._world.pg_names[pg] = "vnccl_test"
            pgs.append(pg)
        self._pgs = pgs
        return pgs

    def _run_on_pgs(self, pgs, fn):
        results = [None] * self._world_size
        errors = [None] * self._world_size

        def _worker(rank):
            try:
                results[rank] = fn(rank, pgs[rank])
            except Exception as e:
                errors[rank] = e

        threads = [
            threading.Thread(target=_worker, args=(r,))
            for r in range(self._world_size)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        for r, e in enumerate(errors):
            if e is not None:
                raise RuntimeError(f"rank {r} failed") from e
        return results

    def test_allreduce_sum(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), float(rank + 1))]
            pg.allreduce(t)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        expected = self._world_size * (self._world_size + 1) / 2
        for r, t in enumerate(results):
            self.assertTrue(
                torch.allclose(t, torch.full((4,), expected)),
                f"rank {r}: allreduce got {t}, expected {expected}",
            )

    def test_broadcast(self):
        pgs = self._make_pgs()

        from torch._C._distributed_c10d import BroadcastOptions

        def _work(rank, pg):
            t = [torch.full((4,), 99.0) if rank == 0 else torch.zeros(4)]
            opts = BroadcastOptions()
            opts.rootRank = 0
            pg.broadcast(t, opts)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        for r, t in enumerate(results):
            self.assertTrue(
                torch.allclose(t, torch.full((4,), 99.0)),
                f"rank {r}: broadcast got {t}, expected 99.0",
            )

    def test_allgather(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            inp = [torch.full((2,), float(rank))]
            out = [[torch.zeros(2) for _ in range(self._world_size)]]
            pg.allgather(out, inp)
            return out[0]

        results = self._run_on_pgs(pgs, _work)
        for r, gathered in enumerate(results):
            for src, t in enumerate(gathered):
                self.assertTrue(
                    torch.allclose(t, torch.full((2,), float(src))),
                    f"rank {r}: allgather[{src}] got {t}, expected {float(src)}",
                )

    def test_barrier(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            pg.barrier()
            return True

        results = self._run_on_pgs(pgs, _work)
        for r, v in enumerate(results):
            self.assertTrue(v, f"rank {r}: barrier failed")


if __name__ == "__main__":
    run_tests()
