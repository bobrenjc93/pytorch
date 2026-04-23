"""Integration test for torchmux trace ordering.

Runs a minimal training script (2 allreduces) under torchmux with 3
workers and asserts the trace event sequence matches the expected
cooperative scheduling pattern.
"""

import json
import os
import sys
import tempfile

import torch
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


class TestTorchmuxTraceOrdering(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self._tmpdir = tempfile.mkdtemp()
        self._script = os.path.join(self._tmpdir, "train.py")
        with open(self._script, "w") as f:
            f.write(TRAIN_SCRIPT)

    def test_3worker_trace_ordering(self):
        """Assert trace events follow cooperative scheduling invariants.

        Expected pattern per collective (3 workers):
          W0: compute → collective_start → snapshot
          W1: compute → collective_start → snapshot
          W2: compute → collective_start → collective_finish (resolves)
          W2: continues to next compute...
          W2: next_collective_start → snapshot (yields for first time)
          W0: restore → collective_finish → compute
          ...
        """
        trace_dir = os.path.join(self._tmpdir, "traces")
        os.makedirs(trace_dir, exist_ok=True)

        ret = os.system(
            f"TORCHMUX_TRACE_DIR={trace_dir} "
            f"TORCHMUX_NGPUS=1 "
            f"{sys.executable} -m torch.distributed.torchmux "
            f"--nproc 3 {self._script} 2>/dev/null"
        )
        self.assertEqual(ret, 0, "torchmux exited with error")

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


if __name__ == "__main__":
    run_tests()
