"""
Tracing for torchmux: produces two chrome://tracing JSON files.

Natural trace: shows the actual serial execution with time slices per
worker, including snapshot/restore overhead spans.

Synthetic trace: reconstructs what a parallel run would look like.
Workers' compute spans are overlapped (aligned at start), and
collective boundaries are placed at max(worker_durations) + a small
collective overhead.
"""

import json
import os
import time
import threading

_tls = threading.local()

# Collected per-worker events: list of (rank, category, name, start_us, dur_us)
_events = []
_events_lock = threading.Lock()


def _us():
    return time.monotonic() * 1e6


def record(rank, cat, name, start_us, dur_us):
    with _events_lock:
        _events.append((rank, cat, name, start_us, dur_us))


def begin_span(rank, cat, name):
    return (rank, cat, name, _us())


def end_span(span):
    rank, cat, name, start = span
    record(rank, cat, name, start, _us() - start)


class SpanContext:
    def __init__(self, rank, cat, name):
        self.rank = rank
        self.cat = cat
        self.name = name

    def __enter__(self):
        self._start = _us()
        return self

    def __exit__(self, *args):
        record(self.rank, self.cat, self.name, self._start, _us() - self._start)


def export_natural(path, nproc):
    """
    Export the natural (serial) trace. Each worker is a separate
    process in the chrome trace, showing the actual execution order.
    """
    trace_events = []

    for rank in range(nproc):
        trace_events.append({
            "ph": "M", "name": "process_name",
            "pid": rank, "tid": 0,
            "args": {"name": f"Worker {rank}"},
        })
        trace_events.append({
            "ph": "M", "name": "thread_name",
            "pid": rank, "tid": 0,
            "args": {"name": "main"},
        })

    with _events_lock:
        events = list(_events)

    base_ts = min(e[3] for e in events) if events else 0
    for rank, cat, name, start_us, dur_us in events:
        trace_events.append({
            "ph": "X",
            "cat": cat,
            "name": name,
            "pid": rank,
            "tid": 0,
            "ts": start_us - base_ts,
            "dur": dur_us,
        })

    trace = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ms",
    }
    with open(path, "w") as f:
        json.dump(trace, f)


def export_synthetic(path, nproc):
    """
    Export the synthetic (parallel) trace. Reconstructs what a parallel
    run would look like by overlapping workers' compute spans between
    collectives.

    Algorithm:
      1. Group events by worker, sorted by time.
      2. Identify collective boundaries (events with cat="collective").
      3. Between each pair of consecutive collectives, each worker has a
         "compute phase." In the synthetic trace, all workers' compute
         phases start at the same time (parallel).
      4. The collective is placed at max(worker_compute_durations), with
         duration = min(measured_durations) across workers — the last
         worker to arrive does the actual work without waiting.
      5. Overhead spans (rng_save/restore) are omitted.
    """
    with _events_lock:
        events = list(_events)

    if not events:
        _write_empty_trace(path, nproc)
        return

    per_rank = {r: [] for r in range(nproc)}
    for rank, cat, name, start_us, dur_us in events:
        per_rank[rank].append((cat, name, start_us, dur_us))

    for r in per_rank:
        per_rank[r].sort(key=lambda e: e[2])

    phases = _extract_phases(per_rank, nproc)

    trace_events = []
    for rank in range(nproc):
        trace_events.append({
            "ph": "M", "name": "process_name",
            "pid": rank, "tid": 0,
            "args": {"name": f"Worker {rank}"},
        })

    synthetic_cursor = 0.0
    for phase in phases:
        max_compute = 0.0
        min_coll_dur = float("inf")
        coll_name = None

        for rank in range(nproc):
            worker_events = phase.get(rank, [])
            compute_dur = sum(
                d for c, _, _, d in worker_events if c == "compute" and d > 0
            )
            max_compute = max(max_compute, compute_dur)
            for c, n, _, d in worker_events:
                if c == "collective" and d > 0:
                    min_coll_dur = min(min_coll_dur, d)
                    coll_name = n

        for rank in range(nproc):
            cursor = synthetic_cursor
            for cat, name, _, dur_us in phase.get(rank, []):
                if cat != "compute" or dur_us <= 0:
                    continue
                trace_events.append({
                    "ph": "X", "cat": "compute", "name": name,
                    "pid": rank, "tid": 0,
                    "ts": cursor, "dur": dur_us,
                })
                cursor += dur_us

        if coll_name and min_coll_dur < float("inf"):
            for rank in range(nproc):
                trace_events.append({
                    "ph": "X", "cat": "collective",
                    "name": coll_name,
                    "pid": rank, "tid": 0,
                    "ts": synthetic_cursor + max_compute,
                    "dur": min_coll_dur,
                })
            synthetic_cursor += max_compute + min_coll_dur
        else:
            synthetic_cursor += max_compute

    trace = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ms",
    }
    with open(path, "w") as f:
        json.dump(trace, f)


def _extract_phases(per_rank, nproc):
    """
    Split each worker's events into phases separated by collectives.
    A phase is a dict mapping rank -> list of events between two
    consecutive collective boundaries.
    """
    coll_indices = {r: [] for r in range(nproc)}
    for r in range(nproc):
        for i, (cat, name, start, dur) in enumerate(per_rank[r]):
            if cat == "collective":
                coll_indices[r].append(i)

    if not coll_indices.get(0):
        return [{r: list(per_rank[r]) for r in range(nproc)}]

    num_colls = len(coll_indices[0])
    phases = []

    for ci in range(num_colls):
        phase = {}
        for r in range(nproc):
            if ci >= len(coll_indices[r]):
                continue
            end_idx = coll_indices[r][ci]
            start_idx = coll_indices[r][ci - 1] + 1 if ci > 0 else 0
            phase[r] = per_rank[r][start_idx : end_idx + 1]
        phases.append(phase)

    last_phase = {}
    for r in range(nproc):
        if coll_indices[r]:
            last_idx = coll_indices[r][-1] + 1
            if last_idx < len(per_rank[r]):
                last_phase[r] = per_rank[r][last_idx:]
    if last_phase:
        phases.append(last_phase)

    return phases


def _write_empty_trace(path, nproc):
    trace = {"traceEvents": [], "displayTimeUnit": "ms"}
    with open(path, "w") as f:
        json.dump(trace, f)


def clear():
    with _events_lock:
        _events.clear()
