"""
Scheduler implementations:
  FIFO, RoundRobin, StaticPriority, EDF, PQDeadline, QoS,
  PAES (original), PAES-S (staleness-aware), PAES-S-Drop (with task dropping)
"""
import heapq
import time
import math
from typing import List, Optional
from models.task import Task


# ── Base simulator ────────────────────────────────────────────────────────────

def simulate(tasks: List[Task], scheduler_fn, drop_enabled: bool = False):
    """
    Discrete-event simulation of a single-CPU non-preemptive queue.

    scheduler_fn(pending: List[Task], current_time: float) -> Task
        Returns the next task to execute from the pending list.
    """
    pending   = []
    completed = []
    dropped   = []
    clock     = 0.0   # seconds (logical, not wall-clock)

    # Sort by arrival
    queue = sorted(tasks, key=lambda t: t.arrival_time)
    idx   = 0

    while idx < len(queue) or pending:
        # Admit all tasks that have arrived by current clock
        while idx < len(queue) and queue[idx].arrival_time <= clock:
            pending.append(queue[idx])
            idx += 1

        if not pending:
            # Jump clock to next arrival
            clock = queue[idx].arrival_time
            continue

        # Optional: drop tasks whose staleness is below threshold
        if drop_enabled:
            survivors = []
            for t in pending:
                if t.should_drop(clock):
                    t.dropped = True
                    dropped.append(t)
                else:
                    survivors.append(t)
            pending = survivors
            if not pending:
                if idx < len(queue):
                    clock = queue[idx].arrival_time
                continue

        # Select next task
        chosen = scheduler_fn(pending, clock)
        pending.remove(chosen)

        chosen.start_time = clock
        clock += chosen.latency_ms / 1000.0   # advance clock by inference time
        chosen.end_time = clock
        completed.append(chosen)

        # Admit any tasks that arrived during inference
        while idx < len(queue) and queue[idx].arrival_time <= clock:
            pending.append(queue[idx])
            idx += 1

    return completed, dropped


# ── Scheduler functions ───────────────────────────────────────────────────────

def fifo(pending: List[Task], clock: float) -> Task:
    return min(pending, key=lambda t: t.arrival_time)


def round_robin(pending: List[Task], clock: float,
                _state={"last_idx": -1}) -> Task:
    models = sorted(set(t.model_name for t in pending))
    _state["last_idx"] = (_state["last_idx"] + 1) % len(models)
    target = models[_state["last_idx"] % len(models)]
    candidates = [t for t in pending if t.model_name == target]
    if not candidates:
        candidates = pending
    return min(candidates, key=lambda t: t.arrival_time)


def static_priority(pending: List[Task], clock: float) -> Task:
    return max(pending, key=lambda t: (t.priority, -t.arrival_time))


def edf(pending: List[Task], clock: float) -> Task:
    return min(pending, key=lambda t: t.arrival_time + t.deadline_ms / 1000.0)


def pq_deadline(pending: List[Task], clock: float) -> Task:
    def score(t):
        urgency = 1.0 / max(0.001, (t.arrival_time + t.deadline_ms / 1000.0) - clock)
        return t.priority + 0.1 * urgency
    return max(pending, key=score)


def qos(pending: List[Task], clock: float) -> Task:
    def tier(t):
        if t.priority >= 3.0:   return 3
        if t.priority >= 2.0:   return 2
        return 1
    best_tier = max(tier(t) for t in pending)
    candidates = [t for t in pending if tier(t) == best_tier]
    return min(candidates, key=lambda t: t.arrival_time + t.deadline_ms / 1000.0)


def paes(alpha=1.0, beta=1.0, gamma=1.0):
    """Original PAES: αP + β/L + γ/E"""
    def _score(t: Task) -> float:
        return (alpha * t.priority
                + beta  / max(0.001, t.latency_ms)
                + gamma / max(0.001, t.energy_mj))
    def _scheduler(pending: List[Task], clock: float) -> Task:
        return max(pending, key=_score)
    return _scheduler


def paes_s(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
    """
    PAES-S: αP + β/L + γ/E + δ·S(w)
    Adds staleness term: tasks decaying fast get a dynamic urgency boost.
    """
    def _score(t: Task, clock: float) -> float:
        wait = clock - t.arrival_time
        s    = t.decay_fn(wait)
        return (alpha * t.priority
                + beta  / max(0.001, t.latency_ms)
                + gamma / max(0.001, t.energy_mj)
                + delta * s)
    def _scheduler(pending: List[Task], clock: float) -> Task:
        return max(pending, key=lambda t: _score(t, clock))
    return _scheduler


def paes_s_drop(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
    """PAES-S with task dropping enabled (passed as drop_enabled=True to simulate())."""
    return paes_s(alpha, beta, gamma, delta)


# ── Registry ──────────────────────────────────────────────────────────────────

SCHEDULERS = {
    "FIFO":           (fifo,                      False),
    "RoundRobin":     (round_robin,               False),
    "StaticPriority": (static_priority,           False),
    "EDF":            (edf,                       False),
    "PQ+Deadline":    (pq_deadline,               False),
    "QoS":            (qos,                       False),
    "PAES":           (paes(),                    False),
    "PAES-S":         (paes_s(delta=1.0),         False),
    "PAES-S-Drop":    (paes_s_drop(delta=1.0),    True),
}
