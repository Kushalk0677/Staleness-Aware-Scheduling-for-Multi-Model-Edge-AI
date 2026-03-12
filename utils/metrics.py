"""
Metrics computation for PAES-S experiments.
"""
import math
import numpy as np
from typing import List, Tuple
from models.task import Task


def queue_wait_ms(tasks: List[Task]) -> List[float]:
    return [t.queue_wait_s * 1000 for t in tasks if not t.dropped]


def staleness_at_inference(tasks: List[Task]) -> List[float]:
    return [t.staleness_at_inference for t in tasks if not t.dropped]


def staleness_weighted_quality(tasks: List[Task]) -> float:
    """
    Primary novel metric: mean S(w) at inference time, weighted by task priority.
    Higher = better (fresher results for more important tasks).
    """
    vals = []
    for t in tasks:
        if not t.dropped:
            vals.append(t.priority * t.staleness_at_inference)
    return float(np.mean(vals)) if vals else 0.0


def deadline_miss_rate(tasks: List[Task]) -> float:
    eligible = [t for t in tasks if not t.dropped]
    if not eligible:
        return 0.0
    misses = sum(1 for t in eligible if not t.deadline_met)
    return misses / len(eligible)


def drop_rate(completed: List[Task], dropped: List[Task]) -> float:
    total = len(completed) + len(dropped)
    return len(dropped) / total if total else 0.0


def compute_all_metrics(completed: List[Task], dropped: List[Task]) -> dict:
    qw   = queue_wait_ms(completed)
    st   = staleness_at_inference(completed)
    return {
        "n_completed":          len(completed),
        "n_dropped":            len(dropped),
        "drop_rate":            drop_rate(completed, dropped),
        "queue_wait_mean_ms":   float(np.mean(qw))   if qw else 0.0,
        "queue_wait_std_ms":    float(np.std(qw))    if qw else 0.0,
        "staleness_mean":       float(np.mean(st))   if st else 0.0,
        "staleness_std":        float(np.std(st))    if st else 0.0,
        "swq":                  staleness_weighted_quality(completed),
        "deadline_miss_rate":   deadline_miss_rate(completed),
        "total_energy_mj":      sum(t.energy_mj for t in completed),
    }


def per_model_staleness(tasks: List[Task]) -> dict:
    """Per-model breakdown of mean staleness at inference."""
    from collections import defaultdict
    groups = defaultdict(list)
    for t in tasks:
        if not t.dropped:
            groups[t.model_name].append(t.staleness_at_inference)
    return {m: float(np.mean(v)) for m, v in groups.items()}


def summary_table(results: dict) -> str:
    """Pretty-print a results dict."""
    lines = [
        f"{'Scheduler':<18} {'QueueWait(ms)':>14} {'Staleness':>10} "
        f"{'SWQ':>8} {'MissRate':>10} {'Dropped':>8}"
    ]
    lines.append("-" * 72)
    for sched, runs in results.items():
        qw  = np.mean([r["queue_wait_mean_ms"] for r in runs])
        st  = np.mean([r["staleness_mean"]     for r in runs])
        swq = np.mean([r["swq"]                for r in runs])
        mr  = np.mean([r["deadline_miss_rate"] for r in runs])
        dr  = np.mean([r["drop_rate"]          for r in runs])
        lines.append(
            f"{sched:<18} {qw:>14.1f} {st:>10.3f} "
            f"{swq:>8.3f} {mr:>10.3f} {dr:>8.3f}"
        )
    return "\n".join(lines)
