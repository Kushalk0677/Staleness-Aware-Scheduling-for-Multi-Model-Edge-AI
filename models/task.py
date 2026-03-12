"""
Task definitions and AI model profiles for PAES-S experiments.
Each model has: priority, expected latency, energy, deadline, staleness decay params.
"""
import time
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Callable


# ── Staleness decay functions ────────────────────────────────────────────────

def exponential_decay(lam: float) -> Callable[[float], float]:
    """S(w) = exp(-lam * w).  Fast decay for time-critical sensors."""
    return lambda w: math.exp(-lam * w)

def linear_decay(rate: float) -> Callable[[float], float]:
    """S(w) = max(0, 1 - rate*w).  Moderate decay."""
    return lambda w: max(0.0, 1.0 - rate * w)

def flat_decay() -> Callable[[float], float]:
    """S(w) = 1.0.  No staleness (NLP, speech)."""
    return lambda w: 1.0

def step_decay(threshold: float) -> Callable[[float], float]:
    """S(w) = 1 if w < threshold else 0.  Hard deadline staleness."""
    return lambda w: 1.0 if w < threshold else 0.0


# ── Model profiles ───────────────────────────────────────────────────────────

MODEL_PROFILES = {
    "YOLOv5n": {
        "priority":        3.0,
        "latency_ms":      80.0,
        "energy_mj":       120.0,
        "deadline_ms":     300.0,
        "decay_fn":        exponential_decay(lam=0.8),   # moderate-fast decay
        "decay_label":     "exponential(λ=0.8)",
        "staleness_drop":  0.25,   # drop threshold: below 25% quality not worth running
        "description":     "Object detection — objects move, frames age quickly",
    },
    "MobileNetV2": {
        "priority":        2.0,
        "latency_ms":      35.0,
        "energy_mj":       55.0,
        "deadline_ms":     200.0,
        "decay_fn":        linear_decay(rate=0.3),
        "decay_label":     "linear(rate=0.3)",
        "staleness_drop":  0.15,
        "description":     "Classification — slower decay, objects are often static",
    },
    "WhisperTiny": {
        "priority":        2.0,
        "latency_ms":      150.0,
        "energy_mj":       200.0,
        "deadline_ms":     500.0,
        "decay_fn":        flat_decay(),
        "decay_label":     "flat",
        "staleness_drop":  0.0,    # audio is recorded — never drop
        "description":     "Speech recognition — recorded audio doesn't expire",
    },
    "DistilBERT": {
        "priority":        1.5,
        "latency_ms":      55.0,
        "energy_mj":       80.0,
        "deadline_ms":     400.0,
        "decay_fn":        flat_decay(),
        "decay_label":     "flat",
        "staleness_drop":  0.0,
        "description":     "NLP — text queries don't expire",
    },
    "MiDaS": {
        "priority":        1.0,
        "latency_ms":      110.0,
        "energy_mj":       160.0,
        "deadline_ms":     600.0,
        "decay_fn":        exponential_decay(lam=2.5),   # fastest decay — depth is most time-critical
        "decay_label":     "exponential(λ=2.5)",
        "staleness_drop":  0.30,
        "description":     "Depth estimation — stale depth = collision risk",
    },
}


# ── Task dataclass ───────────────────────────────────────────────────────────

@dataclass
class Task:
    task_id:       int
    model_name:    str
    priority:      float
    latency_ms:    float          # expected inference time
    energy_mj:     float
    deadline_ms:   float
    arrival_time:  float          # wall-clock seconds since epoch
    decay_fn:      Callable       = field(repr=False)
    staleness_drop: float         = 0.0

    # set at completion
    start_time:    Optional[float] = None
    end_time:      Optional[float] = None
    dropped:       bool            = False

    @property
    def queue_wait_s(self) -> float:
        if self.start_time is None:
            return 0.0
        return self.start_time - self.arrival_time

    @property
    def staleness_at_inference(self) -> float:
        """S(w) where w = queue_wait in seconds."""
        return self.decay_fn(self.queue_wait_s)

    @property
    def deadline_met(self) -> bool:
        if self.end_time is None:
            return False
        total_ms = (self.end_time - self.arrival_time) * 1000
        return total_ms <= self.deadline_ms

    def should_drop(self, current_time: float) -> bool:
        """True if staleness has fallen below the drop threshold."""
        wait = current_time - self.arrival_time
        return self.decay_fn(wait) < self.staleness_drop and self.staleness_drop > 0


def make_task(task_id: int, model_name: str, arrival_time: float,
              noise_pct: float = 0.1) -> Task:
    """Instantiate a Task from MODEL_PROFILES with optional latency noise."""
    p = MODEL_PROFILES[model_name]
    noise = 1.0 + random.gauss(0, noise_pct)
    return Task(
        task_id=task_id,
        model_name=model_name,
        priority=p["priority"],
        latency_ms=max(5.0, p["latency_ms"] * noise),
        energy_mj=p["energy_mj"],
        deadline_ms=p["deadline_ms"],
        arrival_time=arrival_time,
        decay_fn=p["decay_fn"],
        staleness_drop=p["staleness_drop"],
    )
