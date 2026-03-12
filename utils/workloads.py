"""
Workload generators for PAES-S experiments.
"""
import random
import numpy as np
from typing import List
from models.task import Task, make_task, MODEL_PROFILES

MODELS = list(MODEL_PROFILES.keys())


def synthetic_uniform(n_tasks: int = 600, seed: int = 42) -> List[Task]:
    """600 tasks sampled uniformly across models (mirrors PAES paper)."""
    random.seed(seed)
    np.random.seed(seed)
    tasks = []
    t = 0.0
    for i in range(n_tasks):
        model = random.choice(MODELS)
        tasks.append(make_task(i, model, arrival_time=t))
        t += random.expovariate(1 / 0.05)   # ~50ms inter-arrival
    return tasks


def robot_pipeline(duration_s: float = 30.0, seed: int = 42) -> List[Task]:
    """
    Realistic robot pipeline — five concurrent Poisson streams.
    Mirrors PAES robot workload (685 tasks / 30 s).
    """
    random.seed(seed)
    np.random.seed(seed)
    streams = {
        "YOLOv5n":    1 / 0.1,    # camera: ~10 Hz
        "MiDaS":      1 / 0.15,   # depth:  ~7 Hz
        "MobileNetV2":1 / 0.3,    # classifier: ~3 Hz
        "WhisperTiny":1 / 3.0,    # mic: Poisson λ=1/3
        "DistilBERT": 1 / 2.0,    # NLP planner
    }
    tasks = []
    tid = 0
    for model, rate in streams.items():
        t = random.expovariate(rate)
        while t < duration_s:
            tasks.append(make_task(tid, model, arrival_time=t))
            tid += 1
            t += random.expovariate(rate)
    tasks.sort(key=lambda x: x.arrival_time)
    return tasks


def bursty(n_bursts: int = 20, burst_size: int = 15, seed: int = 42) -> List[Task]:
    """
    Bursty arrival: clusters of tasks arrive together, then silence.
    Tests scheduler behaviour under sudden overload.
    """
    random.seed(seed)
    tasks = []
    tid = 0
    t = 0.0
    for _ in range(n_bursts):
        for _ in range(burst_size):
            model = random.choice(MODELS)
            jitter = random.uniform(0, 0.02)
            tasks.append(make_task(tid, model, arrival_time=t + jitter))
            tid += 1
        t += random.uniform(1.0, 3.0)    # quiet period
    tasks.sort(key=lambda x: x.arrival_time)
    return tasks


def staleness_stress(n_tasks: int = 200, seed: int = 42) -> List[Task]:
    """
    Stress test for staleness: floods with low-priority NLP tasks
    while high-staleness depth/vision tasks arrive slowly.
    Designed to maximally expose the staleness vs. queue-wait tradeoff.
    """
    random.seed(seed)
    tasks = []
    tid = 0
    t = 0.0
    for i in range(n_tasks):
        # 70% NLP/Whisper (flat decay), 30% MiDaS/YOLO (fast decay)
        if random.random() < 0.70:
            model = random.choice(["DistilBERT", "WhisperTiny"])
        else:
            model = random.choice(["MiDaS", "YOLOv5n"])
        tasks.append(make_task(tid, model, arrival_time=t))
        tid += 1
        t += random.expovariate(1 / 0.04)
    tasks.sort(key=lambda x: x.arrival_time)
    return tasks
