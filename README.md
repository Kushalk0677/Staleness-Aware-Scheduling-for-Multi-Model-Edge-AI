# PAES-S: Priority-Aware Edge Scheduler with Staleness Awareness

Experimental framework for the paper:
**"Staleness-Aware Scheduling for Multi-Model Edge AI: When Minimizing Queue
Wait Optimizes the Wrong Objective"**

Extends [PAES (Khemani et al.)](https://github.com/Kushalk0677/Priority-Aware-Adaptive-Scheduling-for-Multi-Model-Edge-AI-Systems)
with a formal staleness model and the novel **Staleness-Weighted Quality (SWQ)** metric.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run All Experiments

```bash
# Full run (N_RUNS=10, matches paper methodology, ~5–15 min)
python run_all.py

# Quick smoke-test (N_RUNS=2, ~1 min)
python run_all.py --quick
```

Results (PDFs + PNGs) are saved to `./results/`.

---

## Run Individual Experiments

```bash
cd paes_s/

python experiments/exp1_decay_characterization.py   # Figure 1: decay curves + half-lives
python experiments/exp2_ranking_divergence.py        # Figure 2: QW vs SWQ scatter
python experiments/exp3_delta_sweep.py               # Figure 3: Pareto frontier
python experiments/exp4_drop_threshold.py            # Figure 4: drop threshold sweep
```

---

## Project Structure

```
paes_s/
├── models/
│   └── task.py             # Task dataclass, model profiles, decay functions
├── schedulers/
│   └── schedulers.py       # FIFO, RR, SP, EDF, PQ+DL, QoS, PAES, PAES-S, PAES-S-Drop
├── utils/
│   ├── metrics.py          # SWQ, queue wait, miss rate, drop rate
│   └── workloads.py        # Synthetic, robot pipeline, bursty, staleness stress
├── experiments/
│   ├── exp1_decay_characterization.py
│   ├── exp2_ranking_divergence.py
│   ├── exp3_delta_sweep.py
│   └── exp4_drop_threshold.py
├── results/                # Auto-generated output directory
├── run_all.py
└── requirements.txt
```

---

## Key Novel Contributions

### 1. Staleness Decay Functions
Each AI model type gets a per-model information decay function S(w):
- **MiDaS (depth)**: exponential(λ=2.5) — fastest decay, collision-critical
- **YOLOv5 (detection)**: exponential(λ=0.8) — objects move
- **MobileNetV2**: linear(0.3) — objects often static
- **Whisper / DistilBERT**: flat — recorded audio/text doesn't expire

### 2. Staleness-Weighted Quality (SWQ) Metric
```
SWQ = mean( priority_i × S(queue_wait_i) )  for all completed tasks
```
A scheduler maximizing SWQ delivers *fresher results to more important tasks* —
a strictly richer objective than minimizing mean queue wait.

### 3. PAES-S Scoring Function
```
Score(t_i) = α·P_i + β/L_i + γ/E_i + δ·S_i(w_i)
```
The δ·S(w) term dynamically urgency-boosts tasks whose information is decaying fast.

### 4. Task Drop Policy
When S(w) falls below a per-model threshold, running the inference is worse
than skipping it — the result would mislead downstream decision-making.

---

## Extending to Real Inference

The current framework uses simulated inference times (with Gaussian noise).
To use real PyTorch inference:

1. Replace `latency_ms` in `make_task()` with actual measured inference time
2. For Exp 1 decay curves: feed aged frames (artificially delayed by `queue_wait`)
   through the real model and measure output quality vs. a fresh-frame baseline
3. Suggested datasets: KITTI (depth), MOT17 (detection), LibriSpeech (speech)

---

## Citation

If you use this framework, please cite the paper:

