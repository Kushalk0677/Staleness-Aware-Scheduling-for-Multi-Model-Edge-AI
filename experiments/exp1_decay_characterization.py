"""
Experiment 1 — Staleness Decay Characterisation
================================================
RQ: What is the staleness decay profile of each AI model type?

Method:
  - For each model, simulate tasks arriving at time 0
    but scheduled after artificially imposed wait times [0 … 5 s].
  - Record S(w) = decay_fn(w) and map it to an information-quality proxy.
  - Plot decay curves; fit λ for exponential models.
  - Output: Figure 1 (decay curves) + Table I (fitted parameters).

This is entirely simulation-based — no real model inference required.
To extend with real inference: replace decay_fn values with empirically
measured accuracy vs. frame-age on a dataset (e.g. KITTI for MiDaS,
MOT for YOLOv5n).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.family": "serif", "font.size": 11})

from models.task import MODEL_PROFILES


def run(save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    wait_s = np.linspace(0, 5, 500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: decay curves ────────────────────────────────────────────────
    ax = axes[0]
    styles = ["-", "--", "-.", ":", "-"]
    markers = [None, None, None, None, "x"]
    for i, (model, prof) in enumerate(MODEL_PROFILES.items()):
        decay = np.array([prof["decay_fn"](w) for w in wait_s])
        ax.plot(wait_s, decay, linestyle=styles[i % len(styles)],
                marker=markers[i % len(markers)], markevery=50,
                label=f"{model} ({prof['decay_label']})", linewidth=1.8)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="50% quality")
    ax.set_xlabel("Queue Wait Time (s)")
    ax.set_ylabel("Information Quality S(w)")
    ax.set_title("Figure 1a — Staleness Decay by Model")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # ── Right: half-life (time to reach 50% quality) ─────────────────────
    ax2 = axes[1]
    half_lives = {}
    for model, prof in MODEL_PROFILES.items():
        fn = prof["decay_fn"]
        # Binary search for w where S(w) = 0.5
        lo, hi = 0.0, 100.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if fn(mid) > 0.5:
                lo = mid
            else:
                hi = mid
        hl = (lo + hi) / 2
        half_lives[model] = hl if hl < 99 else float("inf")

    models  = list(half_lives.keys())
    hl_vals = [min(v, 10) for v in half_lives.values()]   # cap at 10 for plot
    colors  = ["#d62728" if v < 1 else "#ff7f0e" if v < 3 else "#2ca02c"
               for v in hl_vals]

    bars = ax2.bar(models, hl_vals, color=colors, edgecolor="black", linewidth=0.7)
    ax2.set_ylabel("Half-life (s)  [time to 50% quality]")
    ax2.set_title("Figure 1b — Information Half-Life per Model")
    ax2.set_xticklabels(models, rotation=20, ha="right")
    ax2.axhline(1.0, color="red",    linestyle="--", linewidth=1, label="1 s threshold")
    ax2.axhline(3.0, color="orange", linestyle="--", linewidth=1, label="3 s threshold")
    for bar, model in zip(bars, models):
        v = half_lives[model]
        label = f"{v:.2f}s" if v < 99 else "∞"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 label, ha="center", va="bottom", fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(save_dir, "exp1_staleness_decay.pdf")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Exp1] Saved → {path}")

    # ── Table I ───────────────────────────────────────────────────────────
    print("\nTable I — Model Staleness Profiles")
    print(f"{'Model':<14} {'Decay Function':<25} {'Half-life':>10} {'Drop Threshold':>15}")
    print("-" * 68)
    for model, prof in MODEL_PROFILES.items():
        hl = half_lives[model]
        hl_str = f"{hl:.2f} s" if hl < 99 else "∞"
        print(f"{model:<14} {prof['decay_label']:<25} {hl_str:>10} "
              f"{prof['staleness_drop']:>14.0%}")

    # ── Quality gap analysis ──────────────────────────────────────────────
    print("\n--- Quality at representative wait times ---")
    print(f"{'Model':<14}", end="")
    waits = [0.0, 0.5, 1.0, 2.0, 5.0]
    for w in waits:
        print(f"  {w:.1f}s", end="")
    print()
    for model, prof in MODEL_PROFILES.items():
        print(f"{model:<14}", end="")
        for w in waits:
            q = prof["decay_fn"](w)
            print(f"  {q:.3f}", end="")
        print()

    return half_lives


if __name__ == "__main__":
    run()
