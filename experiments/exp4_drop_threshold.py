"""
Experiment 4 — Task Drop Threshold Analysis
============================================
RQ: At what staleness level is it better to drop a task than execute it?
    What is the system-level impact of principled task dropping?

Method:
  - Define per-model drop thresholds (already in MODEL_PROFILES).
  - Run PAES-S (no-drop) vs. PAES-S-Drop on staleness_stress workload.
  - Measure: SWQ, drop rate, effective throughput, deadline miss rate.
  - Sweep drop threshold multiplier {0.5x, 1x, 2x, 4x} to show sensitivity.
  - Figure 4: SWQ and throughput vs. drop aggressiveness.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.family": "serif", "font.size": 11})

from models.task import MODEL_PROFILES
from schedulers.schedulers import paes_s, simulate
from utils.workloads import staleness_stress, robot_pipeline
from utils.metrics import compute_all_metrics, per_model_staleness

N_RUNS = 10
DROP_MULTIPLIERS = [0.0, 0.5, 1.0, 2.0, 4.0]   # 0 = no dropping


def make_tasks_with_drop_threshold(workload_fn, seed, multiplier):
    """Generate tasks with drop thresholds scaled by multiplier."""
    tasks = workload_fn(seed=seed)
    for t in tasks:
        base = MODEL_PROFILES[t.model_name]["staleness_drop"]
        t.staleness_drop = base * multiplier
    return tasks


def run(save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    sched_fn = paes_s(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)

    workloads = {
        "Staleness Stress": lambda seed: staleness_stress(n_tasks=200, seed=seed),
        "Robot Pipeline":   lambda seed: robot_pipeline(duration_s=30, seed=seed),
    }

    for wl_name, wl_fn in workloads.items():
        print(f"\n{'='*55}\nWorkload: {wl_name}\n{'='*55}")
        sweep_results = []

        for mult in DROP_MULTIPLIERS:
            swq_l, dr_l, mr_l, tput_l = [], [], [], []
            for run_i in range(N_RUNS):
                tasks = make_tasks_with_drop_threshold(wl_fn, run_i * 13 + 7, mult)
                drop_enabled = mult > 0
                completed, dropped = simulate(
                    copy.deepcopy(tasks), sched_fn, drop_enabled=drop_enabled
                )
                m = compute_all_metrics(completed, dropped)
                swq_l.append(m["swq"])
                dr_l.append(m["drop_rate"])
                mr_l.append(m["deadline_miss_rate"])
                # effective throughput = completed / (completed + dropped)
                total = m["n_completed"] + m["n_dropped"]
                tput_l.append(m["n_completed"] / total if total > 0 else 1.0)

            sweep_results.append({
                "mult":      mult,
                "swq_mean":  np.mean(swq_l),  "swq_std":  np.std(swq_l),
                "dr_mean":   np.mean(dr_l),   "dr_std":   np.std(dr_l),
                "mr_mean":   np.mean(mr_l),   "mr_std":   np.std(mr_l),
                "tput_mean": np.mean(tput_l), "tput_std": np.std(tput_l),
            })

        # ── Print table ───────────────────────────────────────────────────
        print(f"\n{'Drop mult':>10}  {'SWQ':>8}  {'Drop%':>7}  "
              f"{'MissRate':>10}  {'Throughput':>11}")
        print("-" * 55)
        for r in sweep_results:
            drop_str = "disabled" if r["mult"] == 0 else f"{r['mult']:.1f}x"
            print(f"{drop_str:>10}  {r['swq_mean']:>8.4f}  "
                  f"{r['dr_mean']:>6.1%}  {r['mr_mean']:>10.3f}  {r['tput_mean']:>10.1%}")

        # ── Figure 4 ──────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        mults   = [r["mult"] for r in sweep_results]
        swq_v   = [r["swq_mean"]  for r in sweep_results]
        swq_e   = [r["swq_std"]   for r in sweep_results]
        dr_v    = [r["dr_mean"]   for r in sweep_results]
        tput_v  = [r["tput_mean"] for r in sweep_results]
        mr_v    = [r["mr_mean"]   for r in sweep_results]
        xlabels = ["none", "0.5×", "1×\n(default)", "2×", "4×"]

        ax = axes[0]
        ax.errorbar(range(len(mults)), swq_v, yerr=swq_e,
                    fmt="o-", color="#1f77b4", capsize=4,
                    linewidth=1.8, markersize=8, label="SWQ")
        ax.set_xticks(range(len(mults)))
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Drop Threshold (relative to default)")
        ax.set_ylabel("Staleness-Weighted Quality", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")

        ax_r = ax.twinx()
        ax_r.bar(range(len(mults)), dr_v, alpha=0.25,
                 color="#d62728", label="Drop Rate")
        ax_r.set_ylabel("Drop Rate", color="#d62728")
        ax_r.tick_params(axis="y", labelcolor="#d62728")
        ax.set_title(f"Figure 4a — SWQ vs. Drop Aggressiveness\n({wl_name})")
        ax.grid(True, alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

        ax2 = axes[1]
        ax2.plot(range(len(mults)), tput_v, "s--", color="#2ca02c",
                 linewidth=1.8, markersize=8, label="Effective Throughput")
        ax2.plot(range(len(mults)), mr_v,   "^:", color="#ff7f0e",
                 linewidth=1.8, markersize=8, label="Deadline Miss Rate")
        ax2.set_xticks(range(len(mults)))
        ax2.set_xticklabels(xlabels)
        ax2.set_xlabel("Drop Threshold (relative to default)")
        ax2.set_ylabel("Rate")
        ax2.set_title(f"Figure 4b — Throughput & Miss Rate\n({wl_name})")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        tag  = wl_name.lower().replace(" ", "_")
        path = os.path.join(save_dir, f"exp4_drop_threshold_{tag}.pdf")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Exp4] Saved → {path}")

        # ── Per-model staleness breakdown (default drop config) ───────────
        print("\nPer-model staleness at inference (default threshold, PAES-S-Drop):")
        tasks_sample = make_tasks_with_drop_threshold(wl_fn, 0, 1.0)
        completed, dropped = simulate(
            copy.deepcopy(tasks_sample), sched_fn, drop_enabled=True
        )
        breakdown = per_model_staleness(completed)
        for model, st in sorted(breakdown.items(), key=lambda x: x[1]):
            print(f"  {model:<14}  {st:.4f}")
        print(f"  Tasks dropped: {len(dropped)} / {len(tasks_sample)} "
              f"({len(dropped)/len(tasks_sample):.1%})")


if __name__ == "__main__":
    run()
