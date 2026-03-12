"""
Experiment 3 — PAES-S Delta Sweep
===================================
RQ: What is the optimal δ for the staleness term, and what is the
    QW vs. SWQ tradeoff across δ values?

Method:
  - Sweep δ ∈ {0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0}
  - For each δ: run PAES-S on robot + staleness_stress workloads (N_RUNS each).
  - Record: queue_wait_mean, SWQ, deadline_miss_rate.
  - Plot Pareto frontier (QW vs SWQ), identify knee point.
  - Compare best PAES-S config vs. original PAES and FIFO.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.family": "serif", "font.size": 11})

from schedulers.schedulers import paes_s, paes, fifo, simulate
from utils.workloads import robot_pipeline, staleness_stress
from utils.metrics import compute_all_metrics

N_RUNS  = 10
DELTAS  = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def eval_delta(delta, workload_fn, n_runs=N_RUNS):
    sched_fn = paes_s(alpha=1.0, beta=1.0, gamma=1.0, delta=delta)
    qw_list, swq_list, mr_list = [], [], []
    for i in range(n_runs):
        tasks = workload_fn(seed=i * 11 + 5)
        completed, dropped = simulate(copy.deepcopy(tasks), sched_fn, drop_enabled=False)
        m = compute_all_metrics(completed, dropped)
        qw_list.append(m["queue_wait_mean_ms"])
        swq_list.append(m["swq"])
        mr_list.append(m["deadline_miss_rate"])
    return {
        "delta":    delta,
        "qw_mean":  np.mean(qw_list),
        "qw_std":   np.std(qw_list),
        "swq_mean": np.mean(swq_list),
        "swq_std":  np.std(swq_list),
        "mr_mean":  np.mean(mr_list),
    }


def eval_baseline(sched_fn, name, workload_fn, n_runs=N_RUNS):
    qw_list, swq_list, mr_list = [], [], []
    for i in range(n_runs):
        tasks = workload_fn(seed=i * 11 + 5)
        completed, dropped = simulate(copy.deepcopy(tasks), sched_fn, drop_enabled=False)
        m = compute_all_metrics(completed, dropped)
        qw_list.append(m["queue_wait_mean_ms"])
        swq_list.append(m["swq"])
        mr_list.append(m["deadline_miss_rate"])
    return {"name": name,
            "qw_mean": np.mean(qw_list), "qw_std": np.std(qw_list),
            "swq_mean": np.mean(swq_list), "swq_std": np.std(swq_list),
            "mr_mean": np.mean(mr_list)}


def run(save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)

    workloads = {
        "Robot Pipeline":   lambda seed: robot_pipeline(duration_s=30, seed=seed),
        "Staleness Stress": lambda seed: staleness_stress(n_tasks=200, seed=seed),
    }

    for wl_name, wl_fn in workloads.items():
        print(f"\n{'='*55}\nWorkload: {wl_name}\n{'='*55}")

        # Delta sweep
        delta_results = [eval_delta(d, wl_fn) for d in DELTAS]

        # Baselines
        baselines = [
            eval_baseline(fifo,   "FIFO",  wl_fn),
            eval_baseline(paes(), "PAES",  wl_fn),
        ]

        # ── Print table ───────────────────────────────────────────────────
        print(f"\n{'δ':>6}  {'QW (ms)':>10}  {'SWQ':>8}  {'MissRate':>10}")
        print("-" * 42)
        for r in delta_results:
            print(f"{r['delta']:>6.2f}  {r['qw_mean']:>10.1f}  "
                  f"{r['swq_mean']:>8.4f}  {r['mr_mean']:>10.3f}")
        print("\nBaselines:")
        for b in baselines:
            print(f"{'  '+b['name']:<8}  {b['qw_mean']:>10.1f}  "
                  f"{b['swq_mean']:>8.4f}  {b['mr_mean']:>10.3f}")

        # ── Figure 3a: Pareto QW vs SWQ ───────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax = axes[0]
        qw_vals  = [r["qw_mean"]  for r in delta_results]
        swq_vals = [r["swq_mean"] for r in delta_results]

        ax.plot(qw_vals, swq_vals, "o-", color="#1f77b4",
                linewidth=1.8, markersize=8, label="PAES-S (δ sweep)", zorder=3)
        for r in delta_results:
            ax.annotate(f"δ={r['delta']}", (r["qw_mean"], r["swq_mean"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)

        for b in baselines:
            style = "s" if b["name"] == "PAES" else "^"
            ax.plot(b["qw_mean"], b["swq_mean"], style, markersize=10,
                    label=b["name"], zorder=4)
            ax.annotate(b["name"], (b["qw_mean"], b["swq_mean"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=9)

        ax.set_xlabel("Mean Queue Wait (ms)  ← lower is better")
        ax.set_ylabel("Staleness-Weighted Quality  → higher is better")
        ax.set_title(f"Figure 3a — Pareto: QW vs SWQ\n({wl_name})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Figure 3b: SWQ vs δ ──────────────────────────────────────────
        ax2 = axes[1]
        ax2.errorbar(DELTAS, swq_vals,
                     yerr=[r["swq_std"] for r in delta_results],
                     fmt="o-", color="#1f77b4", capsize=4, linewidth=1.8,
                     markersize=7, label="PAES-S SWQ")

        ax2_twin = ax2.twinx()
        ax2_twin.errorbar(DELTAS, qw_vals,
                          yerr=[r["qw_std"] for r in delta_results],
                          fmt="s--", color="#d62728", capsize=4, linewidth=1.5,
                          markersize=7, alpha=0.8, label="Queue Wait (ms)")
        ax2_twin.set_ylabel("Queue Wait (ms)", color="#d62728")
        ax2_twin.tick_params(axis="y", labelcolor="#d62728")

        ax2.set_xlabel("δ (staleness weight)")
        ax2.set_ylabel("SWQ", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.set_title(f"Figure 3b — SWQ & QW vs. δ\n({wl_name})")
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        tag  = wl_name.lower().replace(" ", "_")
        path = os.path.join(save_dir, f"exp3_delta_sweep_{tag}.pdf")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Exp3] Saved → {path}")

        # ── Improvement over PAES ─────────────────────────────────────────
        paes_swq = next(b["swq_mean"] for b in baselines if b["name"] == "PAES")
        best = max(delta_results, key=lambda r: r["swq_mean"])
        gain = (best["swq_mean"] - paes_swq) / paes_swq * 100
        print(f"\nBest δ = {best['delta']:.2f}  →  SWQ {best['swq_mean']:.4f} "
              f"({gain:+.1f}% vs PAES)")


if __name__ == "__main__":
    run()
