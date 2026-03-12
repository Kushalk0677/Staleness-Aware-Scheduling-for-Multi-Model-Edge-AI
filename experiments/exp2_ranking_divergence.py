"""
Experiment 2 — Queue-Wait-Optimal ≠ Staleness-Optimal
======================================================
RQ: Does minimizing queue wait time maximize information quality?

Key claim: Scheduler rankings on queue-wait diverge from rankings on
staleness-weighted quality (SWQ). The winner on one metric loses on the other.

Method:
  - Run all 9 schedulers on synthetic + robot + staleness_stress workloads.
  - N_RUNS repetitions per workload (different random seeds).
  - Record: queue_wait_mean, staleness_mean, SWQ, deadline_miss_rate.
  - Compute rank correlation between the two metric orderings.
  - Plot: Figure 2 — scatter of queue_wait vs SWQ (one point per scheduler).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.family": "serif", "font.size": 11})
from scipy.stats import spearmanr

from schedulers.schedulers import SCHEDULERS, simulate
from utils.workloads import synthetic_uniform, robot_pipeline, staleness_stress
from utils.metrics import compute_all_metrics, summary_table

N_RUNS = 10   # set to 10 for paper (matches PAES methodology); reduce for quick test


def run_workload(workload_fn, workload_name: str, save_dir: str):
    agg = {name: [] for name in SCHEDULERS}

    for run_i in range(N_RUNS):
        tasks_template = workload_fn(seed=run_i * 7 + 13)
        for sched_name, (sched_fn, drop) in SCHEDULERS.items():
            import copy
            tasks = copy.deepcopy(tasks_template)
            completed, dropped = simulate(tasks, sched_fn, drop_enabled=drop)
            m = compute_all_metrics(completed, dropped)
            agg[sched_name].append(m)

    # ── Aggregate ─────────────────────────────────────────────────────────
    summary = {}
    for name, runs in agg.items():
        summary[name] = {
            "queue_wait_mean": np.mean([r["queue_wait_mean_ms"] for r in runs]),
            "queue_wait_std":  np.std( [r["queue_wait_mean_ms"] for r in runs]),
            "staleness_mean":  np.mean([r["staleness_mean"]     for r in runs]),
            "staleness_std":   np.std( [r["staleness_mean"]     for r in runs]),
            "swq_mean":        np.mean([r["swq"]                for r in runs]),
            "swq_std":         np.std( [r["swq"]                for r in runs]),
            "miss_rate":       np.mean([r["deadline_miss_rate"] for r in runs]),
            "drop_rate":       np.mean([r["drop_rate"]          for r in runs]),
        }

    # ── Rank analysis ─────────────────────────────────────────────────────
    names   = list(summary.keys())
    qw_vals = [summary[n]["queue_wait_mean"] for n in names]
    swq_vals= [summary[n]["swq_mean"]        for n in names]

    qw_ranks  = np.argsort(np.argsort(qw_vals))          # lower QW = better
    swq_ranks = np.argsort(np.argsort([-s for s in swq_vals]))  # higher SWQ = better

    rho, pval = spearmanr(qw_ranks, swq_ranks)

    print(f"\n{'='*60}")
    print(f"Workload: {workload_name}")
    print(f"{'='*60}")
    print(f"\n{'Scheduler':<18} {'QW (ms)':>10} {'Staleness':>10} "
          f"{'SWQ':>8} {'QW-rank':>8} {'SWQ-rank':>9}")
    print("-" * 67)
    for i, name in enumerate(names):
        s = summary[name]
        print(f"{name:<18} {s['queue_wait_mean']:>10.1f} {s['staleness_mean']:>10.3f} "
              f"{s['swq_mean']:>8.3f} {qw_ranks[i]+1:>8} {swq_ranks[i]+1:>9}")
    print(f"\nSpearman ρ (QW-rank vs SWQ-rank): {rho:.3f}  p={pval:.4f}")
    if abs(rho) < 0.5:
        print("→ LOW correlation: rankings diverge significantly (supports paper claim)")
    else:
        print("→ Moderate-high correlation: rankings partially agree")

    # ── Figure 2: scatter ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for i, name in enumerate(names):
        s = summary[name]
        ax.errorbar(
            s["queue_wait_mean"], s["swq_mean"],
            xerr=s["queue_wait_std"], yerr=s["swq_std"],
            fmt="o", color=colors[i], capsize=4, markersize=8, label=name
        )
        ax.annotate(name, (s["queue_wait_mean"], s["swq_mean"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Mean Queue Wait Time (ms)  ← lower is better")
    ax.set_ylabel("Staleness-Weighted Quality (SWQ)  → higher is better")
    ax.set_title(f"Figure 2 — Queue Wait vs. Information Quality\n"
                 f"({workload_name}, n={N_RUNS} runs, ρ={rho:.2f})")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    # Ideal direction arrow
    ax.annotate("", xy=(ax.get_xlim()[0]*0.97, ax.get_ylim()[1]*0.97),
                xytext=(ax.get_xlim()[0]*0.97, ax.get_ylim()[1]*0.90),
                arrowprops=dict(arrowstyle="->", color="green"))

    plt.tight_layout()
    tag = workload_name.lower().replace(" ", "_")
    path = os.path.join(save_dir, f"exp2_scatter_{tag}.pdf")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Exp2] Saved → {path}")

    return summary, rho


def run(save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    for wl_name, wl_fn in [
        ("Synthetic Uniform",  lambda seed: synthetic_uniform(n_tasks=600, seed=seed)),
        ("Robot Pipeline",     lambda seed: robot_pipeline(duration_s=30, seed=seed)),
        ("Staleness Stress",   lambda seed: staleness_stress(n_tasks=200, seed=seed)),
    ]:
        summary, rho = run_workload(wl_fn, wl_name, save_dir)
        results[wl_name] = {"summary": summary, "spearman_rho": rho}

    # ── Cross-workload rank-flip table ────────────────────────────────────
    print("\n\nTable II — Scheduler Rank Comparison (QW rank / SWQ rank)")
    print(f"{'Scheduler':<18}", end="")
    for wl in results:
        print(f"  {wl[:20]:<22}", end="")
    print()
    print("-" * (18 + 24 * len(results)))
    for sched in SCHEDULERS:
        print(f"{sched:<18}", end="")
        for wl, res in results.items():
            s = res["summary"]
            names = list(s.keys())
            qw   = [s[n]["queue_wait_mean"] for n in names]
            swq  = [s[n]["swq_mean"]        for n in names]
            qr   = sorted(names, key=lambda n: s[n]["queue_wait_mean"]).index(sched) + 1
            sr   = sorted(names, key=lambda n: -s[n]["swq_mean"]).index(sched) + 1
            flip = "⚠" if abs(qr - sr) >= 3 else ""
            print(f"  QW:{qr} / SWQ:{sr} {flip:<2}        ", end="")
        print()

    return results


if __name__ == "__main__":
    run()
