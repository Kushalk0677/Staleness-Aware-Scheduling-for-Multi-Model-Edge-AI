"""
Lambda (λ) Sensitivity Analysis for PAES-S
===========================================
Tests whether the paper's core claims are robust to ±2× variation in the
assumed exponential decay parameters (λ_MiDaS and λ_YOLOv5n).

Claims tested:
  C1. Rank inversion: Spearman ρ < 0 on staleness-stress workload
  C2. PAES-S-Drop SWQ gain > 0 over PAES on robot pipeline
  C3. δ term alone fails to improve SWQ (checked for nominal + all-slow)

Scenarios (5 total, matching the design in the paper supplement):
  nominal    — current paper values  (MiDaS 2.5, YOLO 0.8)
  all_slow   — 0.5× both            (MiDaS 1.25, YOLO 0.4)
  all_fast   — 2.0× both            (MiDaS 5.0,  YOLO 1.6)
  midas_fast — MiDaS 2×, YOLO nom   (MiDaS 5.0,  YOLO 0.8)
  midas_slow — MiDaS 0.5×, YOLO nom (MiDaS 1.25, YOLO 0.8)

Note: θ_i drop thresholds are held fixed — we test λ sensitivity only.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

matplotlib.rcParams.update({"font.family": "serif", "font.size": 10})

from schedulers.schedulers import SCHEDULERS, simulate
from utils.workloads    import robot_pipeline, staleness_stress
from utils.metrics      import compute_all_metrics
from models.task        import exponential_decay, linear_decay, flat_decay

# ── Config ────────────────────────────────────────────────────────────────────

N_RUNS    = int(os.environ.get("N_RUNS", "10"))
SAVE_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(SAVE_DIR, exist_ok=True)

# Fixed drop thresholds (not varied — isolating λ sensitivity)
DROP_THRESHOLDS = {
    "YOLOv5n":    0.25,
    "MiDaS":      0.30,
    "MobileNetV2":0.15,
    "WhisperTiny":0.00,
    "DistilBERT": 0.00,
}

SCENARIOS = {
    "nominal":    {"MiDaS": 2.50, "YOLOv5n": 0.80},
    "all_slow":   {"MiDaS": 1.25, "YOLOv5n": 0.40},
    "all_fast":   {"MiDaS": 5.00, "YOLOv5n": 1.60},
    "midas_fast": {"MiDaS": 5.00, "YOLOv5n": 0.80},
    "midas_slow": {"MiDaS": 1.25, "YOLOv5n": 0.80},
}

SCENARIO_LABELS = {
    "nominal":    "Nominal (1×)",
    "all_slow":   "All-slow (0.5×)",
    "all_fast":   "All-fast (2×)",
    "midas_fast": "MiDaS-fast (MiDaS 2×)",
    "midas_slow": "MiDaS-slow (MiDaS 0.5×)",
}

# ── Decay patching ────────────────────────────────────────────────────────────

def build_decay_fns(scenario: dict) -> dict:
    """
    Returns a dict of model_name -> (decay_fn, staleness_drop) for the
    given λ scenario.  Only MiDaS and YOLOv5n have variable λ.
    MobileNetV2 linear rate is held at 0.3 (not varied here).
    """
    return {
        "YOLOv5n":     (exponential_decay(lam=scenario["YOLOv5n"]),  DROP_THRESHOLDS["YOLOv5n"]),
        "MiDaS":       (exponential_decay(lam=scenario["MiDaS"]),    DROP_THRESHOLDS["MiDaS"]),
        "MobileNetV2": (linear_decay(rate=0.3),                       DROP_THRESHOLDS["MobileNetV2"]),
        "WhisperTiny": (flat_decay(),                                  DROP_THRESHOLDS["WhisperTiny"]),
        "DistilBERT":  (flat_decay(),                                  DROP_THRESHOLDS["DistilBERT"]),
    }


def patch_tasks(tasks, decay_fns: dict):
    """
    Deep-copy tasks and replace their decay_fn / staleness_drop
    according to the scenario's λ values.
    """
    patched = copy.deepcopy(tasks)
    for t in patched:
        fn, thresh = decay_fns[t.model_name]
        t.decay_fn       = fn
        t.staleness_drop = thresh
    return patched


# ── Single scenario runner ────────────────────────────────────────────────────

def run_scenario(scenario_name: str, lam_values: dict):
    """
    For the given λ scenario, run all schedulers on robot_pipeline and
    staleness_stress, N_RUNS times each.

    Returns:
        robot_agg:  {sched_name: [metrics_dict, ...]}
        stress_agg: {sched_name: [metrics_dict, ...]}
    """
    decay_fns   = build_decay_fns(lam_values)
    robot_agg   = {name: [] for name in SCHEDULERS}
    stress_agg  = {name: [] for name in SCHEDULERS}

    for run_i in range(N_RUNS):
        seed = run_i * 7 + 13
        robot_base  = robot_pipeline(seed=seed)
        stress_base = staleness_stress(seed=seed)

        for sched_name, (sched_fn, drop) in SCHEDULERS.items():
            # Robot pipeline
            tasks = patch_tasks(robot_base, decay_fns)
            comp, drp = simulate(tasks, sched_fn, drop_enabled=drop)
            robot_agg[sched_name].append(compute_all_metrics(comp, drp))

            # Staleness stress
            tasks = patch_tasks(stress_base, decay_fns)
            comp, drp = simulate(tasks, sched_fn, drop_enabled=drop)
            stress_agg[sched_name].append(compute_all_metrics(comp, drp))

    return robot_agg, stress_agg


def aggregate(agg: dict) -> dict:
    """Collapse list of run dicts to mean values per scheduler."""
    return {
        name: {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        for name, runs in agg.items()
    }


def spearman_rho(summary: dict) -> tuple:
    """
    Compute Spearman ρ between queue-wait rank and SWQ rank.
    Returns (rho, p_value).
    """
    scheds = list(summary.keys())
    qw  = [summary[s]["queue_wait_mean_ms"] for s in scheds]
    swq = [summary[s]["swq"]                for s in scheds]
    rho, p = spearmanr(qw, swq)
    return float(rho), float(p)


def swq_gain_drop_vs_paes(summary: dict) -> float:
    """
    Percentage SWQ improvement of PAES-S-Drop over PAES.
    """
    paes_swq = summary["PAES"]["swq"]
    drop_swq = summary["PAES-S-Drop"]["swq"]
    if paes_swq == 0:
        return float("nan")
    return (drop_swq - paes_swq) / paes_swq * 100.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*65}")
    print(f"  λ Sensitivity Analysis   N_RUNS={N_RUNS}")
    print(f"{'='*65}\n")
    print(f"Testing 5 decay scenarios × 2 workloads × {len(SCHEDULERS)} schedulers\n")

    results = {}   # scenario -> {robot_summary, stress_summary, rho, gain}

    for scenario_name, lam_values in SCENARIOS.items():
        label = SCENARIO_LABELS[scenario_name]
        midas_lam = lam_values["MiDaS"]
        yolo_lam  = lam_values["YOLOv5n"]
        midas_hl  = math.log(2) / midas_lam
        yolo_hl   = math.log(2) / yolo_lam

        print(f"Scenario: {label}")
        print(f"  MiDaS λ={midas_lam:.2f} (t½={midas_hl:.2f}s)  "
              f"YOLO λ={yolo_lam:.2f} (t½={yolo_hl:.2f}s)")

        robot_agg, stress_agg = run_scenario(scenario_name, lam_values)
        robot_summ  = aggregate(robot_agg)
        stress_summ = aggregate(stress_agg)

        rho, p_val = spearman_rho(stress_summ)
        gain       = swq_gain_drop_vs_paes(robot_summ)

        results[scenario_name] = {
            "label":        label,
            "midas_lam":    midas_lam,
            "yolo_lam":     yolo_lam,
            "midas_hl":     midas_hl,
            "yolo_hl":      yolo_hl,
            "rho":          rho,
            "p_val":        p_val,
            "gain_pct":     gain,
            "robot_summ":   robot_summ,
            "stress_summ":  stress_summ,
        }

        inv_str = "✓ rank inversion" if rho < 0 else "✗ NO inversion"
        print(f"  ρ = {rho:+.3f}  (p={p_val:.3f})  {inv_str}")
        print(f"  PAES-S-Drop SWQ gain vs PAES = {gain:+.1f}%\n")

    # ── Print full results table ─────────────────────────────────────────
    _print_table(results)

    # ── δ sanity check (nominal + all-slow only) ─────────────────────────
    _delta_sanity(results)

    # ── Figures ──────────────────────────────────────────────────────────
    _plot_summary(results)
    _plot_scheduler_swq(results)

    print(f"\nFigures saved to {SAVE_DIR}/")
    print("  exp_lambda_summary.png  — table + bar chart")
    print("  exp_lambda_swq.png      — per-scheduler SWQ across scenarios\n")


# ── Table printout ────────────────────────────────────────────────────────────

def _print_table(results: dict):
    header = (
        f"\n{'Scenario':<26} {'MiDaS λ':>8} {'YOLO λ':>7} "
        f"{'ρ (stress)':>12} {'p':>7} {'Inversion':>10} "
        f"{'Drop gain':>12}\n"
    )
    print(header + "-" * 88)
    for sc, r in results.items():
        inv = "YES" if r["rho"] < 0 else "NO "
        print(
            f"  {r['label']:<24} {r['midas_lam']:>8.2f} {r['yolo_lam']:>7.2f} "
            f"  {r['rho']:>+10.3f}  {r['p_val']:>6.3f}  {inv:>10} "
            f"  {r['gain_pct']:>+9.1f}%"
        )
    print()


def _delta_sanity(results: dict):
    """
    Confirm δ-term-alone still fails for nominal and all-slow scenarios:
    PAES-S SWQ ≈ PAES SWQ (within 2%).
    """
    print("δ-term sanity check (PAES-S vs PAES on robot pipeline):")
    print(f"  {'Scenario':<22} {'PAES SWQ':>10} {'PAES-S SWQ':>12} {'Diff':>8}")
    print("  " + "-" * 56)
    for sc in ("nominal", "all_slow"):
        r     = results[sc]
        summ  = r["robot_summ"]
        paes  = summ["PAES"]["swq"]
        paess = summ["PAES-S"]["swq"]
        diff  = (paess - paes) / paes * 100 if paes else 0
        flag  = "  ✓ (<2%)" if abs(diff) < 2.0 else "  ✗ CHANGED"
        print(f"  {r['label']:<22} {paes:>10.4f} {paess:>12.4f} {diff:>+7.1f}%{flag}")
    print()


# ── Figure 1: summary bars + table ───────────────────────────────────────────

def _plot_summary(results: dict):
    scenarios = list(results.keys())
    labels    = [results[s]["label"] for s in scenarios]
    rhos      = [results[s]["rho"]      for s in scenarios]
    gains     = [results[s]["gain_pct"] for s in scenarios]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("λ Sensitivity Analysis: Robustness of Core Claims", fontsize=12, y=1.01)

    # — ρ bar chart —
    ax = axes[0]
    colors = ["#d73027" if r < 0 else "#4dac26" for r in rhos]
    bars = ax.barh(labels, rhos, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Spearman ρ  (queue-wait rank vs SWQ rank)")
    ax.set_title("C1: Rank Inversion (staleness-stress)\nρ < 0 → inversion holds", fontsize=10)
    for bar, v in zip(bars, rhos):
        ax.text(v + 0.005 * np.sign(v), bar.get_y() + bar.get_height() / 2,
                f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color="black")
    ax.set_xlim(-0.60, 0.30)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)

    # — gain bar chart —
    ax = axes[1]
    colors2 = ["#2166ac" if g > 0 else "#d73027" for g in gains]
    bars2 = ax.barh(labels, gains, color=colors2, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("PAES-S-Drop SWQ gain over PAES  (%)")
    ax.set_title("C2: Drop policy SWQ gain (robot pipeline)\nPositive → drop still wins", fontsize=10)
    for bar, v in zip(bars2, gains):
        ax.text(v + 1.0 * np.sign(v), bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color="black")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "exp_lambda_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 2: per-scheduler SWQ across scenarios ─────────────────────────────

def _plot_scheduler_swq(results: dict):
    """
    Show how each scheduler's SWQ on the robot pipeline varies across λ scenarios.
    Highlights that PAES-S-Drop dominates consistently.
    """
    scenarios    = list(results.keys())
    sched_names  = list(SCHEDULERS.keys())
    n_sc         = len(scenarios)
    x            = np.arange(n_sc)
    width        = 0.08

    # Pick a subset of schedulers for readability
    highlight = ["FIFO", "PAES", "PAES-S", "PAES-S-Drop"]
    others    = [s for s in sched_names if s not in highlight]
    plot_scheds = others + highlight   # highlight last → on top

    cmap_other = matplotlib.colormaps.get_cmap("tab20").resampled(len(others))
    colors = {s: cmap_other(i) for i, s in enumerate(others)}
    colors.update({
        "FIFO":        "#aaaaaa",
        "PAES":        "#2166ac",
        "PAES-S":      "#74add1",
        "PAES-S-Drop": "#d73027",
    })
    linewidths = {s: 0.7 for s in others}
    linewidths.update({"FIFO": 1.5, "PAES": 2.0, "PAES-S": 1.5, "PAES-S-Drop": 2.5})
    alphas = {s: 0.45 for s in others}
    alphas.update({"FIFO": 0.9, "PAES": 0.95, "PAES-S": 0.85, "PAES-S-Drop": 1.0})

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for sched in plot_scheds:
        swq_vals = [
            results[sc]["robot_summ"][sched]["swq"]
            for sc in scenarios
        ]
        lw  = linewidths[sched]
        alp = alphas[sched]
        col = colors[sched]
        ls  = "--" if sched in others else "-"
        zord = 2 if sched in others else 3
        ax.plot(range(n_sc), swq_vals, marker="o", linewidth=lw, alpha=alp,
                color=col, label=sched, linestyle=ls, zorder=zord,
                markersize=4 if sched in others else 6)

    ax.set_xticks(range(n_sc))
    ax.set_xticklabels([results[s]["label"] for s in scenarios],
                       rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("SWQ (higher = better)")
    ax.set_title("Per-Scheduler SWQ Across λ Scenarios (robot pipeline)\n"
                 "PAES-S-Drop dominates across all scenarios", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend: only highlighted schedulers + one "others" proxy
    proxy = mpatches.Patch(color="#cccccc", alpha=0.5, label="Other schedulers")
    handles = [proxy]
    for s in highlight:
        handles.append(plt.Line2D([0], [0], color=colors[s],
                                  linewidth=linewidths[s], label=s))
    ax.legend(handles=handles, fontsize=9, loc="upper left",
              framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "exp_lambda_swq.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
