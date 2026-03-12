"""
PAES-S: Priority-Aware Edge Scheduler with Staleness Awareness
==============================================================
Master runner — executes all four experiments in sequence.

Usage:
    python run_all.py            # full run (N_RUNS=10 per experiment)
    python run_all.py --quick    # fast smoke-test (N_RUNS=2)

Results saved to: ./results/
"""

import sys
import os
import time
import argparse
import importlib.util

# Quick-mode: patch N_RUNS before importing experiments
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                    help="Use N_RUNS=2 for a fast smoke-test")
args = parser.parse_args()

def patch_nruns(module_path, n=2):
    spec = importlib.util.spec_from_file_location("mod", module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.N_RUNS = n
    return mod

SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)


def section(title):
    print(f"\n{'#'*65}")
    print(f"#  {title}")
    print(f"{'#'*65}\n")


def run_experiment(label, module_path):
    section(label)
    t0  = time.time()
    mod = patch_nruns(module_path, n=2 if args.quick else 10)
    mod.run(SAVE_DIR)
    elapsed = time.time() - t0
    print(f"\n  ✓ {label} completed in {elapsed:.1f}s")


if __name__ == "__main__":
    print("=" * 65)
    print("  PAES-S Experiment Suite")
    if args.quick:
        print("  Mode: QUICK (N_RUNS=2 — for smoke-testing only)")
    else:
        print("  Mode: FULL  (N_RUNS=10 — matches paper methodology)")
    print("=" * 65)

    t_start = time.time()

    run_experiment(
        "Experiment 1 — Staleness Decay Characterisation",
        "experiments/exp1_decay_characterization.py"
    )
    run_experiment(
        "Experiment 2 — Queue-Wait vs. Staleness-Weighted Quality Ranking",
        "experiments/exp2_ranking_divergence.py"
    )
    run_experiment(
        "Experiment 3 — PAES-S Delta Sweep (Pareto Frontier)",
        "experiments/exp3_delta_sweep.py"
    )
    run_experiment(
        "Experiment 4 — Task Drop Threshold Analysis",
        "experiments/exp4_drop_threshold.py"
    )

    total = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  All experiments complete in {total:.1f}s")
    print(f"  Results → ./{SAVE_DIR}/")
    print(f"{'='*65}")
