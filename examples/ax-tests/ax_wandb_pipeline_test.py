#!/usr/bin/env python3
"""Full W&B pipeline integration test for Ax sweeps.

This script tests that benchmark problems work with W&B logging:
  Local Ax sweep loop -> wandb.init() -> wandb.log() -> wandb.finish()

Since "ax" is a local sweep method (not a W&B backend method), we use our
local sweep controller (next_runs) while logging to W&B.

Runs in offline mode (WANDB_MODE=offline) to avoid requiring credentials.

Usage:
    python ax_wandb_pipeline_test.py --problem dtlz2 --trials 10
    python ax_wandb_pipeline_test.py --problem c2dtlz2 --trials 10
"""

import argparse
import os
import sys
import tempfile
import warnings

# Set offline mode BEFORE importing wandb
os.environ["WANDB_MODE"] = "offline"

import numpy as np

try:
    import torch
except ImportError:
    print("Error: torch is required.")
    sys.exit(1)

try:
    import wandb
except ImportError:
    print("Error: wandb is required.")
    print("Install with: pip install wandb")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("Error: pyyaml is required.")
    sys.exit(1)

# Add path to import sweeps
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from benchmark_problems import MOO_PROBLEM_REGISTRY  # noqa: E402
from sweeps.config import SweepConfig  # noqa: E402
from sweeps.run import next_runs, RunState, SweepRun  # noqa: E402

# Suppress warnings
warnings.filterwarnings("ignore")

# Map problem names to YAML config files
YAML_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
PROBLEM_YAML_MAP = {
    "dtlz2": os.path.join(YAML_CONFIG_DIR, "ax_dtlz2_moo.yaml"),
    "c2dtlz2": os.path.join(YAML_CONFIG_DIR, "ax_c2dtlz2_moo.yaml"),
    "welded_beam": os.path.join(YAML_CONFIG_DIR, "ax_welded_beam_moo.yaml"),
}


def run_wandb_pipeline_test(
    problem_name: str,
    num_trials: int,
    random_seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Run benchmark problem with Ax sweep and W&B logging.

    This combines:
    1. Local Ax sweep loop using next_runs() with YAML config
    2. W&B logging for each run (wandb.init, wandb.log, wandb.finish)

    Args:
        problem_name: Name of the MOO problem to test
        num_trials: Number of optimization trials
        random_seed: Random seed for reproducibility
        verbose: Whether to print detailed output

    Returns:
        Dict with test results
    """
    if problem_name not in MOO_PROBLEM_REGISTRY:
        raise ValueError(f"Unknown problem: {problem_name}")

    if problem_name not in PROBLEM_YAML_MAP:
        raise ValueError(f"No YAML config for problem: {problem_name}")

    problem = MOO_PROBLEM_REGISTRY[problem_name]()

    # Load and validate config from YAML
    yaml_path = PROBLEM_YAML_MAP[problem_name]
    with open(yaml_path) as f:
        raw_config = yaml.safe_load(f)
    config = SweepConfig(raw_config)

    print("=" * 70)
    print("W&B PIPELINE INTEGRATION TEST")
    print("=" * 70)
    print(f"Problem: {problem.name} ({problem.dim}D, {problem.num_objectives} objectives)")
    print(f"YAML config: {os.path.basename(yaml_path)}")
    print(f"Trials: {num_trials}")
    print(f"Mode: offline (WANDB_MODE=offline)")
    print("=" * 70)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Track runs for the sweep
    sweep_runs = []
    wandb_run_ids = []
    results = []
    all_Y = []
    all_feasible = []

    print(f"\nRunning {num_trials} trials with W&B logging...")

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["WANDB_DIR"] = tmpdir

        for trial in range(num_trials):
            # Get next suggestion from Ax via next_runs
            suggestions = next_runs(
                config, sweep_runs, validate=False, n=1, random_seed=random_seed
            )

            if not suggestions:
                print(f"  Warning: No suggestion at trial {trial}")
                break

            suggestion = suggestions[0]
            params = {k: v["value"] for k, v in suggestion.config.items()}

            # Run with W&B logging
            run = wandb.init(
                project="ax-pipeline-test",
                config=params,
                reinit=True,
            )
            wandb_run_ids.append(run.id)

            # Evaluate the benchmark problem
            result = problem.evaluate(params)

            # Log metrics through wandb.log()
            wandb.log(result)
            results.append(result)

            # Extract objective values for HV tracking
            obj_values = [result[name] for name in problem.objective_names]
            all_Y.append(obj_values)

            # Check feasibility
            if problem.num_constraints > 0:
                constraint_values = [result[name] for name in problem.constraint_names]
                is_feasible = all(g <= 0 for g in constraint_values)
            else:
                is_feasible = True
            all_feasible.append(is_feasible)

            # Finish the W&B run
            wandb.finish(quiet=True)

            # Create SweepRun for next iteration
            summary_metrics = {}
            for name in problem.objective_names:
                summary_metrics[name] = result[name]
            for name in problem.constraint_names:
                summary_metrics[name] = result[name]

            sweep_run = SweepRun(
                state=RunState.finished,
                config={k: {"value": v} for k, v in params.items()},
                summary_metrics=summary_metrics,
            )
            sweep_runs.append(sweep_run)

            # Compute hypervolume
            Y_array = np.array(all_Y)
            feasible_mask = np.array(all_feasible)
            if np.any(feasible_mask):
                pareto_Y = problem.get_pareto_front(Y_array, feasible_mask)
                hv = problem.compute_hypervolume(pareto_Y) if len(pareto_Y) > 0 else 0.0
            else:
                hv = 0.0

            if verbose or (trial + 1) % 5 == 0:
                print(f"  Trial {trial + 1}/{num_trials}: HV={hv:.4f}")

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nIterations completed: {len(results)}")
    print(f"W&B runs created: {len(wandb_run_ids)}")

    # Check all metrics logged correctly
    metrics_valid = all(
        all(name in r for name in problem.objective_names)
        for r in results
    )
    print(f"All metrics logged: {'PASS' if metrics_valid else 'FAIL'}")

    # Check constraints if applicable
    if problem.num_constraints > 0:
        constraints_valid = all(
            all(name in r for name in problem.constraint_names)
            for r in results
        )
        print(f"Constraints logged: {'PASS' if constraints_valid else 'FAIL'}")
    else:
        constraints_valid = True

    # Final hypervolume
    Y_array = np.array(all_Y)
    feasible_mask = np.array(all_feasible)
    if np.any(feasible_mask):
        pareto_Y = problem.get_pareto_front(Y_array, feasible_mask)
        final_hv = problem.compute_hypervolume(pareto_Y) if len(pareto_Y) > 0 else 0.0
    else:
        final_hv = 0.0
    print(f"Final hypervolume: {final_hv:.4f}")

    # Overall pass/fail
    all_pass = metrics_valid and constraints_valid and len(results) == num_trials

    print("\n" + "=" * 70)
    if all_pass:
        print("CONCLUSION: PASS")
        print("Ax sweep with W&B logging works correctly.")
    else:
        print("CONCLUSION: FAIL")
        print("Issues detected in W&B pipeline.")
    print("=" * 70)

    return {
        "problem": problem_name,
        "num_trials": num_trials,
        "results": results,
        "final_hypervolume": final_hv,
        "wandb_run_ids": wandb_run_ids,
        "metrics_valid": metrics_valid,
        "constraints_valid": constraints_valid,
        "all_pass": all_pass,
    }


def main():
    parser = argparse.ArgumentParser(
        description="W&B pipeline integration test for Ax sweeps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="dtlz2",
        choices=list(PROBLEM_YAML_MAP.keys()),
        help="MOO problem to test",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    args = parser.parse_args()

    result = run_wandb_pipeline_test(
        problem_name=args.problem,
        num_trials=args.trials,
        random_seed=args.seed,
        verbose=args.verbose,
    )

    sys.exit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
