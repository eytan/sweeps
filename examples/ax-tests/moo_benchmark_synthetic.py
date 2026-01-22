#!/usr/bin/env python3
"""Benchmark for Ax MOO on multi-objective test functions.

This script evaluates the performance of Ax-based Multi-Objective Bayesian
Optimization on standard MOO test problems.

The benchmark runs multiple replications with different random seeds to ensure
statistical robustness and tracks:
- Hypervolume progression (main MOO performance metric)
- Final Pareto front quality

Usage:
    # Quick test (10 trials, 3 replications)
    python moo_benchmark_synthetic.py --trials 10 --replications 3

    # Standard benchmark (50 trials, 25 replications)
    python moo_benchmark_synthetic.py --trials 50 --replications 25

    # Custom initial points
    python moo_benchmark_synthetic.py --init 12

Output:
    - Results in JSON format: benchmark_results/moo/results_{problem_name}.json

Note:
    To generate plots from the saved results, use:
    python plot_moo_results.py --results-dir benchmark_results/moo
"""

import argparse
import json
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
except ImportError:
    print("Error: torch is required for this benchmark.")
    print("Install with: pip install torch")
    sys.exit(1)

# Add path to import sweeps
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from benchmark_problems import (  # noqa: E402
    MultiObjectiveBoTorchProblem,
    get_moo_problems,
    list_available_moo_problems,
    MOO_PROBLEM_REGISTRY,
)
from benchmark_utils import convert_to_json_serializable  # noqa: E402
from sweeps.ax_search import ax_search_next_runs  # noqa: E402
from sweeps.run import RunState, SweepRun  # noqa: E402

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_moo_replication(
    problem: MultiObjectiveBoTorchProblem,
    num_trials: int,
    num_init: int,
    random_seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single MOO replication, tracking hypervolume at each trial.

    Args:
        problem: Multi-objective test problem
        num_trials: Total number of trials to run
        num_init: Number of initial points (Ax handles initialization internally)
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        Dict containing:
            - 'hypervolume_curve': List of hypervolume values at each iteration
            - 'final_pareto_Y': Final Pareto front objective values
            - 'final_pareto_X': Final Pareto front parameter values
            - 'all_Y': All objective values evaluated
            - 'all_X': All parameter values evaluated
            - 'all_feasible': Boolean array of feasibility for each point
            - 'seed': Random seed used
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    config = problem.create_sweep_config("ax")

    # Track results
    runs = []
    all_Y = []  # All objective values
    all_X = []  # All parameter values
    all_feasible = []  # Feasibility of each point
    hypervolume_curve = []

    # Get metric names (objectives + constraints)
    objective_names = problem.objective_names
    constraint_names = problem.constraint_names

    for trial in range(num_trials):
        if verbose and (trial + 1) % 10 == 0:
            print(f"    Trial {trial + 1}/{num_trials}")

        try:
            # Generate next suggestion using Ax
            suggestions = ax_search_next_runs(
                runs, config, n=1, random_seed=random_seed
            )
            if not suggestions:
                if verbose:
                    print(f"    Warning: No suggestions at trial {trial}")
                break

            params = {k: v["value"] for k, v in suggestions[0].config.items()}

            # Evaluate objectives and constraints
            result = problem.evaluate(params)

            # Extract objective values
            obj_values = [result[name] for name in objective_names]
            all_Y.append(obj_values)
            all_X.append([params[f"x{i}"] for i in range(problem.dim)])

            # Check feasibility (all constraints <= 0)
            if constraint_names:
                constraint_values = [result[name] for name in constraint_names]
                is_feasible = all(g <= 0 for g in constraint_values)
            else:
                is_feasible = True
                constraint_values = []

            all_feasible.append(is_feasible)

            # Create summary metrics dict for the run
            summary_metrics = {}
            for name in objective_names:
                summary_metrics[name] = result[name]
            for name in constraint_names:
                summary_metrics[name] = result[name]

            # Create run with result
            run = SweepRun(
                state=RunState.finished,
                config={k: {"value": v} for k, v in params.items()},
                summary_metrics=summary_metrics,
            )
            runs.append(run)

            # Compute hypervolume of current Pareto front (feasible points only)
            Y_array = np.array(all_Y)
            feasible_mask = np.array(all_feasible)

            if np.any(feasible_mask):
                pareto_Y = problem.get_pareto_front(Y_array, feasible_mask)
                if len(pareto_Y) > 0:
                    hv = problem.compute_hypervolume(pareto_Y)
                else:
                    hv = 0.0
            else:
                hv = 0.0

            hypervolume_curve.append(hv)

        except Exception as e:
            if verbose:
                print(f"    Warning: Trial {trial} failed with error: {e}")
            # Keep previous hypervolume value
            if hypervolume_curve:
                hypervolume_curve.append(hypervolume_curve[-1])
            else:
                hypervolume_curve.append(0.0)
            continue

    # Compute final Pareto front
    Y_array = np.array(all_Y) if all_Y else np.array([])
    X_array = np.array(all_X) if all_X else np.array([])
    feasible_mask = np.array(all_feasible) if all_feasible else np.array([])

    if len(Y_array) > 0 and np.any(feasible_mask):
        # Get Pareto front from feasible points
        feasible_Y = Y_array[feasible_mask]
        feasible_X = X_array[feasible_mask]

        pareto_Y = problem.get_pareto_front(Y_array, feasible_mask)

        # Find indices of Pareto points in feasible_Y to get corresponding X
        pareto_X = []
        pareto_Y_list = pareto_Y.tolist() if len(pareto_Y) > 0 else []
        for py in pareto_Y_list:
            for i, fy in enumerate(feasible_Y):
                if np.allclose(py, fy):
                    pareto_X.append(feasible_X[i].tolist())
                    break
    else:
        pareto_Y = np.array([])
        pareto_X = []

    return {
        "hypervolume_curve": hypervolume_curve,
        "final_pareto_Y": pareto_Y.tolist() if len(pareto_Y) > 0 else [],
        "final_pareto_X": pareto_X,
        "all_Y": Y_array.tolist() if len(Y_array) > 0 else [],
        "all_X": X_array.tolist() if len(X_array) > 0 else [],
        "all_feasible": feasible_mask.tolist() if len(feasible_mask) > 0 else [],
        "seed": random_seed,
    }


def benchmark_moo(
    problem: MultiObjectiveBoTorchProblem,
    num_trials: int,
    num_init: int,
    num_replications: int,
    random_seeds: Optional[List[int]] = None,
) -> Dict[str, List[Dict]]:
    """Run MOO benchmark with Ax across replications.

    Args:
        problem: Multi-objective test problem
        num_trials: Number of optimization trials per replication
        num_init: Number of initial points for Ax
        num_replications: Number of replications
        random_seeds: List of random seeds (defaults to range(num_replications))

    Returns:
        Dict mapping method name ('ax') to list of replication results
    """
    if random_seeds is None:
        random_seeds = list(range(num_replications))

    results = {"ax": []}

    print(f"\n  Running Ax on {problem.name}...")
    for replication_idx, seed in enumerate(random_seeds):
        replication_result = run_moo_replication(
            problem=problem,
            num_trials=num_trials,
            num_init=num_init,
            random_seed=seed,
            verbose=False,
        )

        results["ax"].append(replication_result)
        final_hv = replication_result["hypervolume_curve"][-1] if replication_result["hypervolume_curve"] else 0.0
        print(
            f"    Replication {replication_idx + 1}/{num_replications} (seed={seed}) "
            f"Final HV: {final_hv:.6f}"
        )

    return results


def compute_moo_statistics(results: Dict[str, List[Dict]]) -> Dict[str, Dict[str, Any]]:
    """Compute aggregate statistics across multiple replications.

    Args:
        results: Dict mapping method to list of replication results

    Returns:
        Dict mapping method to statistics
    """
    stats = {}

    for method, replications in results.items():
        # Extract hypervolume curves
        hv_curves = [r["hypervolume_curve"] for r in replications]
        final_hvs = [curve[-1] if curve else 0.0 for curve in hv_curves]
        num_replications = len(replications)

        # Pad curves to same length if needed
        max_len = max(len(c) for c in hv_curves) if hv_curves else 0
        padded_curves = []
        for curve in hv_curves:
            if len(curve) < max_len:
                # Pad with last value
                padded = curve + [curve[-1]] * (max_len - len(curve))
            else:
                padded = curve
            padded_curves.append(padded)

        padded_curves = np.array(padded_curves) if padded_curves else np.array([[]])

        # Compute statistics
        stats[method] = {
            "mean_hv_curve": np.mean(padded_curves, axis=0).tolist() if max_len > 0 else [],
            "sem_hv_curve": (np.std(padded_curves, axis=0) / np.sqrt(num_replications)).tolist() if max_len > 0 else [],
            "median_hv_curve": np.median(padded_curves, axis=0).tolist() if max_len > 0 else [],
            "final_hv_mean": float(np.mean(final_hvs)),
            "final_hv_sem": float(np.std(final_hvs) / np.sqrt(num_replications)),
            "final_hv_median": float(np.median(final_hvs)),
            "num_replications": num_replications,
        }

    return stats


def save_moo_results(
    problem: MultiObjectiveBoTorchProblem,
    results: Dict[str, List[Dict]],
    stats: Dict[str, Dict[str, Any]],
    num_trials: int,
    num_init: int,
    output_dir: str,
) -> str:
    """Save MOO benchmark results to JSON file.

    Args:
        problem: The benchmark problem
        results: Dict mapping method to list of replication results
        stats: Dict mapping method to statistics
        num_trials: Number of trials per replication
        num_init: Number of initial points
        output_dir: Directory to save results

    Returns:
        Path to the saved file
    """
    results_path = os.path.join(output_dir, f"results_{problem.name}.json")

    detailed_data = {
        "metadata": {
            **convert_to_json_serializable(problem.get_metadata()),
            "num_trials": num_trials,
            "num_init": num_init,
            "num_replications": stats[list(stats.keys())[0]]["num_replications"] if stats else 0,
        },
        "results": convert_to_json_serializable(results),
        "stats": convert_to_json_serializable(stats),
    }

    with open(results_path, "w") as f:
        json.dump(detailed_data, f, indent=2)

    print(f"  Results saved to {results_path}")
    return results_path


def main():
    """Run MOO benchmark using Ax."""
    # Build choices for --problems argument
    problem_choices = list(MOO_PROBLEM_REGISTRY.keys()) + ["all", "moo"]

    parser = argparse.ArgumentParser(
        description="Benchmark Ax MOO on multi-objective test functions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=["dtlz2"],
        choices=problem_choices,
        help="Problems to benchmark (default: dtlz2). Use 'all' for all MOO problems.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials per replication",
    )
    parser.add_argument(
        "--init",
        type=int,
        default=8,
        help="Number of initial points for Ax",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=25,
        help="Number of replications (for statistical robustness)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results/moo",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--list-problems",
        action="store_true",
        help="List available problems and exit",
    )
    args = parser.parse_args()

    # Handle --list-problems
    if args.list_problems:
        print("\nAvailable MOO problems:")
        print("-" * 70)
        for name, desc in list_available_moo_problems().items():
            print(f"  {name}: {desc}")
        print("\nSpecial keywords:")
        print("  all, moo: All available MOO problems")
        return

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)

    # Get problems from command line
    try:
        problems = get_moo_problems(args.problems)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not problems:
        print("Error: No problems selected", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("MULTI-OBJECTIVE OPTIMIZATION BENCHMARK")
    print("=" * 70)
    print(f"Problems: {', '.join([p.name for p in problems])}")
    print(f"Trials per replication: {args.trials}")
    print(f"Initial points: {args.init}")
    print(f"Replications: {args.replications}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)

    # Run benchmarks
    all_results = {}
    all_stats = {}

    for problem in problems:
        print(f"\n{'=' * 70}")
        print(f"Benchmarking on {problem.name}")
        print(f"  Dimensionality: {problem.dim}D")
        print(f"  Objectives: {problem.num_objectives} ({', '.join(problem.objective_names)})")
        print(f"  Constraints: {problem.num_constraints}")
        print(f"  Reference point: {problem.ref_point}")
        print(f"{'=' * 70}")

        results = benchmark_moo(
            problem=problem,
            num_trials=args.trials,
            num_init=args.init,
            num_replications=args.replications,
            random_seeds=[args.seed + i for i in range(args.replications)],
        )

        stats = compute_moo_statistics(results)

        # Save results for this problem
        save_moo_results(
            problem, results, stats, args.trials, args.init, args.output_dir
        )

        all_results[problem.name] = results
        all_stats[problem.name] = stats

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for problem_name in all_results.keys():
        print(f"\n{problem_name}:")
        stats = all_stats[problem_name]
        ax_stats = stats["ax"]
        print(
            f"  Final Hypervolume: {ax_stats['final_hv_mean']:.6f} +/- {ax_stats['final_hv_sem']:.6f}"
        )

    print("\n" + "=" * 70)
    print(f"Benchmark complete! Results saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
