#!/usr/bin/env python3
"""Benchmark comparing bayes and ax on synthetic test functions.

This script evaluates the performance of sklearn-based Bayesian optimization
(method='bayes') versus ax-platform Bayesian optimization (method='ax')
on standard synthetic test functions.

The benchmark runs multiple replications with different random seeds to ensure
statistical robustness and compares:
- Final solution quality (how close to the known global optimum)
- Reliability (consistency across different random seeds)

Available Test Functions:
- Branin (2D): Easy benchmark with 3 global minima (continuous, minimization)
- Hartmann6 (6D): Medium difficulty with 1 global minimum (continuous, minimization)
- AckleyMixed (23D): Mixed search space with 20 binary + 3 continuous params (minimization)
- Labs (20D): Fully discrete (binary) search space (maximization)

Usage:
    # Run all problems (default)
    python bayes_benchmark_synthetic.py

    # Run specific problem
    python bayes_benchmark_synthetic.py --problems ackley_mixed_23d

    # Run multiple specific problems
    python bayes_benchmark_synthetic.py --problems branin labs_20d

    # Run all available problems (explicit)
    python bayes_benchmark_synthetic.py --problems all

    # Run only mixed/discrete problems
    python bayes_benchmark_synthetic.py --problems mixed

    # Run only continuous problems
    python bayes_benchmark_synthetic.py --problems continuous

    # Quick test (10 trials, 3 replications)
    python bayes_benchmark_synthetic.py --trials 10 --replications 3

    # Standard benchmark (50 trials, 100 replications)
    python bayes_benchmark_synthetic.py --trials 50 --replications 100

Output:
    - Results in JSON format: results_{problem_name}.json (one file per problem)

Example Output Structure:
    benchmark_results/
    ├── results_Branin_2D.json
    ├── results_Hartmann6_6D.json
    ├── results_AckleyMixed_23D.json
    └── results_Labs_20D.json

Note:
    To generate convergence plots from the saved results, use:
    python plot_benchmark_results.py --results-dir benchmark_results
"""

import argparse
import json
import os
import sys
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    print("Error: torch is required for this benchmark.")
    print("Install with: pip install torch")
    sys.exit(1)

# Add path to import sweeps
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from benchmark_problems import (  # noqa: E402
    BaseBenchmarkProblem,
    get_problems,
    list_available_problems,
    PROBLEM_REGISTRY,
)
from benchmark_utils import convert_to_json_serializable  # noqa: E402
from sweeps.ax_search import ax_search_next_runs  # noqa: E402
from sweeps.bayes_search import bayes_search_next_runs  # noqa: E402
from sweeps.run import RunState, SweepRun  # noqa: E402

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_optimization_replication(
    problem: BaseBenchmarkProblem,
    method: str,
    num_trials: int,
    random_seed: int = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single optimization replication.

    Args:
        problem: Test problem to optimize
        method: Optimization method ('bayes' or 'ax')
        num_trials: Number of optimization trials
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        Dict containing:
            - 'trials': List of trial numbers
            - 'best_values': List of best values found so far at each trial
            - 'values': List of all function values
            - 'final_best': Best value found overall
            - 'final_gap': Gap between final best and known optimum
            - 'method': Method used
            - 'problem': Problem name
            - 'seed': Random seed used
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Create configuration
    config = problem.create_sweep_config(method)

    # Track results
    runs = []
    best_values = []
    all_values = []

    for trial in range(num_trials):
        if verbose and (trial + 1) % 10 == 0:
            print(f"    Trial {trial + 1}/{num_trials}")

        try:
            # Generate next suggestion
            if method == "bayes":
                suggestions = bayes_search_next_runs(runs, config, n=1)
            elif method == "ax":
                suggestions = ax_search_next_runs(
                    runs, config, n=1, random_seed=random_seed
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # Evaluate suggestion
            for suggestion in suggestions:
                # Extract parameters
                params = {k: v["value"] for k, v in suggestion.config.items()}

                # Evaluate function
                value = problem.evaluate(params)
                all_values.append(value)

                # Create run with result
                run = SweepRun(
                    state=RunState.finished,
                    config=suggestion.config,
                    summary_metrics={"value": value},
                )
                runs.append(run)

                # Track best so far using problem's get_best method
                current_best = problem.get_best(all_values)
                best_values.append(current_best)

        except Exception as e:
            if verbose:
                print(f"    Warning: Trial {trial} failed with error: {e}")
            # Use previous best if trial failed
            if best_values:
                best_values.append(best_values[-1])
            else:
                if problem.is_maximization:
                    best_values.append(float("-inf"))
                else:
                    best_values.append(float("inf"))
            continue

    # Calculate metrics
    if all_values:
        final_best = problem.get_best(all_values)
    else:
        final_best = float("-inf") if problem.is_maximization else float("inf")

    final_gap = problem.compute_gap(final_best)

    return {
        "trials": list(range(len(best_values))),
        "best_values": best_values,
        "values": all_values,
        "final_best": final_best,
        "final_gap": final_gap,
        "method": method,
        "problem": problem.name,
        "seed": random_seed,
    }


def benchmark_methods_on_problem(
    problem: BaseBenchmarkProblem,
    methods: List[str],
    num_trials: int,
    num_replications: int = 10,
    random_seeds: List[int] = None,
) -> Dict[str, List[Dict]]:
    """Benchmark multiple methods on a single problem.

    Args:
        problem: Test problem to benchmark
        methods: List of methods to compare (e.g., ['bayes', 'ax'])
        num_trials: Number of optimization trials per replication
        num_replications: Number of replications per method (for statistical robustness)
        random_seeds: List of random seeds (defaults to range(num_replications))

    Returns:
        Dict mapping method name to list of replication results:
        {
            'bayes': [replication1_results, replication2_results, ...],
            'ax': [replication1_results, replication2_results, ...]
        }
    """
    if random_seeds is None:
        random_seeds = list(range(num_replications))

    results = {method: [] for method in methods}

    for method in methods:
        print(f"\n  Running {method} on {problem.name}...")
        for replication_idx, seed in enumerate(random_seeds):
            replication_result = run_optimization_replication(
                problem=problem,
                method=method,
                num_trials=num_trials,
                random_seed=seed,
                verbose=False,
            )

            results[method].append(replication_result)
            print(
                f"    Replication {replication_idx + 1}/{num_replications} (seed={seed}) "
                f"✓ Best: {replication_result['final_best']:.6f}"
            )

    return results


def compute_statistics(results: Dict[str, List[Dict]]) -> Dict[str, Dict[str, Any]]:
    """Compute aggregate statistics across multiple replications.

    Args:
        results: Dict mapping method to list of replication results

    Returns:
        Dict mapping method to statistics:
        {
            'method_name': {
                'mean_best_curve': array of mean best values over iterations,
                'sem_best_curve': array of standard error over iterations,
                'median_best_curve': array of median over iterations,
                'final_best_mean': mean of final best values,
                'final_best_sem': standard error of final best values,
                'final_best_median': median of final best values,
                'num_replications': number of replications
            }
        }
    """
    stats = {}

    for method, replications in results.items():
        # Extract data from replications
        best_curves = [replication["best_values"] for replication in replications]
        final_bests = [replication["final_best"] for replication in replications]
        num_replications = len(replications)

        # Compute statistics
        stats[method] = {
            "mean_best_curve": np.mean(best_curves, axis=0),
            "sem_best_curve": np.std(best_curves, axis=0) / np.sqrt(num_replications),
            "median_best_curve": np.median(best_curves, axis=0),
            "final_best_mean": np.mean(final_bests),
            "final_best_sem": np.std(final_bests) / np.sqrt(num_replications),
            "final_best_median": np.median(final_bests),
            "num_replications": num_replications,
        }

    return stats


def create_summary_table(
    all_results: Dict[str, Dict[str, List[Dict]]],
    all_stats: Dict[str, Dict[str, Dict[str, Any]]],
) -> pd.DataFrame:
    """Create summary table comparing methods across all problems.

    Args:
        all_results: Dict mapping problem name to results dict
        all_stats: Dict mapping problem name to stats dict

    Returns:
        DataFrame with columns:
        - Problem: Problem name
        - Method: Optimization method
        - Final Best (mean): Mean of final best values
        - Final Best (SEM): Standard error of final best values
        - Gap to Opt (mean): Mean gap to known optimum
    """
    rows = []

    for problem_name, results in all_results.items():
        stats = all_stats[problem_name]

        for method in results.keys():
            method_stats = stats[method]

            # Calculate mean gap to optimum from individual replications
            if results[method]:
                gap_mean = np.mean(
                    [replication["final_gap"] for replication in results[method]]
                )
            else:
                gap_mean = float("nan")

            row = {
                "Problem": problem_name,
                "Method": method,
                "Final Best (mean)": f"{method_stats['final_best_mean']:.6f}",
                "Final Best (SEM)": f"{method_stats['final_best_sem']:.6f}",
                "Gap to Opt (mean)": f"{gap_mean:.6f}",
            }
            rows.append(row)

    return pd.DataFrame(rows)


def save_problem_results(
    problem: BaseBenchmarkProblem,
    results: Dict[str, List[Dict]],
    stats: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> str:
    """Save results for a single problem to its own JSON file.

    Args:
        problem: The benchmark problem
        results: Dict mapping method to list of replication results
        stats: Dict mapping method to statistics
        output_dir: Directory to save results

    Returns:
        Path to the saved file
    """
    results_path = os.path.join(output_dir, f"results_{problem.name}.json")

    detailed_data = {
        "metadata": convert_to_json_serializable(problem.get_metadata()),
        "results": convert_to_json_serializable(results),
        "stats": convert_to_json_serializable(stats),
    }

    with open(results_path, "w") as f:
        json.dump(detailed_data, f, indent=2)

    print(f"  Results saved to {results_path}")
    return results_path


def main():
    """Run complete benchmark comparing bayes and ax methods."""
    # Build choices for --problems argument
    problem_choices = list(PROBLEM_REGISTRY.keys()) + ["all", "continuous", "mixed"]

    parser = argparse.ArgumentParser(
        description="Benchmark bayes vs ax on synthetic test functions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=["branin"],
        choices=problem_choices,
        help=(
            "Problems to benchmark (default: branin). Use 'all' for all problems, "
            "'continuous' for Branin/Hartmann6, 'mixed' for AckleyMixed/Labs"
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials per replication",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=100,
        help="Number of replications per method (for statistical robustness)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results and plots",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["bayes", "ax"],
        help="Methods to benchmark",
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
        print("\nAvailable problems:")
        print("-" * 70)
        for name, desc in list_available_problems().items():
            print(f"  {name}: {desc}")
        print("\nSpecial keywords:")
        print("  all: All available problems")
        print("  continuous: Branin, Hartmann6")
        print("  mixed: AckleyMixed, Labs")
        return

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)

    # Get problems from command line
    try:
        problems = get_problems(args.problems)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not problems:
        print("Error: No problems selected", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("BAYESIAN OPTIMIZATION BENCHMARK")
    print("comparing 'bayes' (sklearn) vs 'ax' (ax-platform)")
    print("=" * 70)
    print(f"Problems: {', '.join([p.name for p in problems])}")
    print(f"Trials per replication: {args.trials}")
    print(f"Replications per method: {args.replications}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)

    # Run benchmarks and save per-problem
    all_results = {}
    all_stats = {}

    for problem in problems:
        print(f"\n{'=' * 70}")
        print(f"Benchmarking on {problem.name}")
        print(f"  Dimensionality: {problem.dim}D")
        print(f"  Goal: {'maximize' if problem.is_maximization else 'minimize'}")
        if problem.optimal_value is not None:
            print(f"  Global optimum: {problem.optimal_value:.6f}")
        else:
            print("  Global optimum: Unknown")

        # Print additional info for mixed/discrete problems
        metadata = problem.get_metadata()
        if metadata.get("num_discrete_params", 0) > 0:
            print(f"  Discrete parameters: {metadata['num_discrete_params']}")
            print(f"  Continuous parameters: {metadata['num_continuous_params']}")

        print(f"{'=' * 70}")

        results = benchmark_methods_on_problem(
            problem=problem,
            methods=args.methods,
            num_trials=args.trials,
            num_replications=args.replications,
            random_seeds=[args.seed + i for i in range(args.replications)],
        )

        stats = compute_statistics(results)

        # Save results for this problem
        save_problem_results(problem, results, stats, args.output_dir)

        all_results[problem.name] = results
        all_stats[problem.name] = stats

    # Create and display summary table
    summary_df = create_summary_table(all_results, all_stats)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    # Print analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    for problem_name in all_results.keys():
        print(f"\n{problem_name}:")
        stats = all_stats[problem_name]

        for method in args.methods:
            method_stats = stats[method]
            print(f"\n  {method}:")
            print(
                f"    Final Best: {method_stats['final_best_mean']:.6f} ± {method_stats['final_best_sem']:.6f}"
            )

    print("\n" + "=" * 70)
    print(f"Benchmark complete! Results saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
