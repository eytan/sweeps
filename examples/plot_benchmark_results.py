#!/usr/bin/env python3
"""Plot benchmark results from saved data.

This script loads previously saved benchmark results and generates
convergence plots. Use this to regenerate plots with different styling
or parameters without re-running the expensive benchmark.

Usage:
    # Plot all results from default directory
    python plot_benchmark_results.py

    # Plot results from custom directory
    python plot_benchmark_results.py --results-dir my_results/

    # Save plots to different directory
    python plot_benchmark_results.py --output-dir my_plots/

    # Plot specific problems only
    python plot_benchmark_results.py --problems Branin_2D Hartmann6_6D

Input Files (from results directory):
    - results_{problem_name}.json (one file per problem)

Output:
    - Convergence plots for each test function (PNG files)
"""

import argparse
import glob
import json
import os
import re
import sys

# Suppress warnings for cleaner output
import warnings
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import LogLocator, ScalarFormatter

warnings.filterwarnings("ignore")


class ProblemMetadata:
    """Problem metadata loaded from results file.

    Attributes:
        name: Problem name
        dim: Dimensionality of the search space
        is_maximization: Whether the problem is a maximization problem
        optimal_value: Known global optimum value (None if unknown)
        problem_type: Type of problem ('continuous', 'mixed', 'discrete')
        num_discrete_params: Number of discrete parameters
        num_continuous_params: Number of continuous parameters
        num_trials: Number of trials in the benchmark
    """

    def __init__(
        self,
        name: str,
        dim: int = 0,
        is_maximization: bool = False,
        optimal_value: Optional[float] = None,
        problem_type: str = "continuous",
        num_discrete_params: int = 0,
        num_continuous_params: int = 0,
        num_trials: int = 0,
    ):
        self.name = name
        self.dim = dim
        self.is_maximization = is_maximization
        self.optimal_value = optimal_value
        self.problem_type = problem_type
        self.num_discrete_params = num_discrete_params
        self.num_continuous_params = num_continuous_params
        self.num_trials = num_trials

    @classmethod
    def from_dict(cls, data: Dict[str, Any], num_trials: int = 0) -> "ProblemMetadata":
        """Create ProblemMetadata from a dictionary.

        Args:
            data: Dictionary containing metadata fields
            num_trials: Number of trials in the benchmark

        Returns:
            ProblemMetadata instance
        """
        return cls(
            name=data.get("problem_name", "Unknown"),
            dim=data.get("dim", 0),
            is_maximization=data.get("is_maximization", False),
            optimal_value=data.get("optimal_value"),
            problem_type=data.get("problem_type", "continuous"),
            num_discrete_params=data.get("num_discrete_params", 0),
            num_continuous_params=data.get("num_continuous_params", 0),
            num_trials=num_trials,
        )


def discover_result_files(results_dir: str) -> List[str]:
    """Find all results_*.json files in the directory.

    Args:
        results_dir: Directory containing results JSON files

    Returns:
        List of file paths to results files
    """
    pattern = os.path.join(results_dir, "results_*.json")
    files = glob.glob(pattern)

    return sorted(files)


def extract_problem_name(filepath: str) -> str:
    """Extract problem name from results file path.

    Args:
        filepath: Path to results file (e.g., 'results_Branin_2D.json')

    Returns:
        Problem name (e.g., 'Branin_2D')
    """
    basename = os.path.basename(filepath)
    match = re.match(r"results_(.+)\.json", basename)
    if match:
        return match.group(1)
    return basename.replace("results_", "").replace(".json", "")


def load_single_result(filepath: str) -> Dict[str, Any]:
    """Load a single results file.

    Args:
        filepath: Path to results JSON file

    Returns:
        Dict containing metadata, results, and stats

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    # Convert numpy arrays back for stats curves
    if "stats" in data:
        for method_stats in data["stats"].values():
            for key in ["mean_best_curve", "sem_best_curve", "median_best_curve"]:
                if key in method_stats:
                    method_stats[key] = np.array(method_stats[key])

    return data


def load_results(
    results_dir: str, problem_filter: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Load benchmark results from JSON file(s).

    Automatically discovers and loads all results_*.json files in the directory.
    Each file contains results for a single problem with its metadata.

    Args:
        results_dir: Directory containing results JSON files
        problem_filter: Optional list of problem names to load (None = all)

    Returns:
        Dict mapping problem name to result data:
        {
            'Branin_2D': {
                'metadata': {...},
                'results': {...},
                'stats': {...}
            },
            ...
        }

    Raises:
        FileNotFoundError: If no results files are found
    """
    result_files = discover_result_files(results_dir)

    if not result_files:
        raise FileNotFoundError(
            f"Could not find results files in {results_dir}\n"
            f"Expected: results_{{problem_name}}.json\n"
            f"Make sure you run the benchmark first to generate results."
        )

    all_data = {}

    for filepath in result_files:
        problem_name = extract_problem_name(filepath)

        # Apply filter if specified
        if problem_filter and problem_name not in problem_filter:
            continue

        print(f"Loading results from {filepath}...")
        try:
            data = load_single_result(filepath)
            all_data[problem_name] = data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Failed to load {filepath}: {e}")
            continue

    if not all_data:
        if problem_filter:
            raise FileNotFoundError(
                f"No results found for specified problems: {problem_filter}\n"
                f"Available files: {[extract_problem_name(f) for f in result_files]}"
            )
        else:
            raise FileNotFoundError(f"No valid results files found in {results_dir}")

    return all_data


def extract_problem_metadata(data: Dict[str, Any]) -> ProblemMetadata:
    """Extract problem metadata from loaded result data.

    Args:
        data: Loaded result data for a single problem

    Returns:
        ProblemMetadata instance

    Raises:
        ValueError: If metadata field is missing from result data
    """
    if "metadata" not in data:
        raise ValueError(
            "Results file is missing 'metadata' field. "
            "Please regenerate results using the latest benchmark script."
        )

    stats = data.get("stats", {})
    if stats:
        first_method_stats = stats[list(stats.keys())[0]]
        num_trials = len(first_method_stats.get("mean_best_curve", []))
    else:
        num_trials = 0

    return ProblemMetadata.from_dict(data["metadata"], num_trials)


def plot_convergence_comparison(
    stats: Dict[str, Dict[str, Any]],
    problem_metadata: ProblemMetadata,
    save_path: str = None,
):
    """Create convergence plot comparing methods.

    Generates a plot showing:
    - Mean best value over trials (solid line)
    - ±1 standard error shaded region
    - Horizontal line at known global optimum
    - Legend with final performance

    Args:
        stats: Dict mapping method to statistics
        problem_metadata: Problem metadata for plotting
        save_path: Path to save plot (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {"bayes": "#1f77b4", "ax": "#ff7f0e"}
    labels_with_stats = {}

    for method, method_stats in stats.items():
        trials = range(len(method_stats["mean_best_curve"]))
        mean_curve = method_stats["mean_best_curve"]
        sem_curve = method_stats["sem_best_curve"]

        # Convert accuracy to error for MNIST
        if "MNIST" in problem_metadata.name:
            mean_curve = 100 - mean_curve

        # Create label with final performance
        final_mean = method_stats["final_best_mean"]
        final_sem = method_stats["final_best_sem"]

        # Format label based on problem type
        if "MNIST" in problem_metadata.name:
            # MNIST: convert accuracy to error and show as percentage
            final_error = 100 - final_mean
            label = f"{method}: {final_error:.2f}% ± {final_sem:.2f}%"
        else:
            # Synthetic: show as float
            label = f"{method}: {final_mean:.4f} ± {final_sem:.4f}"
        labels_with_stats[method] = label

        # Plot mean curve
        ax.plot(
            trials,
            mean_curve,
            label=label,
            color=colors.get(method, None),
            linewidth=2.5,
        )

        # Plot standard error region
        ax.fill_between(
            trials,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            alpha=0.25,
            color=colors.get(method, None),
        )

    # Styling
    ax.set_xlabel("Trial", fontsize=12)

    # Set y-axis label based on problem type
    if "MNIST" in problem_metadata.name:
        ax.set_ylabel("Validation Error (%)", fontsize=12)
    elif problem_metadata.is_maximization:
        ax.set_ylabel("Best Value Found (higher is better)", fontsize=12)
    else:
        ax.set_ylabel("Best Value Found (lower is better)", fontsize=12)

    ax.set_title(
        f"Convergence Comparison on {problem_metadata.name}\n"
        f'({stats[list(stats.keys())[0]]["num_replications"]} replications per method)',
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    if "Branin" in problem_metadata.name or "MNIST" in problem_metadata.name:
        # Set y-axis log scale for Branin and MNIST
        ax.semilogy()
        ax.minorticks_on()
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style="plain", axis="y")
        ax.yaxis.set_minor_locator(LogLocator(subs="all"))
        ax.yaxis.set_minor_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.2g}" if x >= 0.01 else "")
        )
        ax.grid(True, which="minor", alpha=0.3, linestyle=":")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved plot to {save_path}")

    plt.close()


def main():
    """Load saved benchmark results and generate plots."""
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from saved data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmark_results",
        help="Directory containing benchmark results JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (defaults to same as results-dir)",
    )
    parser.add_argument(
        "--problems",
        type=str,
        nargs="*",
        default=None,
        help="Specific problems to plot (default: all found in results directory)",
    )
    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PLOTTING BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {output_dir}")
    if args.problems:
        print(f"Filtering for problems: {args.problems}")
    print("=" * 70)

    try:
        # Discover and load all result files
        all_data = load_results(args.results_dir, args.problems)

        print(f"\nFound results for {len(all_data)} problems:")
        for problem_name, data in all_data.items():
            num_methods = len(data.get("results", {}))
            print(f"  - {problem_name} ({num_methods} methods)")

        # Generate plots for each problem
        print("\nGenerating convergence plots...")
        for problem_name, data in all_data.items():
            # Extract metadata
            metadata = extract_problem_metadata(data)

            results = data.get("results", {})
            stats = data.get("stats", {})

            if not results or not stats:
                print(f"  Skipping {problem_name}: missing results or stats")
                continue

            # Generate mean + SEM plot
            plot_save_path = os.path.join(output_dir, f"{problem_name}_convergence.png")
            plot_convergence_comparison(
                stats=stats, problem_metadata=metadata, save_path=plot_save_path
            )

        print("\n" + "=" * 70)
        print(f"Plotting complete! Plots saved to {output_dir}/")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\nError: Failed to parse JSON file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
