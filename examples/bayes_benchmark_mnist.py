#!/usr/bin/env python3
"""Benchmark comparing bayes and ax on MNIST MLP hyperparameter optimization.

This script evaluates the performance of sklearn-based Bayesian optimization
(method='bayes') versus ax-platform Bayesian optimization (method='ax')
on the real-world task of optimizing hyperparameters for an MLP trained on MNIST.

Unlike the synthetic benchmark, this evaluates optimization methods on an actual
machine learning problem with noisy evaluations and longer runtime per trial.

The benchmark runs multiple replications with different random seeds to ensure
statistical robustness and compares:
- Final solution quality (best validation accuracy achieved)
- Convergence speed (how quickly good hyperparameters are found)
- Reliability (consistency across different random seeds)

Hyperparameters Optimized:
- Learning rate (log-scale, 0.0001 to 0.1)
- Weight decay (0.0 to 0.01)
- Dropout (0.0 to 0.5)
- Momentum (0.0 to 0.9)
- Hidden units per layer (32 to 512)
- Number of hidden layers (1 to 5)
- Batch size (16 to 256)

Usage:
    # Quick test (5 trials, 3 replications)
    python bayes_benchmark_mnist.py --trials 5 --replications 3

    # Standard benchmark (30 trials, 20 replications)
    python bayes_benchmark_mnist.py --trials 30 --replications 20

    # Specify custom output directory
    python bayes_benchmark_mnist.py --output-dir my_mnist_results

Output:
    - Results in JSON format (results_mnist.json)

Note:
    To generate convergence plots from the saved results, use:
    python plot_benchmark_results.py --results-dir mnist_benchmark_results

Example Output:
    mnist_benchmark_results/
    └── results_mnist.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sweeps.ax_search import ax_search_next_runs  # noqa: E402
from sweeps.bayes_search import bayes_search_next_runs  # noqa: E402
from sweeps.run import RunState, SweepRun  # noqa: E402

warnings.filterwarnings("ignore")


def convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy types and arrays to native Python types for JSON serialization.

    Args:
        obj: Object to convert (can be dict, list, numpy array, numpy scalar, etc.)

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        return str(obj)


class MLPTrainingProblem:
    """Wrapper for MNIST MLP training to work with sweeps optimization.

    This class handles running the training script with different hyperparameters
    and extracting the validation accuracy metric.

    Attributes:
        name: Human-readable name of the problem
        data_dir: Directory for MNIST data
        train_script: Path to the training script
        epochs: Number of training epochs (fixed)
    """

    def __init__(
        self,
        name: str = "MNIST_MLP",
        data_dir: str = "./data",
        train_script: str = None,
        config_file: str = None,
    ):
        """Initialize MLP training problem.

        Args:
            name: Descriptive name for the problem
            data_dir: Directory to store MNIST data
            train_script: Path to train_mnist_mlp.py script
            config_file: Path to YAML config file with parameter space
        """
        self.name = name
        self.data_dir = data_dir
        self.config_file = config_file
        self.is_maximization = True  # We maximize validation accuracy

        if train_script is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.train_script = os.path.join(script_dir, "train_mnist_mlp.py")
        else:
            self.train_script = train_script

        if not os.path.exists(self.train_script):
            raise FileNotFoundError(f"Training script not found: {self.train_script}")

        # Load config if provided, otherwise use defaults
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                self.yaml_config = yaml.safe_load(f)
        else:
            self.yaml_config = None

        # Extract epochs from config or use default
        if self.yaml_config and "parameters" in self.yaml_config:
            params = self.yaml_config["parameters"]
            if "epochs" in params and "value" in params["epochs"]:
                self.epochs = params["epochs"]["value"]
            else:
                self.epochs = 1
        else:
            self.epochs = 1

        # Count parameters for metadata
        self._count_parameters()

    def _count_parameters(self):
        """Count continuous and discrete parameters from config."""
        self.num_continuous_params = 0
        self.num_discrete_params = 0
        self.dim = 0

        config = self.create_sweep_config("bayes")
        params = config.get("parameters", {})

        for param_name, param_config in params.items():
            self.dim += 1
            if "values" in param_config:
                self.num_discrete_params += 1
            else:
                self.num_continuous_params += 1

    def evaluate(self, params_dict: Dict[str, float], seed: int = None) -> float:
        """Evaluate MLP training with given hyperparameters.

        Runs the training script as a subprocess and extracts validation accuracy.

        Args:
            params_dict: Dict like {'learning_rate': 0.01, 'hidden_units': 128, ...}
            seed: Random seed for reproducibility

        Returns:
            Validation accuracy (0-100 scale)
        """
        cmd = [
            sys.executable,
            self.train_script,
            f"--learning_rate={params_dict['learning_rate']}",
            f"--weight_decay={params_dict['weight_decay']}",
            f"--dropout={params_dict['dropout']}",
            f"--momentum={params_dict['momentum']}",
            f"--hidden_units={int(params_dict['hidden_units'])}",
            f"--num_layers={int(params_dict['num_layers'])}",
            f"--batch_size={int(params_dict['batch_size'])}",
            f"--epochs={self.epochs}",
            f"--data_dir={self.data_dir}",
        ]

        if seed is not None:
            cmd.append(f"--seed={seed}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                msg = "Training failed with return code {}".format(result.returncode)
                print("    Warning: {}".format(msg))
                print("    stderr: {}".format(result.stderr[:200]))
                return 0.0

            match = re.search(r"RESULT: val_accuracy=([\d.]+)", result.stdout)
            if match:
                val_accuracy = float(match.group(1))
                return val_accuracy
            else:
                print("    Warning: Could not parse validation accuracy from output")
                print("    stdout: {}".format(result.stdout[-200:]))
                return 0.0

        except subprocess.TimeoutExpired:
            print("    Warning: Training timed out after 600 seconds")
            return 0.0
        except Exception as e:
            print("    Warning: Training failed with exception: {}".format(e))
            return 0.0

    def create_sweep_config(self, method: str) -> Dict[str, Any]:
        """Create sweep configuration for this problem.

        Loads parameter space from YAML config file if available,
        otherwise uses default hardcoded values.

        Args:
            method: Optimization method ('bayes' or 'ax')

        Returns:
            Sweep configuration dict
        """
        if self.yaml_config and "parameters" in self.yaml_config:
            # Load from YAML config
            config = {
                "method": method,
                "parameters": {},
                "metric": self.yaml_config.get(
                    "metric", {"name": "val_accuracy", "goal": "maximize"}
                ),
            }

            # Extract optimizable parameters (exclude fixed parameters like epochs)
            for param_name, param_config in self.yaml_config["parameters"].items():
                # Skip fixed parameters
                if "value" in param_config:
                    continue

                # Copy parameter config
                config["parameters"][param_name] = param_config.copy()

            return config
        else:
            # Fallback to hardcoded defaults
            config = {
                "method": method,
                "parameters": {
                    "learning_rate": {
                        "min": 0.0001,
                        "max": 0.1,
                        "distribution": "log_uniform_values",
                    },
                    "weight_decay": {"min": 0.0, "max": 0.01},
                    "dropout": {"min": 0.0, "max": 0.5},
                    "momentum": {"min": 0.0, "max": 0.9},
                    "hidden_units": {"min": 32, "max": 512},
                    "num_layers": {"min": 1, "max": 5},
                    "batch_size": {"min": 16, "max": 256},
                },
                "metric": {"name": "val_accuracy", "goal": "maximize"},
            }
            return config

    def get_metadata(self) -> Dict[str, Any]:
        """Get problem metadata for result files.

        Returns:
            Dict containing problem metadata for serialization
        """
        # Determine problem type
        if self.num_discrete_params == 0:
            problem_type = "continuous"
        elif self.num_continuous_params == 0:
            problem_type = "discrete"
        else:
            problem_type = "mixed"

        return {
            "problem_name": self.name,
            "dim": self.dim,
            "is_maximization": self.is_maximization,
            "optimal_value": None,  # Unknown for real-world problems
            "problem_type": problem_type,
            "num_discrete_params": self.num_discrete_params,
            "num_continuous_params": self.num_continuous_params,
        }


def run_optimization_replication(
    problem: MLPTrainingProblem,
    method: str,
    num_trials: int,
    random_seed: int = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single optimization replication.

    Args:
        problem: MLP training problem to optimize
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
            - 'method': Method used
            - 'problem': Problem name
            - 'seed': Random seed used
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    config = problem.create_sweep_config(method)

    runs = []
    best_values = []
    all_values = []

    for trial in range(num_trials):
        if verbose and (trial + 1) % 10 == 0:
            print(f"    Trial {trial + 1}/{num_trials}")

        try:
            if method == "bayes":
                suggestions = bayes_search_next_runs(runs, config, n=1)
            elif method == "ax":
                suggestions = ax_search_next_runs(
                    runs, config, n=1, random_seed=random_seed
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            for suggestion in suggestions:
                params = {k: v["value"] for k, v in suggestion.config.items()}

                value = problem.evaluate(params, seed=random_seed)
                all_values.append(value)

                run = SweepRun(
                    state=RunState.finished,
                    config=suggestion.config,
                    summary_metrics={"val_accuracy": value},
                )
                runs.append(run)

                current_best = max(all_values)
                best_values.append(current_best)

                if verbose:
                    print(
                        f"      Accuracy: {value:.2f}%, Best so far: {current_best:.2f}%"
                    )

        except Exception as e:
            if verbose:
                print(f"    Warning: Trial {trial} failed with error: {e}")
            if best_values:
                best_values.append(best_values[-1])
            else:
                best_values.append(0.0)
            continue

    final_best = max(all_values) if all_values else 0.0

    return {
        "trials": list(range(len(best_values))),
        "best_values": best_values,
        "values": all_values,
        "final_best": final_best,
        "method": method,
        "problem": problem.name,
        "seed": random_seed,
    }


def benchmark_methods_on_problem(
    problem: MLPTrainingProblem,
    methods: List[str],
    num_trials: int,
    num_replications: int = 10,
    random_seeds: List[int] = None,
) -> Dict[str, List[Dict]]:
    """Benchmark multiple methods on MNIST MLP problem.

    Args:
        problem: MLP training problem to benchmark
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
                f"✓ Best: {replication_result['final_best']:.2f}%"
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
                'mean_best_curve': array of mean best values over trials,
                'sem_best_curve': array of standard error over trials,
                'median_best_curve': array of median over trials,
                'final_best_mean': mean of final best values,
                'final_best_sem': standard error of final best values,
                'final_best_median': median of final best values,
                'num_replications': number of replications
            }
        }
    """
    stats = {}

    for method, replications in results.items():
        best_curves = [replication["best_values"] for replication in replications]
        final_bests = [replication["final_best"] for replication in replications]
        num_replications = len(replications)

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
        - Final Best (median): Median of final best values
    """
    rows = []

    for problem_name, results in all_results.items():
        stats = all_stats[problem_name]

        for method in results.keys():
            method_stats = stats[method]

            row = {
                "Problem": problem_name,
                "Method": method,
                "Final Best (mean)": f"{method_stats['final_best_mean']:.2f}%",
                "Final Best (SEM)": f"{method_stats['final_best_sem']:.2f}%",
                "Final Best (median)": f"{method_stats['final_best_median']:.2f}%",
            }
            rows.append(row)

    return pd.DataFrame(rows)


def save_replications_csv(
    results: Dict[str, List[Dict]], problem_name: str, save_path: str
):
    """Save individual replication data to CSV for easy analysis.

    Creates a CSV file with one row per trial per replication.

    Args:
        results: Dict mapping method to list of replication results
        problem_name: Name of the problem (for reference)
        save_path: Path to save the CSV file
    """
    rows = []

    for method, replications in results.items():
        for replication_idx, replication in enumerate(replications):
            for trial_idx in range(len(replication["trials"])):
                row = {
                    "replication": replication_idx,
                    "method": method,
                    "seed": replication["seed"],
                    "trial": replication["trials"][trial_idx],
                    "value": (
                        replication["values"][trial_idx]
                        if trial_idx < len(replication["values"])
                        else None
                    ),
                    "best_value": replication["best_values"][trial_idx],
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"    Replication data saved to {save_path}")


def save_statistics_csv(
    all_stats: Dict[str, Dict[str, Dict[str, Any]]], save_path: str
):
    """Save detailed statistics to CSV.

    Creates a CSV with one row per problem/method/metric combination.

    Args:
        all_stats: Dict mapping problem name to stats dict
        save_path: Path to save the CSV file
    """
    rows = []

    for problem_name, problem_stats in all_stats.items():
        for method, method_stats in problem_stats.items():
            for metric, value in method_stats.items():
                if metric not in [
                    "mean_best_curve",
                    "sem_best_curve",
                    "median_best_curve",
                ]:
                    rows.append(
                        {
                            "problem": problem_name,
                            "method": method,
                            "metric": metric,
                            "value": value if value is not None else "N/A",
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Statistics saved to {save_path}")


def save_problem_results(
    problem: MLPTrainingProblem,
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
    """Run complete benchmark comparing bayes and ax methods on MNIST MLP."""
    parser = argparse.ArgumentParser(
        description="Benchmark bayes vs ax on MNIST MLP hyperparameter optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default=30,
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
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for MNIST data",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sweep-mnist-mlp.yaml",
        help="Path to YAML config file with parameter space",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for reproducibility",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("MNIST MLP HYPERPARAMETER OPTIMIZATION BENCHMARK")
    print("comparing 'bayes' (sklearn) vs 'ax' (ax-platform)")
    print("=" * 70)
    print(f"Trials per replication: {args.trials}")
    print(f"Replications per method: {args.replications}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Config file: {args.config}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)

    # Determine config file path
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        # Look in examples directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, args.config)

    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        print("Using hardcoded default parameters")
        config_path = None

    problem = MLPTrainingProblem(
        name="MNIST_MLP",
        data_dir=args.data_dir,
        config_file=config_path,
    )

    print(f"\nProblem: {problem.name}")
    print(f"  Training script: {problem.train_script}")
    print(f"  Data directory: {problem.data_dir}")
    print(f"  Epochs per run: {problem.epochs}")

    print(f"\n{'=' * 70}")
    print(f"Starting benchmark on {problem.name}")
    print(f"{'=' * 70}")

    results = benchmark_methods_on_problem(
        problem=problem,
        methods=args.methods,
        num_trials=args.trials,
        num_replications=args.replications,
        random_seeds=[args.seed + i for i in range(args.replications)],
    )

    stats = compute_statistics(results)

    all_results = {problem.name: results}
    all_stats = {problem.name: stats}

    # Create and display summary table
    summary_df = create_summary_table(all_results, all_stats)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    # Save results as JSON
    save_problem_results(problem, results, stats, args.output_dir)

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
                f"    Final Best Accuracy: {method_stats['final_best_mean']:.2f}% ± {method_stats['final_best_sem']:.2f}%"
            )
            print(f"    Median Final Best: {method_stats['final_best_median']:.2f}%")

    print("\n" + "=" * 70)
    print(f"Benchmark complete! Results saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
