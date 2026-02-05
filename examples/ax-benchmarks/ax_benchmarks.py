#!/usr/bin/env python3
"""Benchmarking with YAML config validation through W&B's SweepConfig.

This module validates the full YAML serialization pipeline:
1. Load sweep configs from YAML files
2. Validate through W&B's SweepConfig
3. Run optimization using ax_search_next_runs
4. Optionally log through W&B's logger flow

Usage:
    # Run SOO benchmark (branin)
    python ax_benchmarks.py --mode soo --trials 50 --replications 10

    # Run MOO benchmark (dtlz2, c2dtlz2)
    python ax_benchmarks.py --mode moo --trials 50 --replications 10

    # Run integration test (direct ax vs JSON roundtrip)
    python ax_benchmarks.py --mode integration --trials 20 --replications 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import yaml

# Set W&B offline before importing
os.environ.setdefault("WANDB_MODE", "offline")

try:
    import torch
    from botorch.test_functions.multi_objective import C2DTLZ2, DTLZ2, MultiObjectiveTestProblem
    from botorch.test_functions.synthetic import Branin
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    from botorch.utils.multi_objective.pareto import is_non_dominated
except ImportError as e:
    print(f"Error: Required package missing: {e}")
    sys.exit(1)

# Add sweeps to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sweeps.ax_search import ax_search_next_runs  # noqa: E402
from sweeps.bayes_search import bayes_search_next_runs  # noqa: E402
from sweeps.config import SweepConfig  # noqa: E402
from sweeps.run import next_runs, RunState, SweepRun  # noqa: E402

warnings.filterwarnings("ignore")


# =============================================================================
# Problem Registry (BoTorch test functions - everything else from YAML)
# =============================================================================

PROBLEMS = {
    "branin": Branin(),
    "dtlz2": DTLZ2(dim=6, num_objectives=2, negate=False),
    "c2dtlz2": C2DTLZ2(dim=6, num_objectives=2, negate=False),
}


def create_evaluator(func, config: dict[str, Any]):
    """Create an evaluate function from a BoTorch function and YAML config."""
    param_names = sorted(config["parameters"].keys())

    assert len(param_names) == func.dim

    if is_moo_config(config):
        assert isinstance(func, MultiObjectiveTestProblem)
        obj_names = [m["name"] for m in config["metrics"]]
        assert len(obj_names) == func.num_objectives
    else:
        obj_names = [config["metric"]["name"]]

    constraint_names = [c.split()[0] for c in config.get("metric_constraints", [])]

    def evaluate(params: dict[str, float]) -> dict[str, float]:
        X = torch.tensor([[params[p] for p in param_names]], dtype=torch.double)
        result = {}

        # Objectives
        Y = func(X)
        if Y.dim() == 1:
            result[obj_names[0]] = float(Y.item())
        else:
            for i, name in enumerate(obj_names):
                result[name] = float(Y[0, i].item())

        # Constraints
        if constraint_names and hasattr(func, "evaluate_slack"):
            slack = func.evaluate_slack(X)
            for i, name in enumerate(constraint_names):
                result[name] = float(-slack[0, i].item())  # Convert slack to g <= 0

        return result

    return evaluate

# Map from problem name to YAML config file
YAML_CONFIGS = {
    "branin": "ax_branin_soo.yaml",
    "dtlz2": "ax_dtlz2_moo.yaml",
    "c2dtlz2": "ax_c2dtlz2_moo.yaml",
}


def load_yaml_config(problem_name: str) -> dict[str, Any]:
    """Load and validate a YAML config for a problem."""
    if problem_name not in YAML_CONFIGS:
        raise ValueError(f"No YAML config for problem '{problem_name}'")
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", YAML_CONFIGS[problem_name]
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    SweepConfig(config)  # validate through W&B's SweepConfig
    return config


def is_moo_config(config: dict[str, Any]) -> bool:
    """Check if a config is multi-objective (has 'metrics' instead of 'metric')."""
    return "metrics" in config


def get_ref_point(config: dict[str, Any]) -> list[float] | None:
    """Get reference point from metric thresholds in config."""
    if not is_moo_config(config):
        return None
    thresholds = [m.get("threshold") for m in config["metrics"]]
    if all(t is not None for t in thresholds):
        return thresholds
    return None


# =============================================================================
# W&B Offline File Reading Utilities
# =============================================================================


def read_roundtrip_runs(data_dir: str) -> list[SweepRun]:
    """Read all runs from JSON sidecar files for serialization round-trip testing."""
    runs = []
    runs_file = os.path.join(data_dir, "roundtrip_runs.json")

    if not os.path.exists(runs_file):
        return runs

    with open(runs_file) as f:
        runs_data = json.load(f)

    for run_data in runs_data:
        param_config = {k: {"value": v} for k, v in run_data["params"].items()}
        runs.append(
            SweepRun(
                state=RunState.finished,
                config=param_config,
                summary_metrics=run_data["metrics"],
            )
        )

    return runs


def append_roundtrip_run(
    data_dir: str, params: dict[str, float], metrics: dict[str, float]
) -> None:
    """Append a run to the JSON sidecar file for round-trip testing."""
    runs_file = os.path.join(data_dir, "roundtrip_runs.json")

    if os.path.exists(runs_file):
        with open(runs_file) as f:
            runs_data = json.load(f)
    else:
        runs_data = []

    runs_data.append({"params": params, "metrics": metrics})

    with open(runs_file, "w") as f:
        json.dump(runs_data, f)


# =============================================================================
# Search Nodes (wrap W&B searchers for benchmark framework)
# =============================================================================


class BaseSearchNode:
    """Base class for search method wrappers."""

    def __init__(self, sweep_config: dict[str, Any], random_seed: int = 42):
        self.sweep_config = sweep_config
        self.random_seed = random_seed
        self.sweep_runs: list[SweepRun] = []

    @abstractmethod
    def get_next_candidate(self) -> dict[str, float]:
        """Get next parameter suggestion."""
        pass

    def record_observation(
        self, params: dict[str, float], metrics: dict[str, float]
    ) -> None:
        """Record a completed trial."""
        self.sweep_runs.append(
            SweepRun(
                state=RunState.finished,
                config={k: {"value": v} for k, v in params.items()},
                summary_metrics=metrics,
            )
        )


class DirectAxNode(BaseSearchNode):
    """Direct call to ax_search_next_runs (no W&B logging)."""

    def get_next_candidate(self) -> dict[str, float]:
        suggestions = ax_search_next_runs(
            self.sweep_runs, self.sweep_config, n=1, random_seed=self.random_seed
        )
        return {k: v["value"] for k, v in suggestions[0].config.items()}


# Not used in this benchmark, but can be used for comparing bayes vs ax
class DirectBayesNode(BaseSearchNode):
    """Direct call to bayes_search_next_runs (no W&B logging)."""

    def get_next_candidate(self) -> dict[str, float]:
        suggestions = bayes_search_next_runs(self.sweep_runs, self.sweep_config, n=1)
        return {k: v["value"] for k, v in suggestions[0].config.items()}


class WandBRoundTripNode(BaseSearchNode):
    """Serialization round-trip test: verifies data survives JSON serialization."""

    def __init__(
        self,
        sweep_config: dict[str, Any],
        random_seed: int = 42,
        wandb_dir: str | None = None,
    ):
        super().__init__(sweep_config, random_seed)
        self.wandb_dir = wandb_dir or tempfile.mkdtemp()
        self._run_count = 0

    def get_next_candidate(self) -> dict[str, float]:
        # Read back ALL runs from JSON sidecar file (TRUE round-trip)
        reconstructed_runs = read_roundtrip_runs(self.wandb_dir)

        # Use reconstructed runs (NOT in-memory self.sweep_runs)
        config = SweepConfig(self.sweep_config)
        suggestions = next_runs(
            config,
            reconstructed_runs,
            validate=False,
            n=1,
            random_seed=self.random_seed,
        )
        return {k: v["value"] for k, v in suggestions[0].config.items()}

    def record_observation(
        self, params: dict[str, float], metrics: dict[str, float]
    ) -> None:
        """Log to W&B and JSON sidecar - do NOT store in memory."""
        import wandb

        self._run_count += 1
        wandb.init(
            project="ax-e2e-roundtrip",
            name=f"trial-{self._run_count}",
            config=params,
            dir=self.wandb_dir,
            reinit=True,
        )
        wandb.log(metrics)
        for k, v in metrics.items():
            wandb.run.summary[k] = v
        wandb.finish(quiet=True)

        # Write to JSON sidecar for round-trip testing
        append_roundtrip_run(self.wandb_dir, params, metrics)


# =============================================================================
# Benchmark Runner
# =============================================================================


@dataclass
class ReplicationResult:
    """Result of a single benchmark replication."""

    seed: int
    best_values: list[float]  # SOO: best value at each trial
    hypervolume_curve: list[float]  # MOO: hypervolume at each trial
    final_value: float  # SOO: final best, MOO: final hypervolume


def compute_hypervolume(Y: np.ndarray, ref_point: list[float]) -> float:
    """Compute hypervolume for minimization objectives."""
    if len(Y) == 0:
        return 0.0
    neg_ref = torch.tensor([-r for r in ref_point], dtype=torch.double)
    hv = Hypervolume(ref_point=neg_ref)
    return float(hv.compute(torch.tensor(-Y, dtype=torch.double)))


def get_pareto_front(Y: np.ndarray, feasible: np.ndarray | None = None) -> np.ndarray:
    """Get Pareto front from objective values."""
    if len(Y) == 0:
        return np.array([])
    if feasible is not None:
        Y = Y[feasible]
        if len(Y) == 0:
            return np.array([])
    Y_tensor = torch.tensor(Y, dtype=torch.double)
    mask = is_non_dominated(-Y_tensor)  # Negate for minimization
    return Y_tensor[mask].numpy()


def run_replication(
    evaluate: Callable[[dict[str, float]], dict[str, float]],
    config: dict[str, Any],
    ref_point: list[float] | None,
    node: BaseSearchNode,
    num_trials: int,
) -> ReplicationResult:
    """Run a single benchmark replication."""
    is_moo = is_moo_config(config)

    # Get metric/objective names from config
    if is_moo:
        objective_names = [m["name"] for m in config["metrics"]]
    else:
        objective_names = [config["metric"]["name"]]

    # Get constraint names from config
    constraint_names = []
    for constraint in config.get("metric_constraints", []):
        # Parse "c1 <= 0" to get "c1"
        constraint_names.append(constraint.split()[0])

    all_values = []
    all_Y = []
    all_feasible = []
    best_values = []
    hv_curve = []

    for _ in range(num_trials):
        params = node.get_next_candidate()
        result = evaluate(params)
        node.record_observation(params, result)

        if not is_moo:
            # SOO tracking
            val = result[objective_names[0]]
            all_values.append(val)
            best_so_far = min(all_values)  # assume minimization
            best_values.append(best_so_far)
        else:
            # MOO tracking
            obj_vals = [result[name] for name in objective_names]
            all_Y.append(obj_vals)

            if constraint_names:
                is_feasible = all(result[c] <= 0 for c in constraint_names)
            else:
                is_feasible = True
            all_feasible.append(is_feasible)

            Y_array = np.array(all_Y)
            feasible_mask = np.array(all_feasible)
            if np.any(feasible_mask):
                pareto = get_pareto_front(Y_array, feasible_mask)
                hv = compute_hypervolume(pareto, ref_point) if len(pareto) > 0 else 0.0
            else:
                hv = 0.0
            hv_curve.append(hv)

    if not is_moo:
        final = best_values[-1] if best_values else float("inf")
    else:
        final = hv_curve[-1] if hv_curve else 0.0

    return ReplicationResult(
        seed=node.random_seed,
        best_values=best_values,
        hypervolume_curve=hv_curve,
        final_value=final,
    )


def benchmark_method(
    evaluate: Callable[[dict[str, float]], dict[str, float]],
    config: dict[str, Any],
    node_factory: Callable[[dict, int], BaseSearchNode],
    num_trials: int,
    num_replications: int,
    base_seed: int = 0,
) -> list[ReplicationResult]:
    """Run benchmark across multiple replications."""
    results = []
    is_moo = is_moo_config(config)
    ref_point = get_ref_point(config)

    for i in range(num_replications):
        seed = base_seed + i
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"    Rep {i+1}/{num_replications} (seed={seed})... ", end="", flush=True)

        node = node_factory(config, seed)
        result = run_replication(evaluate, config, ref_point, node, num_trials)
        results.append(result)

        metric = "HV" if is_moo else "Best"
        print(f"{metric}={result.final_value:.6f}")

    return results


def compute_stats(results: list[ReplicationResult], is_moo: bool) -> dict[str, float]:
    """Compute aggregate statistics."""
    finals = [r.final_value for r in results]
    return {
        "mean": float(np.mean(finals)),
        "sem": float(np.std(finals) / np.sqrt(len(finals))),
        "median": float(np.median(finals)),
    }


# =============================================================================
# Main Benchmark Modes
# =============================================================================


def run_soo_benchmark(
    problems: list[str], num_trials: int, num_replications: int, seed: int
):
    """SOO benchmark: load config from YAML, validate through SweepConfig, run ax."""
    print("=" * 70)
    print("SOO BENCHMARK: ax (from YAML config)")
    print("=" * 70)

    for problem_name in problems:
        if problem_name not in YAML_CONFIGS:
            continue
        config = load_yaml_config(problem_name)
        if is_moo_config(config):
            continue

        evaluate = create_evaluator(PROBLEMS[problem_name], config)
        print(f"\n{config.get('name', problem_name)}")
        print(f"  ax:")

        results = benchmark_method(
            evaluate,
            config,
            lambda c, s: DirectAxNode(c, s),
            num_trials,
            num_replications,
            seed,
        )
        stats = compute_stats(results, is_moo=False)
        print(f"    -> {stats['mean']:.6f} +/- {stats['sem']:.6f}")


def run_moo_benchmark(
    problems: list[str], num_trials: int, num_replications: int, seed: int
):
    """MOO benchmark: load config from YAML, validate through SweepConfig, run ax."""
    print("=" * 70)
    print("MOO BENCHMARK: ax (from YAML config)")
    print("=" * 70)

    for problem_name in problems:
        if problem_name not in YAML_CONFIGS:
            continue
        config = load_yaml_config(problem_name)
        if not is_moo_config(config):
            continue

        evaluate = create_evaluator(PROBLEMS[problem_name], config)
        num_obj = len(config["metrics"])
        num_constraints = len(config.get("metric_constraints", []))
        print(
            f"\n{config.get('name', problem_name)} ({num_obj} obj, {num_constraints} constraints)"
        )
        print(f"  ax:")

        results = benchmark_method(
            evaluate,
            config,
            lambda c, s: DirectAxNode(c, s),
            num_trials,
            num_replications,
            seed,
        )
        stats = compute_stats(results, is_moo=True)
        print(f"    -> Final HV: {stats['mean']:.6f} +/- {stats['sem']:.6f}")


def run_integration_test(
    problem_name: str, num_trials: int, num_replications: int, seed: int
):
    """Integration test: vanilla ax (direct) vs W&B roundtrip (serialization)."""
    print("=" * 70)
    print("INTEGRATION TEST: Direct ax vs W&B Roundtrip")
    print("=" * 70)

    if problem_name not in YAML_CONFIGS:
        print(f"No YAML config for problem '{problem_name}'")
        return False

    config = load_yaml_config(problem_name)
    evaluate = create_evaluator(PROBLEMS[problem_name], config)
    ref_point = get_ref_point(config)
    print(f"\nProblem: {config.get('name', problem_name)}")

    with tempfile.TemporaryDirectory() as wandb_dir:
        all_match = True

        for i in range(num_replications):
            rep_seed = seed + i
            np.random.seed(rep_seed)
            torch.manual_seed(rep_seed)

            # Direct path
            direct_node = DirectAxNode(config, rep_seed)
            direct_result = run_replication(
                evaluate,
                config,
                ref_point,
                direct_node,
                num_trials,
            )

            # Reset seed for roundtrip path
            np.random.seed(rep_seed)
            torch.manual_seed(rep_seed)

            # W&B roundtrip path (reads back from JSON each iteration)
            roundtrip_node = WandBRoundTripNode(config, rep_seed, wandb_dir)
            roundtrip_result = run_replication(
                evaluate,
                config,
                ref_point,
                roundtrip_node,
                num_trials,
            )

            diff = roundtrip_result.final_value - direct_result.final_value
            match = abs(diff) < 1e-9
            all_match = all_match and match
            status = "OK" if match else "MISMATCH"

            print(
                f"  Rep {i+1}/{num_replications}: Direct={direct_result.final_value:.6f} "
                f"Roundtrip={roundtrip_result.final_value:.6f} Diff={diff:+.2e} [{status}]"
            )

    print("=" * 70)
    if all_match:
        print("PASS: Roundtrip serialization matches direct ax")
    else:
        print("FAIL: Results differ")
    print("=" * 70)

    return all_match


def main():
    parser = argparse.ArgumentParser(
        description="Ax benchmarking with W&B sweeps integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["soo", "moo", "integration", "all"],
        default="all",
        help="Benchmark mode: soo, moo, integration, or all",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=["branin", "dtlz2", "c2dtlz2"],
        choices=list(PROBLEMS.keys()),
        help="Problems to benchmark",
    )
    parser.add_argument("--trials", type=int, default=30, help="Trials per replication")
    parser.add_argument(
        "--replications", type=int, default=5, help="Number of replications"
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    args = parser.parse_args()

    if args.mode in ("soo", "all"):
        run_soo_benchmark(args.problems, args.trials, args.replications, args.seed)

    if args.mode in ("moo", "all"):
        run_moo_benchmark(args.problems, args.trials, args.replications, args.seed)

    if args.mode in ("integration", "all"):
        # Test on first MOO problem
        moo_problems = [
            p for p in args.problems
            if p in YAML_CONFIGS and is_moo_config(load_yaml_config(p))
        ]
        if moo_problems:
            run_integration_test(
                moo_problems[0], args.trials, args.replications, args.seed
            )


if __name__ == "__main__":
    main()
