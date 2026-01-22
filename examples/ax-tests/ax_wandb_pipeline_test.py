#!/usr/bin/env python3
"""Integration test: W&B Ax sweep vs vanilla Ax.

Validates that running Ax through the full W&B sweep pipeline produces
identical results to running Ax directly via the Client API.

The W&B path includes:
  - YAML config loading
  - SweepConfig validation
  - next_runs() for suggestions
  - wandb.init(), wandb.log(), wandb.finish() for each trial

Uses same random seed for both paths - results should match exactly.

Usage:
    # Quick sanity check
    python ax_wandb_pipeline_test.py --trials 10 --replications 3

    # Standard test
    python ax_wandb_pipeline_test.py --trials 20 --replications 5
"""

import argparse
import os
import sys
import tempfile
import warnings
from typing import Any, Dict

import numpy as np
import yaml

# Set W&B offline mode before importing wandb
os.environ["WANDB_MODE"] = "offline"

try:
    import torch
except ImportError:
    print("Error: torch is required.")
    sys.exit(1)

try:
    import wandb
except ImportError:
    print("Error: wandb is required. Install with: pip install wandb")
    sys.exit(1)

# Ax imports
try:
    from ax.api.client import Client
    from ax.api.configs import RangeParameterConfig
    from ax.api.utils.instantiation.from_string import parse_outcome_constraint
    from ax.core.metric import Metric
    from ax.core.objective import MultiObjective, Objective
    from ax.core.optimization_config import MultiObjectiveOptimizationConfig
    from ax.core.outcome_constraint import ObjectiveThreshold
    from ax.core.types import ComparisonOp
except ImportError:
    print("Error: ax-platform is required.")
    sys.exit(1)

# Add path to import sweeps
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from benchmark_problems import (  # noqa: E402
    MultiObjectiveBoTorchProblem,
    MOO_PROBLEM_REGISTRY,
)
from sweeps.config import SweepConfig  # noqa: E402
from sweeps.run import next_runs, RunState, SweepRun  # noqa: E402

warnings.filterwarnings("ignore")

YAML_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
PROBLEM_YAML_MAP = {
    "dtlz2": os.path.join(YAML_CONFIG_DIR, "ax_dtlz2_moo.yaml"),
    "c2dtlz2": os.path.join(YAML_CONFIG_DIR, "ax_c2dtlz2_moo.yaml"),
    "welded_beam": os.path.join(YAML_CONFIG_DIR, "ax_welded_beam_moo.yaml"),
}


def run_vanilla_ax(
    problem: MultiObjectiveBoTorchProblem,
    num_trials: int,
    random_seed: int,
) -> Dict[str, Any]:
    """Run optimization using vanilla Ax Client API."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Create Ax client
    ax_parameters = []
    for i in range(problem.dim):
        lower, upper = problem.bounds[i]
        ax_parameters.append(
            RangeParameterConfig(
                name=f"x{i}",
                bounds=(float(lower), float(upper)),
                parameter_type="float",
            )
        )

    client = Client(random_seed=random_seed)
    client.configure_experiment(parameters=ax_parameters, name=problem.name)

    # Configure MOO
    objectives = []
    objective_thresholds = []
    for i, name in enumerate(problem.objective_names):
        metric = Metric(name=name)
        objectives.append(Objective(metric=metric, minimize=True))
        objective_thresholds.append(
            ObjectiveThreshold(
                metric=metric,
                bound=problem.ref_point[i],
                relative=False,
                op=ComparisonOp.LEQ,
            )
        )

    parsed_constraints = []
    if problem.num_constraints > 0:
        for name in problem.constraint_names:
            parsed_constraints.append(parse_outcome_constraint(f"{name} <= 0"))

    opt_config = MultiObjectiveOptimizationConfig(
        objective=MultiObjective(objectives=objectives),
        objective_thresholds=objective_thresholds,
        outcome_constraints=parsed_constraints,
    )
    client.set_optimization_config(opt_config)
    client.configure_generation_strategy(
        method="fast",
        initialization_random_seed=random_seed,
        initialize_with_center=False,
    )

    # Run trials
    all_Y = []
    all_feasible = []

    for _ in range(num_trials):
        next_trials = client.get_next_trials(max_trials=1)
        if not next_trials:
            break

        trial_index, params = list(next_trials.items())[0]
        result = problem.evaluate(params)

        obj_values = [result[name] for name in problem.objective_names]
        all_Y.append(obj_values)

        if problem.num_constraints > 0:
            constraint_values = [result[name] for name in problem.constraint_names]
            is_feasible = all(g <= 0 for g in constraint_values)
        else:
            is_feasible = True
        all_feasible.append(is_feasible)

        raw_data = {}
        for name in problem.objective_names:
            raw_data[name] = result[name]
        for name in problem.constraint_names:
            raw_data[name] = result[name]
        client.complete_trial(trial_index=trial_index, raw_data=raw_data)

    # Compute final hypervolume
    Y_array = np.array(all_Y)
    feasible_mask = np.array(all_feasible)
    if np.any(feasible_mask):
        pareto_Y = problem.get_pareto_front(Y_array, feasible_mask)
        final_hv = problem.compute_hypervolume(pareto_Y) if len(pareto_Y) > 0 else 0.0
    else:
        final_hv = 0.0

    return {"final_hypervolume": final_hv, "all_Y": all_Y}


def run_wandb_ax(
    problem: MultiObjectiveBoTorchProblem,
    problem_name: str,
    num_trials: int,
    random_seed: int,
    wandb_dir: str,
) -> Dict[str, Any]:
    """Run optimization using full W&B sweep pipeline.

    This simulates a real W&B sweep run:
      YAML -> SweepConfig -> next_runs() -> wandb.init/log/finish
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    yaml_path = PROBLEM_YAML_MAP[problem_name]
    with open(yaml_path) as f:
        raw_config = yaml.safe_load(f)
    config = SweepConfig(raw_config)

    sweep_runs = []
    all_Y = []
    all_feasible = []

    for _ in range(num_trials):
        suggestions = next_runs(
            config, sweep_runs, validate=False, n=1, random_seed=random_seed
        )
        if not suggestions:
            break

        params = {k: v["value"] for k, v in suggestions[0].config.items()}

        # Simulate W&B run lifecycle
        run = wandb.init(
            project="ax-pipeline-test",
            config=params,
            dir=wandb_dir,
            reinit=True,
        )

        result = problem.evaluate(params)

        # Log metrics through W&B
        wandb.log(result)

        obj_values = [result[name] for name in problem.objective_names]
        all_Y.append(obj_values)

        if problem.num_constraints > 0:
            constraint_values = [result[name] for name in problem.constraint_names]
            is_feasible = all(g <= 0 for g in constraint_values)
        else:
            is_feasible = True
        all_feasible.append(is_feasible)

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

    # Compute final hypervolume
    Y_array = np.array(all_Y)
    feasible_mask = np.array(all_feasible)
    if np.any(feasible_mask):
        pareto_Y = problem.get_pareto_front(Y_array, feasible_mask)
        final_hv = problem.compute_hypervolume(pareto_Y) if len(pareto_Y) > 0 else 0.0
    else:
        final_hv = 0.0

    return {"final_hypervolume": final_hv, "all_Y": all_Y}


def main():
    parser = argparse.ArgumentParser(
        description="Integration test: W&B Ax vs vanilla Ax",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="dtlz2",
        choices=list(PROBLEM_YAML_MAP.keys()),
        help="Problem to test",
    )
    parser.add_argument("--trials", type=int, default=20, help="Trials per replication")
    parser.add_argument("--replications", type=int, default=5, help="Number of replications")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    args = parser.parse_args()

    problem = MOO_PROBLEM_REGISTRY[args.problem]()

    print("=" * 70)
    print("INTEGRATION TEST: W&B Ax (full pipeline) vs Vanilla Ax")
    print("=" * 70)
    print(f"Problem: {problem.name} ({problem.dim}D, {problem.num_objectives} obj)")
    print(f"Trials: {args.trials}, Replications: {args.replications}")
    print(f"W&B mode: offline")
    print("=" * 70)

    all_match = True
    wandb_hvs = []
    vanilla_hvs = []

    with tempfile.TemporaryDirectory() as wandb_dir:
        for rep in range(args.replications):
            seed = args.seed + rep

            wandb_result = run_wandb_ax(
                problem, args.problem, args.trials, seed, wandb_dir
            )
            vanilla_result = run_vanilla_ax(problem, args.trials, seed)

            wandb_hv = wandb_result["final_hypervolume"]
            vanilla_hv = vanilla_result["final_hypervolume"]
            diff = wandb_hv - vanilla_hv

            wandb_hvs.append(wandb_hv)
            vanilla_hvs.append(vanilla_hv)

            match = abs(diff) < 1e-10
            all_match = all_match and match
            status = "OK" if match else "MISMATCH"

            print(
                f"  Rep {rep + 1}/{args.replications} (seed={seed}): "
                f"W&B={wandb_hv:.4f} Vanilla={vanilla_hv:.4f} Diff={diff:+.2e} [{status}]"
            )

    print("=" * 70)
    if all_match:
        print("PASS: W&B Ax produces identical results to vanilla Ax")
    else:
        print("FAIL: Results differ between W&B Ax and vanilla Ax")
    print("=" * 70)

    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
