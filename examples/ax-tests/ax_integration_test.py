#!/usr/bin/env python3
"""Integration test comparing W&B Ax wrapper vs Direct Ax Client.

This script validates that Ax run through the W&B Sweep API produces
equivalent results to Ax run directly via the Ax Client API.

The test uses YAML config files that go through the full parsing path:
  YAML file -> SweepConfig validation -> next_runs() -> ax_search_next_runs()

This ensures the W&B configuration parsing and validation works correctly.

Two test modes are available:
  - deterministic: Verifies exact match with same random seeds (default)
  - statistical: Uses TOST equivalence test for statistical validation

Usage:
    # Deterministic mode: Quick sanity check (5 replications)
    python ax_integration_test.py --mode deterministic --replications 5

    # Deterministic mode: Standard test (25 replications)
    python ax_integration_test.py --mode deterministic --replications 25

    # Statistical mode: TOST equivalence test (30 replications recommended)
    python ax_integration_test.py --mode statistical --replications 30

    # Statistical mode: Custom epsilon and alpha
    python ax_integration_test.py --mode statistical --replications 30 \
        --epsilon 0.01 --alpha 0.05

Output:
    - Comparison of hypervolume values between implementations
    - Deterministic mode: Verification that results match exactly
    - Statistical mode: TOST equivalence test with CI and effect size
"""

import argparse
import os
import sys
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

try:
    import torch
except ImportError:
    print("Error: torch is required for this benchmark.")
    print("Install with: pip install torch")
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
except ImportError as e:
    print("Error: ax-platform is required.")
    print("Install with: pip install ax-platform")
    sys.exit(1)

# Add path to import sweeps
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from benchmark_problems import (  # noqa: E402
    MultiObjectiveBoTorchProblem,
    MOO_PROBLEM_REGISTRY,
)
from sweeps.config import SweepConfig  # noqa: E402
from sweeps.run import next_runs, RunState, SweepRun  # noqa: E402

# Map problem names to YAML config files
YAML_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
PROBLEM_YAML_MAP = {
    "dtlz2": os.path.join(YAML_CONFIG_DIR, "ax_dtlz2_moo.yaml"),
    "c2dtlz2": os.path.join(YAML_CONFIG_DIR, "ax_c2dtlz2_moo.yaml"),
    "welded_beam": os.path.join(YAML_CONFIG_DIR, "ax_welded_beam_moo.yaml"),
}

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_direct_ax_client(
    problem: MultiObjectiveBoTorchProblem,
    random_seed: int,
) -> Client:
    """Create an Ax Client matching ax_search.py configuration exactly.

    This mirrors the configuration in ax_search._create_ax_client_from_config
    (lines 231-338) to ensure the direct client uses identical settings.

    Args:
        problem: Multi-objective test problem
        random_seed: Random seed for reproducibility

    Returns:
        Configured Ax Client ready for optimization
    """
    # Convert problem bounds to Ax parameter configs
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

    # Create client with random seed (matches ax_search.py line 231)
    client = Client(random_seed=random_seed)

    # Configure experiment (matches ax_search.py lines 234-238)
    client.configure_experiment(
        parameters=ax_parameters,
        name=problem.name,
    )

    # Build optimization config with thresholds (matches ax_search.py lines 248-289)
    # This uses the official Ax API for MOO with objective thresholds
    objectives = []
    objective_thresholds = []

    for i, name in enumerate(problem.objective_names):
        metric = Metric(name=name)
        # All objectives are minimization for our test problems
        objectives.append(Objective(metric=metric, minimize=True))
        # Use reference point as threshold
        objective_thresholds.append(
            ObjectiveThreshold(
                metric=metric,
                bound=problem.ref_point[i],
                relative=False,
                op=ComparisonOp.LEQ,  # LEQ for minimization
            )
        )

    # Parse outcome constraints if present
    parsed_constraints = []
    if problem.num_constraints > 0:
        for name in problem.constraint_names:
            parsed_constraints.append(parse_outcome_constraint(f"{name} <= 0"))

    # Create optimization config with thresholds
    opt_config = MultiObjectiveOptimizationConfig(
        objective=MultiObjective(objectives=objectives),
        objective_thresholds=objective_thresholds,
        outcome_constraints=parsed_constraints,
    )
    client.set_optimization_config(opt_config)

    # Configure generation strategy (matches ax_search.py lines 334-338)
    # CRITICAL: initialize_with_center=False must match ax_search.py
    client.configure_generation_strategy(
        method="fast",
        initialization_random_seed=random_seed,
        initialize_with_center=False,  # CRITICAL: must match ax_search.py
    )

    return client


def run_direct_ax_replication(
    problem: MultiObjectiveBoTorchProblem,
    num_trials: int,
    random_seed: int,
) -> Dict[str, Any]:
    """Run a single MOO replication using direct Ax Client API.

    Args:
        problem: Multi-objective test problem
        num_trials: Total number of trials to run
        random_seed: Random seed for reproducibility

    Returns:
        Dict containing:
            - 'hypervolume_curve': List of hypervolume values at each trial
            - 'final_hypervolume': Final hypervolume value
            - 'seed': Random seed used
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Create and configure client
    client = create_direct_ax_client(problem, random_seed)

    # Track results
    all_Y = []  # All objective values
    all_feasible = []  # Feasibility of each point
    hypervolume_curve = []

    for trial in range(num_trials):
        # Get next trial from Ax
        next_trials = client.get_next_trials(max_trials=1)
        if not next_trials:
            break

        trial_index, params = list(next_trials.items())[0]

        # Evaluate the problem
        result = problem.evaluate(params)

        # Extract objective values
        obj_values = [result[name] for name in problem.objective_names]
        all_Y.append(obj_values)

        # Check feasibility (all constraints <= 0)
        if problem.num_constraints > 0:
            constraint_values = [result[name] for name in problem.constraint_names]
            is_feasible = all(g <= 0 for g in constraint_values)
        else:
            is_feasible = True

        all_feasible.append(is_feasible)

        # Complete trial with results
        raw_data = {}
        for name in problem.objective_names:
            raw_data[name] = result[name]
        for name in problem.constraint_names:
            raw_data[name] = result[name]

        client.complete_trial(trial_index=trial_index, raw_data=raw_data)

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

    final_hv = hypervolume_curve[-1] if hypervolume_curve else 0.0

    return {
        "hypervolume_curve": hypervolume_curve,
        "final_hypervolume": final_hv,
        "seed": random_seed,
    }


def run_wandb_ax_replication(
    problem: MultiObjectiveBoTorchProblem,
    problem_name: str,
    num_trials: int,
    random_seed: int,
) -> Dict[str, Any]:
    """Run a single MOO replication using W&B sweep API with YAML config.

    This goes through the full config parsing path:
      YAML file -> SweepConfig validation -> next_runs() -> ax_search_next_runs()

    Args:
        problem: Multi-objective test problem
        problem_name: Name of the problem (used to look up YAML config)
        num_trials: Total number of trials to run
        random_seed: Random seed for reproducibility

    Returns:
        Dict containing:
            - 'hypervolume_curve': List of hypervolume values at each trial
            - 'final_hypervolume': Final hypervolume value
            - 'seed': Random seed used
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Load config from YAML file and validate through SweepConfig
    yaml_path = PROBLEM_YAML_MAP[problem_name]
    with open(yaml_path) as f:
        raw_config = yaml.safe_load(f)

    # Validate config through SweepConfig (this is the key test!)
    config = SweepConfig(raw_config)

    # Track results
    runs = []
    all_Y = []
    all_feasible = []
    hypervolume_curve = []

    for trial in range(num_trials):
        # Get next suggestion through full W&B path: next_runs() -> ax_search_next_runs()
        suggestions = next_runs(
            config, runs, validate=False, n=1, random_seed=random_seed
        )

        if not suggestions:
            break

        # Extract parameters
        params = {k: v["value"] for k, v in suggestions[0].config.items()}

        # Evaluate the problem
        result = problem.evaluate(params)

        # Extract objective values
        obj_values = [result[name] for name in problem.objective_names]
        all_Y.append(obj_values)

        # Check feasibility
        if problem.num_constraints > 0:
            constraint_values = [result[name] for name in problem.constraint_names]
            is_feasible = all(g <= 0 for g in constraint_values)
        else:
            is_feasible = True

        all_feasible.append(is_feasible)

        # Create summary metrics dict for the run
        summary_metrics = {}
        for name in problem.objective_names:
            summary_metrics[name] = result[name]
        for name in problem.constraint_names:
            summary_metrics[name] = result[name]

        # Create run with result
        run = SweepRun(
            state=RunState.finished,
            config={k: {"value": v} for k, v in params.items()},
            summary_metrics=summary_metrics,
        )
        runs.append(run)

        # Compute hypervolume of current Pareto front
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

    final_hv = hypervolume_curve[-1] if hypervolume_curve else 0.0

    return {
        "hypervolume_curve": hypervolume_curve,
        "final_hypervolume": final_hv,
        "seed": random_seed,
    }


def compute_paired_statistics(
    wandb_hvs: List[float],
    direct_hvs: List[float],
) -> Dict[str, float]:
    """Compute paired statistics comparing W&B Ax vs Direct Ax.

    With same seeds, implementations should produce identical results.
    We compute max absolute difference to verify equivalence.

    Args:
        wandb_hvs: Final hypervolume values from W&B Ax replications
        direct_hvs: Final hypervolume values from Direct Ax replications

    Returns:
        Dict containing:
            - 'wandb_mean': Mean HV from W&B Ax
            - 'wandb_std': Std HV from W&B Ax
            - 'direct_mean': Mean HV from Direct Ax
            - 'direct_std': Std HV from Direct Ax
            - 'mean_diff': Mean difference (W&B - Direct)
            - 'max_abs_diff': Maximum absolute difference
            - 'all_match': True if all values match exactly
    """
    wandb_arr = np.array(wandb_hvs)
    direct_arr = np.array(direct_hvs)
    differences = wandb_arr - direct_arr

    return {
        "wandb_mean": float(np.mean(wandb_arr)),
        "wandb_std": float(np.std(wandb_arr, ddof=1)),
        "direct_mean": float(np.mean(direct_arr)),
        "direct_std": float(np.std(direct_arr, ddof=1)),
        "mean_diff": float(np.mean(differences)),
        "max_abs_diff": float(np.max(np.abs(differences))),
        "all_match": bool(np.allclose(differences, 0, atol=1e-10)),
    }


def compute_statistical_equivalence(
    wandb_hvs: np.ndarray,
    direct_hvs: np.ndarray,
    epsilon: float = 0.01,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """TOST equivalence test for paired hypervolume values.

    Tests whether W&B Ax and Direct Ax produce equivalent results within
    a specified margin (epsilon). Uses Two One-Sided Tests (TOST) procedure.

    TOST tests the null hypothesis that the methods differ by MORE than epsilon.
    If we reject this (p < alpha), we conclude the methods are equivalent.

    Args:
        wandb_hvs: Hypervolume values from W&B Ax replications
        direct_hvs: Hypervolume values from Direct Ax replications
        epsilon: Equivalence margin (default 0.01 = 1% hypervolume)
        alpha: Significance level (default 0.05)

    Returns:
        Dict containing:
            - 'tost_p_value': p-value from TOST procedure
            - 'is_equivalent': True if p < alpha (methods equivalent)
            - 'mean_diff': Mean paired difference
            - 'ci_95': 95% confidence interval for the difference
            - 'cohens_d': Effect size (Cohen's d)
            - 'se_diff': Standard error of the difference
    """
    from scipy import stats

    wandb_arr = np.array(wandb_hvs)
    direct_arr = np.array(direct_hvs)
    diff = wandb_arr - direct_arr

    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se_diff = std_diff / np.sqrt(n)

    # TOST: Two one-sided t-tests
    # Test 1: H0: mean_diff >= epsilon vs H1: mean_diff < epsilon
    # Test 2: H0: mean_diff <= -epsilon vs H1: mean_diff > -epsilon
    if se_diff > 0:
        t_upper = (mean_diff - epsilon) / se_diff
        t_lower = (mean_diff + epsilon) / se_diff
        p_upper = stats.t.cdf(t_upper, df=n - 1)
        p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)
        p_tost = max(p_upper, p_lower)
    else:
        # If no variance, check if mean is within bounds
        p_tost = 0.0 if abs(mean_diff) < epsilon else 1.0

    # Effect size (Cohen's d)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    # 95% CI for the difference
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    return {
        "tost_p_value": float(p_tost),
        "is_equivalent": bool(p_tost < alpha),
        "mean_diff": float(mean_diff),
        "ci_95": (float(ci_low), float(ci_high)),
        "cohens_d": float(cohens_d),
        "se_diff": float(se_diff),
        "n": n,
        "epsilon": epsilon,
        "alpha": alpha,
    }


def run_integration_test(
    problem_name: str,
    num_init: int,
    num_trials: int,
    num_replications: int,
    base_seed: int = 0,
    mode: str = "deterministic",
    epsilon: float = 0.01,
    alpha: float = 0.05,
) -> Tuple[Dict[str, Any], bool]:
    """Run the full integration test.

    Two modes are supported:
    - deterministic: With same random seeds, implementations should produce
      identical results. Verifies exact match.
    - statistical: Uses TOST equivalence test to verify implementations
      produce statistically equivalent results within margin epsilon.

    Args:
        problem_name: Name of the MOO problem to test
        num_init: Number of initial points (used by Ax internally)
        num_trials: Total trials per replication
        num_replications: Number of paired replications
        base_seed: Base random seed
        mode: Test mode ('deterministic' or 'statistical')
        epsilon: Equivalence margin for statistical mode
        alpha: Significance level for statistical mode

    Returns:
        Tuple of (results dict, passed bool)
    """
    # Get problem
    if problem_name not in MOO_PROBLEM_REGISTRY:
        raise ValueError(f"Unknown problem: {problem_name}")

    problem = MOO_PROBLEM_REGISTRY[problem_name]()

    # Print header
    yaml_path = PROBLEM_YAML_MAP[problem_name]
    print("=" * 70)
    if mode == "deterministic":
        print("DETERMINISTIC TEST: W&B Ax vs Direct Ax (exact match)")
    else:
        print("STATISTICAL EQUIVALENCE TEST: W&B Ax vs Direct Ax")
    print("=" * 70)
    print(f"Problem: {problem.name} ({problem.dim}D, {problem.num_objectives} objectives)")
    print(f"YAML config: {os.path.basename(yaml_path)}")
    print(f"Trials: {num_trials}, Replications: {num_replications}")
    if mode == "statistical":
        print(f"Epsilon: {epsilon}, Alpha: {alpha}")
    print("=" * 70)

    print("\nRunning paired comparisons...")

    wandb_hvs = []
    direct_hvs = []

    for rep in range(num_replications):
        seed = base_seed + rep

        # Run W&B Ax (through YAML -> SweepConfig -> next_runs path)
        wandb_result = run_wandb_ax_replication(problem, problem_name, num_trials, seed)
        wandb_hv = wandb_result["final_hypervolume"]
        wandb_hvs.append(wandb_hv)

        # Run Direct Ax
        direct_result = run_direct_ax_replication(problem, num_trials, seed)
        direct_hv = direct_result["final_hypervolume"]
        direct_hvs.append(direct_hv)

        diff = wandb_hv - direct_hv
        print(
            f"  Rep {rep + 1:3d}/{num_replications} (seed={seed:3d})   "
            f"W&B: {wandb_hv:.4f}  Direct: {direct_hv:.4f}  Diff: {diff:+.6f}"
        )

    # Compute basic statistics
    stats = compute_paired_statistics(wandb_hvs, direct_hvs)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'':24s}{'W&B Ax':>14s}{'Direct Ax':>14s}")
    print(f"{'Mean HV:':24s}{stats['wandb_mean']:14.4f}{stats['direct_mean']:14.4f}")
    print(f"{'Std HV:':24s}{stats['wandb_std']:14.4f}{stats['direct_std']:14.4f}")

    print(f"\nPaired Difference:")
    print(f"  Mean:      {stats['mean_diff']:.6f}")
    print(f"  Max |diff|: {stats['max_abs_diff']:.6f}")

    if mode == "deterministic":
        # Deterministic mode: check exact match
        passed = stats["all_match"]

        print("\nCONCLUSION:")
        if passed:
            print("  PASS: All hypervolume values match exactly.")
            print("  The W&B Sweep API produces identical results to direct Ax.")
        else:
            print("  FAIL: Differences detected!")
            print(f"  Max abs difference: {stats['max_abs_diff']:.6f} HV")
            print("  WARNING: The W&B Sweep API may not match direct Ax behavior!")

        results = {
            "problem": problem_name,
            "num_trials": num_trials,
            "num_replications": num_replications,
            "mode": mode,
            "wandb_hvs": wandb_hvs,
            "direct_hvs": direct_hvs,
            "statistics": stats,
            "passed": passed,
        }

    else:
        # Statistical mode: TOST equivalence test
        equiv_stats = compute_statistical_equivalence(
            wandb_hvs, direct_hvs, epsilon=epsilon, alpha=alpha
        )

        print(f"  95% CI:    [{equiv_stats['ci_95'][0]:.6f}, {equiv_stats['ci_95'][1]:.6f}]")
        print(f"  Cohen's d: {equiv_stats['cohens_d']:.4f}")
        print(f"\nTOST Equivalence Test (epsilon={epsilon}):")
        print(f"  p-value: {equiv_stats['tost_p_value']:.4f}")

        passed = equiv_stats["is_equivalent"]

        print("\nCONCLUSION:")
        if passed:
            print(f"  PASS: Methods equivalent within epsilon={epsilon}")
            print(f"  (TOST p={equiv_stats['tost_p_value']:.4f} < alpha={alpha})")
        else:
            print(f"  FAIL: Cannot conclude equivalence within epsilon={epsilon}")
            print(f"  (TOST p={equiv_stats['tost_p_value']:.4f} >= alpha={alpha})")
            if num_replications < 30:
                print(f"  Note: Consider increasing replications (current: {num_replications})")

        results = {
            "problem": problem_name,
            "num_trials": num_trials,
            "num_replications": num_replications,
            "mode": mode,
            "epsilon": epsilon,
            "alpha": alpha,
            "wandb_hvs": wandb_hvs,
            "direct_hvs": direct_hvs,
            "statistics": stats,
            "equivalence_stats": equiv_stats,
            "passed": passed,
        }

    print("=" * 70)

    return results, passed


def main():
    """Run MOO integration test comparing W&B Ax vs Direct Ax."""
    parser = argparse.ArgumentParser(
        description="Integration test: W&B Ax vs Direct Ax Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="dtlz2",
        choices=list(MOO_PROBLEM_REGISTRY.keys()),
        help="MOO problem to test",
    )
    parser.add_argument(
        "--init",
        type=int,
        default=None,
        help="Number of initial points (default: 2*d+1)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of optimization trials per replication",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=25,
        help="Number of replications for statistical comparison",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["deterministic", "statistical"],
        default="deterministic",
        help="Test mode: 'deterministic' checks exact match with same seeds, "
        "'statistical' uses TOST equivalence test",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Equivalence margin for statistical mode (default: 0.01 = 1%% HV)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical mode",
    )
    args = parser.parse_args()

    # Get problem to determine default init
    problem = MOO_PROBLEM_REGISTRY[args.problem]()
    num_init = args.init if args.init is not None else 2 * problem.dim + 1

    # Run test
    results, passed = run_integration_test(
        problem_name=args.problem,
        num_init=num_init,
        num_trials=args.trials,
        num_replications=args.replications,
        base_seed=args.seed,
        mode=args.mode,
        epsilon=args.epsilon,
        alpha=args.alpha,
    )

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
