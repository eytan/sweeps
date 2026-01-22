#!/usr/bin/env python3
"""Unified benchmark problem definitions for synthetic function optimization.

This module provides a common interface for all benchmark problems, wrapping
BoTorch synthetic test functions (Branin, Hartmann, AckleyMixed, Labs).

The problem registry allows easy selection of problems by name for benchmarking.

Example Usage:
    from benchmark_problems import PROBLEM_REGISTRY, get_problems

    # Get a single problem
    branin = PROBLEM_REGISTRY["branin"]()

    # Get multiple problems by name
    problems = get_problems(["branin", "hartmann6"])

    # Get all continuous problems
    problems = get_problems(["continuous"])

    # Get all available problems
    problems = get_problems(["all"])
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from botorch.test_functions.synthetic import (
    AckleyMixed,
    Branin,
    Hartmann,
    Labs,
    SyntheticTestFunction,
)
from botorch.test_functions.multi_objective import (
    C2DTLZ2,
    ConstrainedBaseTestProblem,
    DTLZ2,
    WeldedBeam,
)
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated


class BaseBenchmarkProblem(ABC):
    """Abstract base class for benchmark problems."""

    name: str
    dim: int
    is_maximization: bool
    optimal_value: Optional[float]

    @abstractmethod
    def evaluate(self, params_dict: Dict[str, float]) -> float:
        """Evaluate function given parameter dictionary.

        Args:
            params_dict: Dict like {'x0': 0.5, 'x1': 0.3, ...}

        Returns:
            Function value at the given point
        """
        pass

    @abstractmethod
    def create_sweep_config(self, method: str) -> Dict[str, Any]:
        """Create sweep configuration for this problem.

        Args:
            method: Optimization method ('bayes' or 'ax')

        Returns:
            Sweep configuration dict
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get problem metadata for result files.

        Returns:
            Dict containing problem metadata for serialization
        """
        pass

    def get_best(self, values: List[float]) -> float:
        """Returns max or min based on is_maximization.

        Args:
            values: List of function values

        Returns:
            Best value (max if maximization, min if minimization)
        """
        if not values:
            return float("-inf") if self.is_maximization else float("inf")
        return max(values) if self.is_maximization else min(values)

    def compute_gap(self, best_value: float) -> float:
        """Compute gap to optimum.

        Args:
            best_value: Best value found

        Returns:
            Gap to optimum (positive means suboptimal)
        """
        if self.optimal_value is None:
            return float("nan")
        if self.is_maximization:
            return self.optimal_value - best_value
        else:
            return best_value - self.optimal_value


class BoTorchProblem(BaseBenchmarkProblem):
    """Wrapper for BoTorch synthetic test functions."""

    def __init__(
        self,
        function: SyntheticTestFunction,
        name: str,
        is_maximization: Optional[bool] = None,
    ):
        """Initialize test problem wrapper.

        Args:
            function: BoTorch SyntheticTestFunction instance
            name: Descriptive name for the problem
            is_maximization: Whether the problem is a maximization problem.
                If None, inferred from function._is_minimization_by_default.
        """
        self.function = function
        self.name = name
        self.dim = function.dim
        self.discrete_inds = getattr(function, "discrete_inds", [])
        self.continuous_inds = getattr(
            function, "continuous_inds", list(range(self.dim))
        )
        # BoTorch bounds have shape (2, dim) where bounds[0] are lower and bounds[1] are upper
        raw_bounds = function.bounds.tolist()
        # Convert to list of (lower, upper) tuples for each dimension
        self.bounds = list(zip(raw_bounds[0], raw_bounds[1]))

        # Determine if maximization problem
        if is_maximization is not None:
            self.is_maximization = is_maximization
        else:
            # BoTorch uses _is_minimization_by_default (default True)
            is_min_by_default = getattr(function, "_is_minimization_by_default", True)
            self.is_maximization = not is_min_by_default

        # Get optimal value
        try:
            self.optimal_value = function.optimal_value
        except (NotImplementedError, AttributeError):
            self.optimal_value = None

    def evaluate(self, params_dict: Dict[str, float]) -> float:
        """Evaluate function given parameter dictionary.

        Args:
            params_dict: Dict like {'x0': 0.5, 'x1': 0.3, ...}

        Returns:
            Function value at the given point
        """
        X = torch.tensor(
            [[params_dict[f"x{i}"] for i in range(self.dim)]],
            dtype=torch.double,
        )
        return float(self.function(X).item())

    def create_sweep_config(self, method: str) -> Dict[str, Any]:
        """Create sweep configuration for this problem.

        Args:
            method: Optimization method ('bayes' or 'ax')

        Returns:
            Sweep configuration dict with appropriate parameter types
        """
        parameters = {}

        for i in range(self.dim):
            lower, upper = self.bounds[i]

            if i in self.discrete_inds:
                # Discrete (binary) parameter - use values
                parameters[f"x{i}"] = {"values": [float(lower), float(upper)]}
            else:
                # Continuous parameter - use min/max
                parameters[f"x{i}"] = {"min": float(lower), "max": float(upper)}

        return {
            "method": method,
            "parameters": parameters,
            "metric": {
                "name": "value",
                "goal": "maximize" if self.is_maximization else "minimize",
            },
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get problem metadata for result files.

        Returns:
            Dict containing problem metadata for serialization
        """
        # Determine problem type
        if not self.discrete_inds:
            problem_type = "continuous"
        elif not self.continuous_inds or len(self.continuous_inds) == 0:
            problem_type = "discrete"
        else:
            problem_type = "mixed"

        return {
            "problem_name": self.name,
            "dim": self.dim,
            "is_maximization": self.is_maximization,
            "optimal_value": self.optimal_value,
            "problem_type": problem_type,
            "num_discrete_params": len(self.discrete_inds),
            "num_continuous_params": len(self.continuous_inds),
        }


class MultiObjectiveBoTorchProblem:
    """Wrapper for BoTorch multi-objective test functions."""

    def __init__(
        self,
        func: Union[ConstrainedBaseTestProblem, Any],
        name: str,
        ref_point: List[float],
        objective_names: Optional[List[str]] = None,
        constraint_names: Optional[List[str]] = None,
    ):
        """Initialize multi-objective test problem wrapper.

        Args:
            func: BoTorch multi-objective test function instance
            name: Descriptive name for the problem
            ref_point: Reference point for hypervolume calculation (for minimization,
                this should be worse than any feasible point in each objective)
            objective_names: Optional names for each objective (defaults to obj0, obj1, ...)
            constraint_names: Optional names for each constraint (defaults to g0, g1, ...)
        """
        self.func = func
        self.name = name
        self.ref_point = ref_point
        self.dim = func.dim
        self.num_objectives = func.num_objectives

        # Get bounds from function
        raw_bounds = func.bounds.tolist()
        self.bounds = list(zip(raw_bounds[0], raw_bounds[1]))

        # Handle constraints
        self.num_constraints = getattr(func, "num_constraints", 0)

        # Set objective names
        if objective_names is not None:
            self.objective_names = objective_names
        else:
            self.objective_names = [f"obj{i}" for i in range(self.num_objectives)]

        # Set constraint names
        if constraint_names is not None:
            self.constraint_names = constraint_names
        else:
            self.constraint_names = [f"g{i}" for i in range(self.num_constraints)]

        # Create hypervolume calculator
        # For minimization problems, we negate objectives and reference point
        # so that BoTorch's Hypervolume (which expects maximization) works correctly
        self._neg_ref_point = torch.tensor(
            [-r for r in ref_point], dtype=torch.double
        )
        self._hv = Hypervolume(ref_point=self._neg_ref_point)

    def evaluate(self, params_dict: Dict[str, float]) -> Dict[str, float]:
        """Evaluate function given parameter dictionary.

        Args:
            params_dict: Dict like {'x0': 0.5, 'x1': 0.3, ...}

        Returns:
            Dict with objective values and constraint values:
            {
                'obj0': float, 'obj1': float, ...,  # objective values
                'g0': float, 'g1': float, ...       # constraint values (if any)
            }
        """
        X = torch.tensor(
            [[params_dict[f"x{i}"] for i in range(self.dim)]],
            dtype=torch.double,
        )

        # Evaluate objectives
        objectives = self.func(X)  # Shape: (1, num_objectives)

        result = {}
        for i, name in enumerate(self.objective_names):
            result[name] = float(objectives[0, i].item())

        # Evaluate constraints if present
        if self.num_constraints > 0:
            constraints = self.func.evaluate_slack(X)  # Shape: (1, num_constraints)
            for i, name in enumerate(self.constraint_names):
                # BoTorch returns slack (positive = feasible), we want g <= 0
                # So we negate: constraint violation is g = -slack
                result[name] = float(-constraints[0, i].item())

        return result

    def create_sweep_config(self, method: str = "ax") -> Dict[str, Any]:
        """Create sweep configuration for this problem.

        Args:
            method: Optimization method ('ax' for multi-objective)

        Returns:
            Sweep configuration dict with metrics (plural) for MOO
        """
        parameters = {}

        for i in range(self.dim):
            lower, upper = self.bounds[i]
            parameters[f"x{i}"] = {"min": float(lower), "max": float(upper)}

        # Build metrics list for multi-objective
        metrics = []
        for name in self.objective_names:
            # All objectives are minimization (negate=True used in WeldedBeam)
            metrics.append({"name": name, "goal": "minimize"})

        config = {
            "method": method,
            "parameters": parameters,
            "metrics": metrics,
        }

        # Add metric constraints if present
        if self.num_constraints > 0:
            metric_constraints = []
            for name in self.constraint_names:
                metric_constraints.append(f"{name} <= 0")
            config["metric_constraints"] = metric_constraints

        return config

    def compute_hypervolume(self, pareto_Y: np.ndarray) -> float:
        """Compute hypervolume given Pareto front objective values.

        For minimization problems, this negates the objectives before computing
        hypervolume since BoTorch's Hypervolume expects maximization.

        Args:
            pareto_Y: Array of Pareto front objective values, shape (n_points, num_objectives)

        Returns:
            Hypervolume dominated by the Pareto front
        """
        if len(pareto_Y) == 0:
            return 0.0

        # Negate for minimization (BoTorch Hypervolume expects maximization)
        pareto_Y_tensor = torch.tensor(-pareto_Y, dtype=torch.double)
        return float(self._hv.compute(pareto_Y_tensor))

    def get_pareto_front(
        self, all_Y: np.ndarray, all_feasible: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extract Pareto front from all evaluated points.

        Args:
            all_Y: Array of all objective values, shape (n_points, num_objectives)
            all_feasible: Optional boolean array indicating feasibility of each point

        Returns:
            Array of Pareto-optimal objective values
        """
        if len(all_Y) == 0:
            return np.array([])

        # Filter to feasible points if mask provided
        if all_feasible is not None:
            feasible_Y = all_Y[all_feasible]
            if len(feasible_Y) == 0:
                return np.array([])
            Y_tensor = torch.tensor(feasible_Y, dtype=torch.double)
        else:
            Y_tensor = torch.tensor(all_Y, dtype=torch.double)

        # For minimization, we want points where no other point dominates
        # is_non_dominated expects maximization, so we negate
        pareto_mask = is_non_dominated(-Y_tensor)

        return Y_tensor[pareto_mask].numpy()

    def get_metadata(self) -> Dict[str, Any]:
        """Get problem metadata for result files.

        Returns:
            Dict containing problem metadata for serialization
        """
        return {
            "problem_name": self.name,
            "dim": self.dim,
            "num_objectives": self.num_objectives,
            "num_constraints": self.num_constraints,
            "ref_point": self.ref_point,
            "objective_names": self.objective_names,
            "constraint_names": self.constraint_names,
            "bounds": self.bounds,
            "problem_type": "multi_objective",
        }


# Problem registry mapping names to factory functions
PROBLEM_REGISTRY: Dict[str, Any] = {
    # Single-objective: continuous
    "branin": lambda: BoTorchProblem(Branin(), "Branin_2D"),
    "hartmann6": lambda: BoTorchProblem(Hartmann(dim=6), "Hartmann6_6D"),
    # Single-objective: mixed/discrete
    "ackley_mixed_23d": lambda: BoTorchProblem(AckleyMixed(dim=23), "AckleyMixed_23D"),
    "labs_20d": lambda: BoTorchProblem(Labs(dim=20), "Labs_20D"),
    # Multi-objective
    # WeldedBeam: 4D, 2 obj (cost, deflection), 4 constraints, minimize, ref=[40, 0.015]
    "welded_beam": lambda: MultiObjectiveBoTorchProblem(
        WeldedBeam(negate=False),
        "WeldedBeam_4D",
        ref_point=[40.0, 0.015],
        objective_names=["cost", "deflection"],
        constraint_names=["g1", "g2", "g3", "g4"],
    ),
    # DTLZ2: 6D, 2 obj, unconstrained, Pareto front on unit circle arc
    "dtlz2": lambda: MultiObjectiveBoTorchProblem(
        DTLZ2(dim=6, num_objectives=2, negate=False),
        "DTLZ2_6D",
        ref_point=[1.1, 1.1],
        objective_names=["f1", "f2"],
    ),
    # C2DTLZ2: 6D, 2 obj, 1 constraint, constrained DTLZ2
    "c2dtlz2": lambda: MultiObjectiveBoTorchProblem(
        C2DTLZ2(dim=6, num_objectives=2, negate=False),
        "C2DTLZ2_6D",
        ref_point=[1.1, 1.1],
        objective_names=["f1", "f2"],
        constraint_names=["c1"],
    ),
}

# Backwards compatibility alias
MOO_PROBLEM_REGISTRY = {
    k: v for k, v in PROBLEM_REGISTRY.items()
    if k in ("welded_beam", "dtlz2", "c2dtlz2")
}

# Problem groups for convenience
CONTINUOUS_PROBLEMS = ["branin", "hartmann6"]
MIXED_PROBLEMS = ["ackley_mixed_23d", "labs_20d"]
SOO_PROBLEMS = CONTINUOUS_PROBLEMS + MIXED_PROBLEMS
MOO_PROBLEMS = ["welded_beam", "dtlz2", "c2dtlz2"]
ALL_PROBLEMS = SOO_PROBLEMS + MOO_PROBLEMS


def get_problems(problem_names: List[str]) -> List[Any]:
    """Get problem instances from names.

    Args:
        problem_names: List of problem names or keywords ('all', 'soo', 'moo',
            'continuous', 'mixed')

    Returns:
        List of problem instances (BoTorchProblem or MultiObjectiveBoTorchProblem)
    """
    resolved_names = []
    keywords = ["all", "soo", "moo", "continuous", "mixed"]

    for name in problem_names:
        if name == "all":
            resolved_names.extend(ALL_PROBLEMS)
        elif name == "soo":
            resolved_names.extend(SOO_PROBLEMS)
        elif name == "moo":
            resolved_names.extend(MOO_PROBLEMS)
        elif name == "continuous":
            resolved_names.extend(CONTINUOUS_PROBLEMS)
        elif name == "mixed":
            resolved_names.extend(MIXED_PROBLEMS)
        elif name in PROBLEM_REGISTRY:
            resolved_names.append(name)
        else:
            available = list(PROBLEM_REGISTRY.keys()) + keywords
            raise ValueError(
                f"Unknown problem: {name}. Available: {', '.join(available)}"
            )

    return [PROBLEM_REGISTRY[name]() for name in resolved_names]


def get_moo_problems(problem_names: List[str]) -> List[MultiObjectiveBoTorchProblem]:
    """Get multi-objective problem instances from names. Wrapper for get_problems()."""
    # Handle 'all' as 'moo' for backwards compatibility
    names = ["moo" if n == "all" else n for n in problem_names]
    return get_problems(names)


PROBLEM_DESCRIPTIONS: Dict[str, str] = {
    # Single-objective
    "branin": "Branin 2D (continuous, minimization, optimal=0.398)",
    "hartmann6": "Hartmann6 6D (continuous, minimization, optimal=-3.322)",
    "ackley_mixed_23d": "AckleyMixed 23D (20 binary + 3 continuous, minimization, optimal=0.0)",
    "labs_20d": "Labs 20D (all binary, maximization)",
    # Multi-objective
    "welded_beam": "WeldedBeam 4D (2 objectives: cost, deflection; 4 constraints)",
    "dtlz2": "DTLZ2 6D (2 objectives: f1, f2; unconstrained)",
    "c2dtlz2": "C2DTLZ2 6D (2 objectives: f1, f2; 1 constraint)",
}


def list_available_problems(problem_type: Optional[str] = None) -> Dict[str, str]:
    """List available problems with descriptions.

    Args:
        problem_type: Filter by type - 'soo', 'moo', or None for all
    """
    if problem_type == "soo":
        names = SOO_PROBLEMS
    elif problem_type == "moo":
        names = MOO_PROBLEMS
    else:
        names = list(PROBLEM_REGISTRY.keys())
    return {name: PROBLEM_DESCRIPTIONS[name] for name in names}


def list_available_moo_problems() -> Dict[str, str]:
    """List available MOO problems. Wrapper for list_available_problems()."""
    return list_available_problems("moo")
