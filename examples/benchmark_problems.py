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
from typing import Any, Dict, List, Optional

import torch
from botorch.test_functions.synthetic import (
    AckleyMixed,
    Branin,
    Hartmann,
    Labs,
    SyntheticTestFunction,
)


class BaseBenchmarkProblem(ABC):
    """Abstract base class for benchmark problems.

    All benchmark problems (continuous, mixed, discrete) implement this interface
    to enable unified benchmarking.

    Attributes:
        name: Human-readable name of the problem
        dim: Dimensionality of the search space
        is_maximization: Whether the problem is a maximization problem
        optimal_value: Known global optimum value (None if unknown)
    """

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
    """Wrapper for BoTorch synthetic test functions.

    This class adapts BoTorch's SyntheticTestFunction interface to work with the
    sweeps library's parameter configuration format. It handles continuous,
    mixed, and discrete parameter spaces.

    Attributes:
        function: The BoTorch SyntheticTestFunction instance
        name: Human-readable name of the problem
        dim: Dimensionality of the search space
        discrete_inds: List of indices for discrete (binary) parameters
        continuous_inds: List of indices for continuous parameters
        is_maximization: Whether the problem is a maximization problem
        optimal_value: Known global optimum value (None if unknown)
    """

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


# Problem registry mapping names to factory functions
PROBLEM_REGISTRY: Dict[str, Any] = {
    # Continuous problems (minimization)
    "branin": lambda: BoTorchProblem(Branin(), "Branin_2D"),
    "hartmann6": lambda: BoTorchProblem(Hartmann(dim=6), "Hartmann6_6D"),
    # Mixed/discrete problems
    "ackley_mixed_23d": lambda: BoTorchProblem(AckleyMixed(dim=23), "AckleyMixed_23D"),
    "labs_20d": lambda: BoTorchProblem(Labs(dim=20), "Labs_20D"),
}

# Problem groups for convenience
CONTINUOUS_PROBLEMS = ["branin", "hartmann6"]
MIXED_PROBLEMS = ["ackley_mixed_23d", "labs_20d"]
ALL_PROBLEMS = CONTINUOUS_PROBLEMS + MIXED_PROBLEMS


def get_problems(problem_names: List[str]) -> List[BaseBenchmarkProblem]:
    """Get problem instances from names.

    Supports special keywords:
    - 'all': All available problems
    - 'continuous': Branin, Hartmann6
    - 'mixed': AckleyMixed, Labs

    Args:
        problem_names: List of problem names or keywords

    Returns:
        List of BaseBenchmarkProblem instances

    Raises:
        ValueError: If a problem name is not recognized
    """
    resolved_names = []

    for name in problem_names:
        if name == "all":
            resolved_names.extend(ALL_PROBLEMS)
        elif name == "continuous":
            resolved_names.extend(CONTINUOUS_PROBLEMS)
        elif name == "mixed":
            resolved_names.extend(MIXED_PROBLEMS)
        elif name in PROBLEM_REGISTRY:
            resolved_names.append(name)
        else:
            available = list(PROBLEM_REGISTRY.keys()) + ["all", "continuous", "mixed"]
            raise ValueError(
                f"Unknown problem: {name}. Available: {', '.join(available)}"
            )

    # Remove duplicates while preserving order
    seen = set()
    unique_names = []
    for name in resolved_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    # Create problem instances
    problems = []
    for name in unique_names:
        if name in PROBLEM_REGISTRY:
            problems.append(PROBLEM_REGISTRY[name]())
        else:
            raise ValueError(f"Problem '{name}' is not available (missing dependency)")

    return problems


def list_available_problems() -> Dict[str, str]:
    """List all available problems with descriptions.

    Returns:
        Dict mapping problem name to description
    """
    descriptions = {
        "branin": "Branin 2D (continuous, minimization, optimal=0.398)",
        "hartmann6": "Hartmann6 6D (continuous, minimization, optimal=-3.322)",
        "ackley_mixed_23d": "AckleyMixed 23D (20 binary + 3 continuous, minimization, optimal=0.0)",
        "labs_20d": "Labs 20D (all binary, maximization)",
    }

    available = {}
    for name, desc in descriptions.items():
        if name in PROBLEM_REGISTRY:
            available[name] = desc

    return available
