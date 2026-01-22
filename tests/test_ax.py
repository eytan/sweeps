"""Integration tests for Ax Bayesian Optimization."""

import pytest
from sweeps.ax_search import ax_search_next_runs
from sweeps.run import RunState, SweepRun


class TestSOOIntegration:
    """Integration tests for single-objective optimization."""

    def test_branin_style_optimization(self):
        """Single-objective optimization with Branin-like 2D search space."""
        config = {
            "method": "ax",
            "parameters": {
                "x1": {"min": -5.0, "max": 10.0},
                "x2": {"min": 0.0, "max": 15.0},
            },
            "metric": {"name": "objective", "goal": "minimize"},
        }

        suggestions = ax_search_next_runs([], config, n=3)

        assert len(suggestions) == 3
        for s in suggestions:
            assert -5.0 <= s.config["x1"]["value"] <= 10.0
            assert 0.0 <= s.config["x2"]["value"] <= 15.0
            assert s.search_info.get("is_multi_objective") is None

    def test_soo_with_historical_data(self):
        """Single-objective optimization uses historical data."""
        config = {
            "method": "ax",
            "parameters": {
                "x1": {"min": -5.0, "max": 10.0},
                "x2": {"min": 0.0, "max": 15.0},
            },
            "metric": {"name": "objective", "goal": "minimize"},
        }

        runs = [
            SweepRun(
                state=RunState.finished,
                config={"x1": {"value": 0.0}, "x2": {"value": 5.0}},
                summary_metrics={"objective": 10.5},
            ),
            SweepRun(
                state=RunState.finished,
                config={"x1": {"value": 3.14}, "x2": {"value": 2.27}},
                summary_metrics={"objective": 0.4},
            ),
            SweepRun(
                state=RunState.finished,
                config={"x1": {"value": -3.0}, "x2": {"value": 12.0}},
                summary_metrics={"objective": 25.0},
            ),
        ]

        suggestions = ax_search_next_runs(runs, config, n=2)

        assert len(suggestions) == 2

    def test_soo_maximize(self):
        """Single-objective maximization works correctly."""
        config = {
            "method": "ax",
            "parameters": {
                "x": {"min": 0.0, "max": 1.0},
            },
            "metric": {"name": "score", "goal": "maximize"},
        }

        suggestions = ax_search_next_runs([], config, n=2)

        assert len(suggestions) == 2
        for s in suggestions:
            assert 0.0 <= s.config["x"]["value"] <= 1.0


class TestMOOIntegration:
    """Integration tests for MOO functionality."""

    def test_basic_2_objectives(self):
        """Basic MOO with 2 objectives generates valid suggestions."""
        config = {
            "method": "ax",
            "parameters": {
                "learning_rate": {"min": 0.001, "max": 0.1},
                "batch_size": {"min": 16, "max": 128},
            },
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
                {"name": "latency", "goal": "minimize"},
            ],
        }

        suggestions = ax_search_next_runs([], config, n=2)

        assert len(suggestions) == 2
        assert suggestions[0].search_info.get("is_multi_objective") is True
        assert suggestions[0].search_info.get("objective_names") == ["accuracy", "latency"]

    def test_moo_with_constraints(self):
        """MOO with metric constraints works correctly."""
        config = {
            "method": "ax",
            "parameters": {
                "learning_rate": {"min": 0.001, "max": 0.1},
                "batch_size": {"min": 16, "max": 128},
            },
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
                {"name": "model_size", "goal": "minimize"},
            ],
            "metric_constraints": ["latency <= 100"],
        }

        suggestions = ax_search_next_runs([], config, n=2)

        assert len(suggestions) == 2
        assert suggestions[0].search_info["is_multi_objective"] is True

    def test_moo_with_historical_data(self):
        """MOO uses historical data to guide optimization."""
        config = {
            "method": "ax",
            "parameters": {
                "learning_rate": {"min": 0.001, "max": 0.1},
                "batch_size": {"min": 16, "max": 128},
            },
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
                {"name": "latency", "goal": "minimize"},
            ],
        }

        runs = [
            SweepRun(
                state=RunState.finished,
                config={"learning_rate": {"value": 0.01}, "batch_size": {"value": 32}},
                summary_metrics={"accuracy": 0.85, "latency": 50},
            ),
            SweepRun(
                state=RunState.finished,
                config={"learning_rate": {"value": 0.001}, "batch_size": {"value": 64}},
                summary_metrics={"accuracy": 0.90, "latency": 80},
            ),
        ]

        suggestions = ax_search_next_runs(runs, config, n=2)

        assert len(suggestions) == 2
        assert suggestions[0].search_info["is_multi_objective"] is True

    def test_single_objective_backward_compatibility(self):
        """Single 'metric' config still works (backward compatibility)."""
        config = {
            "method": "ax",
            "parameters": {
                "learning_rate": {"min": 0.001, "max": 0.1},
                "batch_size": {"min": 16, "max": 128},
            },
            "metric": {"name": "loss", "goal": "minimize"},
        }

        suggestions = ax_search_next_runs([], config, n=2)

        assert len(suggestions) == 2
        # Single objective should NOT have is_multi_objective
        assert suggestions[0].search_info.get("is_multi_objective") is None

    def test_single_objective_with_constraints(self):
        """Single objective with metric constraints works."""
        config = {
            "method": "ax",
            "parameters": {
                "learning_rate": {"min": 0.001, "max": 0.1},
                "batch_size": {"min": 16, "max": 128},
            },
            "metric": {"name": "accuracy", "goal": "maximize"},
            "metric_constraints": ["latency <= 100"],
        }

        suggestions = ax_search_next_runs([], config, n=2)

        assert len(suggestions) == 2


class TestAxParameterTypes:
    """Tests for different parameter types in Ax search."""

    def test_categorical_parameters(self):
        """Categorical parameters work correctly."""
        config = {
            "method": "ax",
            "parameters": {
                "optimizer": {"values": ["adam", "sgd", "rmsprop"]},
                "lr": {"min": 0.001, "max": 0.1},
            },
            "metric": {"name": "acc", "goal": "maximize"},
        }

        suggestions = ax_search_next_runs([], config, n=2)

        assert len(suggestions) == 2
        for s in suggestions:
            assert s.config["optimizer"]["value"] in ["adam", "sgd", "rmsprop"]
            assert 0.001 <= s.config["lr"]["value"] <= 0.1

    def test_log_scale_parameters(self):
        """Log-scale parameters work correctly."""
        config = {
            "method": "ax",
            "parameters": {
                "learning_rate": {
                    "min": 0.0001,
                    "max": 0.1,
                    "distribution": "log_uniform_values",
                },
                "weight_decay": {
                    "min": 0.00001,
                    "max": 0.01,
                    "distribution": "log_uniform_values",
                },
            },
            "metric": {"name": "loss", "goal": "minimize"},
        }

        suggestions = ax_search_next_runs([], config, n=3)

        assert len(suggestions) == 3
        for s in suggestions:
            assert 0.0001 <= s.config["learning_rate"]["value"] <= 0.1
            assert 0.00001 <= s.config["weight_decay"]["value"] <= 0.01

    def test_with_failed_trials(self):
        """Ax handles failed trials gracefully."""
        config = {
            "method": "ax",
            "parameters": {
                "learning_rate": {"min": 0.001, "max": 0.1},
                "batch_size": {"min": 16, "max": 128},
            },
            "metric": {"name": "loss", "goal": "minimize"},
        }

        runs = [
            SweepRun(
                state=RunState.finished,
                config={"learning_rate": {"value": 0.01}, "batch_size": {"value": 32}},
                summary_metrics={"loss": 0.5},
            ),
            SweepRun(
                state=RunState.failed,
                config={"learning_rate": {"value": 0.1}, "batch_size": {"value": 16}},
            ),
            SweepRun(
                state=RunState.finished,
                config={"learning_rate": {"value": 0.001}, "batch_size": {"value": 64}},
                summary_metrics={"loss": 0.3},
            ),
        ]

        suggestions = ax_search_next_runs(runs, config, n=2)

        assert len(suggestions) == 2

    def test_integer_parameters(self):
        """Integer parameters are returned with correct type."""
        config = {
            "method": "ax",
            "parameters": {
                "num_layers": {"min": 1, "max": 5},
                "hidden_size": {"min": 64, "max": 512},
            },
            "metric": {"name": "loss", "goal": "minimize"},
        }

        suggestions = ax_search_next_runs([], config, n=3)

        assert len(suggestions) == 3
        for s in suggestions:
            # Integer params should be in range
            assert 1 <= s.config["num_layers"]["value"] <= 5
            assert 64 <= s.config["hidden_size"]["value"] <= 512
