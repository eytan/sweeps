"""Integration tests for Ax Multi-Objective Optimization (MOO).

These tests verify that MOO configs work end-to-end with ax_search_next_runs.
"""

import pytest
from sweeps.ax_search import ax_search_next_runs
from sweeps.run import RunState, SweepRun


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
        """MOO with outcome constraints works correctly."""
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
        """Single objective with outcome constraints works."""
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
