"""Pytest tests for Ax Multi-Objective Optimization (MOO) and Outcome Constraints.

This module contains comprehensive tests for:
- Metric constraint parsing validation
- Multi-objective metrics validation
- Ax config validation for MOO
- Integration tests for ax_search_next_runs with MOO
- Backward compatibility tests for single-objective optimization
"""

import pytest
from sweeps.ax_search import ax_search_next_runs, _is_moo_config, _validate_config
from sweeps.config.schema import (
    parse_metric_constraint,
    fill_validate_metrics,
    validate_metric_constraints,
    fill_validate_schema,
)
from sweeps.run import RunState, SweepRun


# ============================================================================
# Unit Tests for Constraint Parsing (schema.py)
# ============================================================================


class TestParseMetricConstraint:
    """Unit tests for parse_metric_constraint function."""

    @pytest.mark.parametrize(
        "constraint,expected",
        [
            ("latency <= 100", ("latency", "<=", 100.0)),
            ("accuracy >= 0.9", ("accuracy", ">=", 0.9)),
            ("memory_usage <= 512.5", ("memory_usage", "<=", 512.5)),
            ("loss <= -0.5", ("loss", "<=", -0.5)),
            ("val_loss <= 0.001", ("val_loss", "<=", 0.001)),
            ("training_time_seconds <= 3600", ("training_time_seconds", "<=", 3600.0)),
            ("f1_score >= 0.85", ("f1_score", ">=", 0.85)),
        ],
    )
    def test_parse_valid_constraints(self, constraint, expected):
        """Valid constraints should parse correctly."""
        result = parse_metric_constraint(constraint)
        assert result == expected

    @pytest.mark.parametrize(
        "constraint",
        [
            "latency > 100",  # Invalid operator (> not allowed)
            "latency < 100",  # Invalid operator (< not allowed)
            "123metric <= 100",  # Invalid metric name (starts with number)
            "latency <=",  # Missing value
            "<= 100",  # Missing metric name
            "latency",  # Missing operator and value
            "",  # Empty string
            "latency = 100",  # Invalid operator (= not allowed)
            "latency == 100",  # Invalid operator (== not allowed)
            "latency != 100",  # Invalid operator (!= not allowed)
        ],
    )
    def test_parse_invalid_constraints(self, constraint):
        """Invalid constraints should raise ValueError."""
        with pytest.raises(ValueError):
            parse_metric_constraint(constraint)

    def test_parse_constraint_with_whitespace(self):
        """Constraints with various whitespace should still parse."""
        # Extra whitespace should be handled
        result = parse_metric_constraint("  latency   <=   100  ")
        assert result == ("latency", "<=", 100.0)


class TestValidateMetricConstraints:
    """Unit tests for validate_metric_constraints function."""

    def test_valid_constraints_list(self):
        """Valid list of constraints should not raise."""
        constraints = ["latency <= 100", "memory_usage <= 512"]
        # Should not raise
        validate_metric_constraints(constraints)

    def test_constraints_not_list_raises(self):
        """Non-list constraints should raise ValueError."""
        with pytest.raises(ValueError, match="must be a list"):
            validate_metric_constraints("latency <= 100")

    def test_non_string_constraint_raises(self):
        """Non-string constraint in list should raise ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            validate_metric_constraints(["latency <= 100", 123])


# ============================================================================
# Unit Tests for MOO Metrics Validation (schema.py)
# ============================================================================


class TestFillValidateMetrics:
    """Unit tests for fill_validate_metrics function."""

    def test_valid_2_objectives(self):
        """Valid config with 2 objectives should pass."""
        config = {
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
                {"name": "latency", "goal": "minimize"},
            ]
        }
        result = fill_validate_metrics(config)
        assert result["metrics"] == config["metrics"]

    def test_valid_3_objectives_with_thresholds(self):
        """Valid config with 3 objectives and thresholds should pass."""
        config = {
            "metrics": [
                {"name": "accuracy", "goal": "maximize", "threshold": 0.9},
                {"name": "latency", "goal": "minimize", "threshold": 100},
                {"name": "model_size", "goal": "minimize", "threshold": 50},
            ]
        }
        result = fill_validate_metrics(config)
        assert len(result["metrics"]) == 3

    def test_single_metric_raises(self):
        """Config with only 1 metric in metrics list should raise."""
        config = {
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
            ]
        }
        with pytest.raises(ValueError, match="at least 2"):
            fill_validate_metrics(config)

    def test_metrics_not_list_raises(self):
        """Non-list metrics should raise ValueError."""
        config = {
            "metrics": {"name": "accuracy", "goal": "maximize"}
        }
        with pytest.raises(ValueError, match="expected list"):
            fill_validate_metrics(config)

    def test_metric_not_dict_raises(self):
        """Non-dict metric in list should raise ValueError."""
        config = {
            "metrics": [
                "accuracy",
                {"name": "latency", "goal": "minimize"},
            ]
        }
        with pytest.raises(ValueError, match="must be a dict"):
            fill_validate_metrics(config)

    def test_missing_name_raises(self):
        """Metric without 'name' should raise ValueError."""
        config = {
            "metrics": [
                {"goal": "maximize"},
                {"name": "latency", "goal": "minimize"},
            ]
        }
        with pytest.raises(ValueError, match="must contain 'name'"):
            fill_validate_metrics(config)

    def test_missing_goal_raises(self):
        """Metric without 'goal' should raise ValueError."""
        config = {
            "metrics": [
                {"name": "accuracy"},
                {"name": "latency", "goal": "minimize"},
            ]
        }
        with pytest.raises(ValueError, match="must contain 'goal'"):
            fill_validate_metrics(config)

    def test_invalid_goal_raises(self):
        """Metric with invalid goal should raise ValueError."""
        config = {
            "metrics": [
                {"name": "accuracy", "goal": "max"},  # Should be "maximize"
                {"name": "latency", "goal": "minimize"},
            ]
        }
        with pytest.raises(ValueError, match="'minimize' or 'maximize'"):
            fill_validate_metrics(config)

    def test_invalid_threshold_type_raises(self):
        """Metric with non-numeric threshold should raise ValueError."""
        config = {
            "metrics": [
                {"name": "accuracy", "goal": "maximize", "threshold": "high"},
                {"name": "latency", "goal": "minimize"},
            ]
        }
        with pytest.raises(ValueError, match="must be a number"):
            fill_validate_metrics(config)

    def test_no_metrics_key_returns_unchanged(self):
        """Config without 'metrics' key should return unchanged."""
        config = {"method": "ax", "parameters": {"lr": {"min": 0.001, "max": 0.1}}}
        result = fill_validate_metrics(config)
        assert result == config


class TestMetricMetricsMutualExclusion:
    """Test mutual exclusion of 'metric' and 'metrics' in fill_validate_schema."""

    def test_both_metric_and_metrics_raises(self):
        """Config with both 'metric' and 'metrics' should raise ValueError.

        Note: We use 'bayes' method here because fill_validate_schema validates
        against the JSON schema first, and 'ax' method is handled separately
        in ax_search.py. The mutual exclusion check applies to all methods.
        """
        config = {
            "method": "bayes",  # Use 'bayes' to pass JSON schema validation
            "parameters": {"lr": {"min": 0.001, "max": 0.1}},
            "metric": {"name": "loss", "goal": "minimize"},
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
                {"name": "latency", "goal": "minimize"},
            ],
        }
        with pytest.raises(ValueError, match="Cannot specify both"):
            fill_validate_schema(config)


# ============================================================================
# Unit Tests for Ax Config Validation (ax_search.py)
# ============================================================================


class TestIsMooConfig:
    """Unit tests for _is_moo_config function."""

    def test_single_objective_returns_false(self):
        """Config with 'metric' should return False."""
        config = {
            "method": "ax",
            "parameters": {"lr": {"min": 0.001, "max": 0.1}},
            "metric": {"name": "loss", "goal": "minimize"},
        }
        assert _is_moo_config(config) is False

    def test_multi_objective_returns_true(self):
        """Config with 'metrics' should return True."""
        config = {
            "method": "ax",
            "parameters": {"lr": {"min": 0.001, "max": 0.1}},
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
                {"name": "latency", "goal": "minimize"},
            ],
        }
        assert _is_moo_config(config) is True


class TestValidateConfig:
    """Unit tests for _validate_config function."""

    def test_valid_single_objective(self, config_ax_single_objective):
        """Valid single-objective config should not raise."""
        # Should not raise
        _validate_config(config_ax_single_objective)

    def test_valid_multi_objective(self, config_ax_moo_2_objectives):
        """Valid multi-objective config should not raise."""
        # Should not raise
        _validate_config(config_ax_moo_2_objectives)

    def test_both_metric_and_metrics_raises(self):
        """Config with both 'metric' and 'metrics' should raise."""
        config = {
            "method": "ax",
            "parameters": {"lr": {"min": 0.001, "max": 0.1}},
            "metric": {"name": "loss", "goal": "minimize"},
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
                {"name": "latency", "goal": "minimize"},
            ],
        }
        with pytest.raises(ValueError, match="Cannot specify both"):
            _validate_config(config)

    def test_neither_metric_nor_metrics_raises(self):
        """Config without 'metric' or 'metrics' should raise."""
        config = {
            "method": "ax",
            "parameters": {"lr": {"min": 0.001, "max": 0.1}},
        }
        with pytest.raises(ValueError, match='requires either "metric"'):
            _validate_config(config)

    def test_early_terminate_with_moo_raises(self):
        """Config with 'early_terminate' and MOO should raise."""
        config = {
            "method": "ax",
            "parameters": {"lr": {"min": 0.001, "max": 0.1}},
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
                {"name": "latency", "goal": "minimize"},
            ],
            "early_terminate": {"type": "hyperband", "min_iter": 3},
        }
        with pytest.raises(ValueError, match="early_terminate is not supported"):
            _validate_config(config)

    def test_early_terminate_with_single_objective_allowed(self):
        """Config with 'early_terminate' and single objective should not raise in validate."""
        # Note: The actual early_terminate functionality with Ax may have other limitations,
        # but _validate_config allows it for single-objective
        config = {
            "method": "ax",
            "parameters": {"lr": {"min": 0.001, "max": 0.1}},
            "metric": {"name": "loss", "goal": "minimize"},
            "early_terminate": {"type": "hyperband", "min_iter": 3},
        }
        # Should not raise for single-objective
        _validate_config(config)

    def test_metrics_with_less_than_2_raises(self):
        """Config with 'metrics' list having < 2 items should raise."""
        config = {
            "method": "ax",
            "parameters": {"lr": {"min": 0.001, "max": 0.1}},
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
            ],
        }
        with pytest.raises(ValueError, match="at least 2"):
            _validate_config(config)

    def test_missing_method_raises(self):
        """Config without 'method' should raise."""
        config = {
            "parameters": {"lr": {"min": 0.001, "max": 0.1}},
            "metric": {"name": "loss", "goal": "minimize"},
        }
        with pytest.raises(ValueError, match="must contain 'method'"):
            _validate_config(config)

    def test_wrong_method_raises(self):
        """Config with method != 'ax' should raise."""
        config = {
            "method": "bayes",
            "parameters": {"lr": {"min": 0.001, "max": 0.1}},
            "metric": {"name": "loss", "goal": "minimize"},
        }
        with pytest.raises(ValueError, match="Expected method='ax'"):
            _validate_config(config)

    def test_missing_parameters_raises(self):
        """Config without 'parameters' should raise."""
        config = {
            "method": "ax",
            "metric": {"name": "loss", "goal": "minimize"},
        }
        with pytest.raises(ValueError, match='requires "parameters"'):
            _validate_config(config)

    def test_empty_parameters_raises(self):
        """Config with empty parameters should raise."""
        config = {
            "method": "ax",
            "parameters": {},
            "metric": {"name": "loss", "goal": "minimize"},
        }
        with pytest.raises(ValueError, match="non-empty dict"):
            _validate_config(config)


# ============================================================================
# Integration Tests for ax_search_next_runs (ax_search.py)
# ============================================================================


class TestMOOBasicIntegration:
    """Integration tests for basic MOO functionality."""

    def test_moo_generates_suggestions_2_objectives(self, config_ax_moo_2_objectives):
        """MOO with 2 objectives should generate valid suggestions."""
        suggestions = ax_search_next_runs([], config_ax_moo_2_objectives, n=3)

        assert len(suggestions) == 3
        for s in suggestions:
            assert "learning_rate" in s.config
            assert "batch_size" in s.config
            assert s.search_info is not None
            assert s.search_info.get("is_multi_objective") is True
            assert s.search_info.get("objective_names") == ["accuracy", "latency"]

    def test_moo_generates_suggestions_3_objectives(
        self, config_ax_moo_3_objectives_with_thresholds
    ):
        """MOO with 3 objectives and thresholds should generate valid suggestions."""
        suggestions = ax_search_next_runs(
            [], config_ax_moo_3_objectives_with_thresholds, n=2
        )

        assert len(suggestions) == 2
        for s in suggestions:
            assert s.search_info.get("is_multi_objective") is True
            assert s.search_info.get("objective_names") == [
                "accuracy",
                "latency",
                "model_size",
            ]

    def test_moo_search_info_metadata(self, config_ax_moo_2_objectives):
        """MOO search_info should contain correct metadata."""
        suggestions = ax_search_next_runs([], config_ax_moo_2_objectives, n=1)

        search_info = suggestions[0].search_info
        assert search_info["method"] == "ax"
        assert search_info["is_multi_objective"] is True
        assert search_info["objective_names"] == ["accuracy", "latency"]
        assert "ax_trial_index" in search_info


class TestMOOWithHistoricalData:
    """Integration tests for MOO with historical data."""

    def test_moo_with_completed_trials(self, config_ax_moo_2_objectives):
        """MOO should handle completed trials correctly."""
        runs = [
            SweepRun(
                state=RunState.finished,
                config={
                    "learning_rate": {"value": 0.01},
                    "batch_size": {"value": 32},
                },
                summary_metrics={"accuracy": 0.85, "latency": 50},
            ),
            SweepRun(
                state=RunState.finished,
                config={
                    "learning_rate": {"value": 0.001},
                    "batch_size": {"value": 64},
                },
                summary_metrics={"accuracy": 0.90, "latency": 80},
            ),
            SweepRun(
                state=RunState.finished,
                config={
                    "learning_rate": {"value": 0.05},
                    "batch_size": {"value": 96},
                },
                summary_metrics={"accuracy": 0.75, "latency": 30},
            ),
        ]

        suggestions = ax_search_next_runs(runs, config_ax_moo_2_objectives, n=2)

        assert len(suggestions) == 2
        assert suggestions[0].search_info["is_multi_objective"] is True

    def test_moo_with_failed_trials(self, config_ax_moo_2_objectives):
        """MOO should handle failed trials correctly."""
        runs = [
            SweepRun(
                state=RunState.finished,
                config={
                    "learning_rate": {"value": 0.01},
                    "batch_size": {"value": 32},
                },
                summary_metrics={"accuracy": 0.85, "latency": 50},
            ),
            SweepRun(
                state=RunState.failed,
                config={
                    "learning_rate": {"value": 0.1},
                    "batch_size": {"value": 16},
                },
                # No summary_metrics for failed run
            ),
            SweepRun(
                state=RunState.finished,
                config={
                    "learning_rate": {"value": 0.001},
                    "batch_size": {"value": 64},
                },
                summary_metrics={"accuracy": 0.90, "latency": 80},
            ),
        ]

        suggestions = ax_search_next_runs(runs, config_ax_moo_2_objectives, n=2)

        assert len(suggestions) == 2

    def test_moo_with_running_trials(self, config_ax_moo_2_objectives):
        """MOO should handle running trials correctly."""
        runs = [
            SweepRun(
                state=RunState.finished,
                config={
                    "learning_rate": {"value": 0.01},
                    "batch_size": {"value": 32},
                },
                summary_metrics={"accuracy": 0.85, "latency": 50},
            ),
            SweepRun(
                state=RunState.running,
                config={
                    "learning_rate": {"value": 0.05},
                    "batch_size": {"value": 48},
                },
                history=[{"accuracy": 0.7, "latency": 60}],
            ),
        ]

        suggestions = ax_search_next_runs(runs, config_ax_moo_2_objectives, n=2)

        assert len(suggestions) == 2


class TestOutcomeConstraints:
    """Integration tests for outcome constraints."""

    def test_single_objective_with_constraints(self, config_ax_single_with_constraints):
        """Single objective with outcome constraints should work."""
        suggestions = ax_search_next_runs([], config_ax_single_with_constraints, n=2)

        assert len(suggestions) == 2
        # Single objective should not have MOO metadata
        assert suggestions[0].search_info.get("is_multi_objective") is None

    def test_moo_with_constraints(self, config_ax_moo_with_constraints):
        """MOO with outcome constraints should work."""
        suggestions = ax_search_next_runs([], config_ax_moo_with_constraints, n=2)

        assert len(suggestions) == 2
        assert suggestions[0].search_info["is_multi_objective"] is True
        assert suggestions[0].search_info["objective_names"] == [
            "accuracy",
            "model_size",
        ]


class TestBackwardCompatibility:
    """Tests for backward compatibility with single-objective optimization."""

    def test_single_metric_still_works(self, config_ax_single_objective):
        """Single 'metric' config should still work (backward compatibility)."""
        suggestions = ax_search_next_runs([], config_ax_single_objective, n=2)

        assert len(suggestions) == 2
        for s in suggestions:
            assert "learning_rate" in s.config
            assert "batch_size" in s.config
            # Single objective should NOT have is_multi_objective in search_info
            assert s.search_info.get("is_multi_objective") is None

    def test_single_metric_with_historical_data(self, config_ax_single_objective):
        """Single objective with historical data should work."""
        runs = [
            SweepRun(
                state=RunState.finished,
                config={
                    "learning_rate": {"value": 0.01},
                    "batch_size": {"value": 32},
                },
                summary_metrics={"loss": 0.5},
            ),
            SweepRun(
                state=RunState.finished,
                config={
                    "learning_rate": {"value": 0.001},
                    "batch_size": {"value": 64},
                },
                summary_metrics={"loss": 0.3},
            ),
        ]

        suggestions = ax_search_next_runs(runs, config_ax_single_objective, n=2)

        assert len(suggestions) == 2


class TestParameterTypes:
    """Tests for different parameter types in MOO."""

    def test_moo_with_categorical_parameters(self):
        """MOO should work with categorical parameters."""
        config = {
            "method": "ax",
            "parameters": {
                "optimizer": {"values": ["adam", "sgd", "rmsprop"]},
                "learning_rate": {"min": 0.001, "max": 0.1},
            },
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
                {"name": "training_time", "goal": "minimize"},
            ],
        }

        suggestions = ax_search_next_runs([], config, n=2)

        assert len(suggestions) == 2
        for s in suggestions:
            assert s.config["optimizer"]["value"] in ["adam", "sgd", "rmsprop"]

    def test_moo_with_log_scale_parameters(self):
        """MOO should work with log-scale parameters."""
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
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
                {"name": "loss", "goal": "minimize"},
            ],
        }

        suggestions = ax_search_next_runs([], config, n=2)

        assert len(suggestions) == 2
        for s in suggestions:
            lr = s.config["learning_rate"]["value"]
            assert 0.0001 <= lr <= 0.1

    def test_moo_with_integer_parameters(self):
        """MOO should work with integer parameters."""
        config = {
            "method": "ax",
            "parameters": {
                "num_layers": {"min": 1, "max": 5},
                "hidden_size": {"min": 64, "max": 512},
            },
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
                {"name": "inference_time", "goal": "minimize"},
            ],
        }

        suggestions = ax_search_next_runs([], config, n=2)

        assert len(suggestions) == 2
        for s in suggestions:
            num_layers = s.config["num_layers"]["value"]
            hidden_size = s.config["hidden_size"]["value"]
            assert isinstance(num_layers, int)
            assert isinstance(hidden_size, int)


class TestValidationErrors:
    """Tests for validation error handling."""

    def test_both_metric_and_metrics_in_ax_search(self):
        """ax_search_next_runs should reject config with both metric and metrics."""
        config = {
            "method": "ax",
            "parameters": {"lr": {"min": 0.001, "max": 0.1}},
            "metric": {"name": "loss", "goal": "minimize"},
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
                {"name": "latency", "goal": "minimize"},
            ],
        }
        with pytest.raises(ValueError, match="Cannot specify both"):
            ax_search_next_runs([], config, n=1)

    def test_metrics_with_one_item_in_ax_search(self):
        """ax_search_next_runs should reject config with < 2 metrics."""
        config = {
            "method": "ax",
            "parameters": {"lr": {"min": 0.001, "max": 0.1}},
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
            ],
        }
        with pytest.raises(ValueError, match="at least 2"):
            ax_search_next_runs([], config, n=1)

    def test_early_terminate_with_moo_in_ax_search(self):
        """ax_search_next_runs should reject config with early_terminate + MOO."""
        config = {
            "method": "ax",
            "parameters": {"lr": {"min": 0.001, "max": 0.1}},
            "metrics": [
                {"name": "accuracy", "goal": "maximize"},
                {"name": "latency", "goal": "minimize"},
            ],
            "early_terminate": {"type": "hyperband", "min_iter": 3},
        }
        with pytest.raises(ValueError, match="early_terminate is not supported"):
            ax_search_next_runs([], config, n=1)
