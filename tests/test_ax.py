"""Integration tests for Ax Bayesian Optimization."""

import os
import pytest
import yaml
from sweeps.ax_search import ax_search_next_runs
from sweeps.config import SweepConfig
from sweeps.run import RunState, SweepRun


# Path to example YAML configs
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
BENCHMARK_CONFIGS_DIR = os.path.join(EXAMPLES_DIR, "ax-benchmarks", "configs")


class TestYAMLSerialization:
    """Tests for YAML config loading and serialization."""

    def test_load_soo_yaml_config(self):
        """Load SOO config from YAML and run optimization."""
        yaml_path = os.path.join(EXAMPLES_DIR, "sweep-ax.yaml")

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        # Validate through SweepConfig
        validated_config = SweepConfig(config)

        assert validated_config["method"] == "ax"
        assert "metric" in validated_config
        assert validated_config["metric"]["name"] == "val_loss"

        # Run optimization with validated config
        suggestions = ax_search_next_runs([], validated_config, n=2)

        assert len(suggestions) == 2
        # Check that parameters from YAML are present
        assert "learning_rate" in suggestions[0].config
        assert "batch_size" in suggestions[0].config

    def test_load_moo_yaml_config(self):
        """Load MOO config from YAML and run optimization."""
        yaml_path = os.path.join(EXAMPLES_DIR, "sweep-ax-moo.yaml")

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        # Validate through SweepConfig
        validated_config = SweepConfig(config)

        assert validated_config["method"] == "ax"
        assert "metrics" in validated_config
        assert len(validated_config["metrics"]) >= 2

        # Run optimization with validated config
        suggestions = ax_search_next_runs([], validated_config, n=2)

        assert len(suggestions) == 2
        assert suggestions[0].search_info.get("is_multi_objective") is True

    def test_load_dtlz2_benchmark_config(self):
        """Load DTLZ2 benchmark config from YAML."""
        yaml_path = os.path.join(BENCHMARK_CONFIGS_DIR, "ax_dtlz2_moo.yaml")

        if not os.path.exists(yaml_path):
            pytest.skip("Benchmark config not found")

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        # Validate through SweepConfig
        validated_config = SweepConfig(config)

        assert validated_config["method"] == "ax"
        assert "metrics" in validated_config
        assert len(validated_config["metrics"]) == 2

        # Check parameter bounds
        assert "x0" in validated_config["parameters"]

        # Run optimization
        suggestions = ax_search_next_runs([], validated_config, n=2)

        assert len(suggestions) == 2
        assert suggestions[0].search_info.get("is_multi_objective") is True

    def test_load_constrained_moo_config(self):
        """Load constrained MOO config from YAML."""
        yaml_path = os.path.join(BENCHMARK_CONFIGS_DIR, "ax_c2dtlz2_moo.yaml")

        if not os.path.exists(yaml_path):
            pytest.skip("Benchmark config not found")

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        # Validate through SweepConfig
        validated_config = SweepConfig(config)

        assert validated_config["method"] == "ax"
        assert "metrics" in validated_config
        assert "metric_constraints" in validated_config
        assert len(validated_config["metric_constraints"]) >= 1

        # Run optimization
        suggestions = ax_search_next_runs([], validated_config, n=2)

        assert len(suggestions) == 2

    def test_yaml_roundtrip(self):
        """Test that config survives YAML serialization roundtrip."""
        original_config = {
            "method": "ax",
            "name": "test-sweep",
            "parameters": {
                "learning_rate": {
                    "min": 0.0001,
                    "max": 0.1,
                    "distribution": "log_uniform_values",
                },
                "batch_size": {"min": 16, "max": 128},
                "optimizer": {"values": ["adam", "sgd"]},
            },
            "metrics": [
                {"name": "accuracy", "goal": "maximize", "threshold": 0.95},
                {"name": "latency", "goal": "minimize", "threshold": 100},
            ],
            "metric_constraints": ["memory <= 512"],
        }

        # Serialize to YAML string
        yaml_str = yaml.dump(original_config)

        # Deserialize back
        loaded_config = yaml.safe_load(yaml_str)

        # Validate through SweepConfig
        validated_config = SweepConfig(loaded_config)

        # Run optimization
        suggestions = ax_search_next_runs([], validated_config, n=2)

        assert len(suggestions) == 2
        assert suggestions[0].search_info.get("is_multi_objective") is True
        assert "learning_rate" in suggestions[0].config
        assert "batch_size" in suggestions[0].config
        assert "optimizer" in suggestions[0].config


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
