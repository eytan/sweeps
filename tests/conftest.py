import pytest
from sweeps.config import SweepConfig


def pytest_addoption(parser):
    parser.addoption(
        "--plot",
        action="store_true",
        help="Plot true and predicted distributions "
        "for tests involving random sampling.",
    )


@pytest.fixture
def plot(request):
    return request.config.getoption("--plot")


@pytest.fixture()
def sweep_config_bayes_search_2params_with_metric():
    return SweepConfig(
        {
            "metric": {"name": "loss", "goal": "minimize"},
            "method": "bayes",
            "parameters": {
                "v1": {"min": 1, "max": 10},
                "v2": {"min": 1.0, "max": 10.0},
            },
        }
    )


@pytest.fixture()
def sweep_config_2params_grid_search():
    return SweepConfig(
        {
            "method": "grid",
            "parameters": {"v1": {"values": [1, 2, 3]}, "v2": {"values": [4, 5]}},
        }
    )


# ============================================================================
# Ax MOO (Multi-Objective Optimization) Fixtures
# ============================================================================


@pytest.fixture()
def config_ax_single_objective():
    """Basic Ax config with single objective (backward compatibility)."""
    return {
        "method": "ax",
        "parameters": {
            "learning_rate": {"min": 0.001, "max": 0.1},
            "batch_size": {"min": 16, "max": 128},
        },
        "metric": {"name": "loss", "goal": "minimize"},
    }


@pytest.fixture()
def config_ax_moo_2_objectives():
    """Ax config with 2 objectives for multi-objective optimization."""
    return {
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


@pytest.fixture()
def config_ax_moo_3_objectives_with_thresholds():
    """Ax config with 3 objectives and threshold values."""
    return {
        "method": "ax",
        "parameters": {
            "learning_rate": {"min": 0.001, "max": 0.1},
            "batch_size": {"min": 16, "max": 128},
        },
        "metrics": [
            {"name": "accuracy", "goal": "maximize", "threshold": 0.94},
            {"name": "latency", "goal": "minimize", "threshold": 100},
            {"name": "model_size", "goal": "minimize", "threshold": 50},
        ],
    }


@pytest.fixture()
def config_ax_single_with_constraints():
    """Ax config with single objective and outcome constraints."""
    return {
        "method": "ax",
        "parameters": {
            "learning_rate": {"min": 0.001, "max": 0.1},
            "batch_size": {"min": 16, "max": 128},
        },
        "metric": {"name": "accuracy", "goal": "maximize"},
        "metric_constraints": [
            "latency <= 100",
            "memory_usage <= 512",
        ],
    }


@pytest.fixture()
def config_ax_moo_with_constraints():
    """Ax config with MOO and outcome constraints."""
    return {
        "method": "ax",
        "parameters": {
            "learning_rate": {"min": 0.001, "max": 0.1},
            "batch_size": {"min": 16, "max": 128},
        },
        "metrics": [
            {"name": "accuracy", "goal": "maximize"},
            {"name": "model_size", "goal": "minimize"},
        ],
        "metric_constraints": [
            "inference_latency <= 50",
        ],
    }
