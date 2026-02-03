"""Ax Bayesian optimization integration for W&B Sweeps.

Provides single-objective optimization via Ax platform.
Install with: pip install sweeps[ax]
"""

import logging
from contextlib import contextmanager
from typing import List, Union

from .config.cfg import SweepConfig
from .params import HyperParameter, HyperParameterSet
from .run import RunState, SweepRun

try:
    from ax.api.client import Client
    from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
    AX_AVAILABLE = True
except ImportError:
    Client = None
    ChoiceParameterConfig = None
    RangeParameterConfig = None
    AX_AVAILABLE = False

logger = logging.getLogger(__name__)


def _check_ax_available():
    """Raise ImportError with helpful message if ax-platform is not installed."""
    if not AX_AVAILABLE:
        raise ImportError(
            "ax method requires ax-platform. Install with: pip install sweeps[ax]"
        )


@contextmanager
def _suppress_ax_logging():
    """Temporarily suppress Ax client's info-level logging during bulk operations."""
    ax_logger = logging.getLogger("ax.api.client")
    original_level = ax_logger.level
    ax_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        ax_logger.setLevel(original_level)


UNSUPPORTED_TYPES = {
    HyperParameter.NORMAL: "normal",
    HyperParameter.Q_NORMAL: "q_normal",
    HyperParameter.LOG_NORMAL: "log_normal",
    HyperParameter.Q_LOG_NORMAL: "q_log_normal",
    HyperParameter.BETA: "beta",
    HyperParameter.Q_BETA: "q_beta",
    HyperParameter.INV_LOG_UNIFORM_V1: "inv_log_uniform (deprecated)",
    HyperParameter.INV_LOG_UNIFORM_V2: "inv_log_uniform",
    HyperParameter.LOG_UNIFORM_V1: "log_uniform (deprecated)",
    HyperParameter.Q_LOG_UNIFORM_V1: "q_log_uniform (deprecated)",
}


def _convert_parameter_to_ax_config(
    param: HyperParameter,
) -> Union[RangeParameterConfig, ChoiceParameterConfig, None]:
    """Convert a sweeps HyperParameter to an Ax ParameterConfig."""
    if param.type == HyperParameter.CONSTANT:
        return None

    if param.type in UNSUPPORTED_TYPES:
        raise ValueError(
            f"Parameter '{param.name}' uses distribution '{UNSUPPORTED_TYPES[param.type]}' "
            f"which is not supported by ax method. "
            f"Supported types: uniform, int_uniform, q_uniform, log_uniform_values, categorical. "
            f"Consider using method='bayes' for sklearn-based optimization with these distributions."
        )

    if param.type == HyperParameter.UNIFORM:
        return RangeParameterConfig(
            name=param.name,
            bounds=(param.config["min"], param.config["max"]),
            parameter_type="float",
        )

    elif param.type == HyperParameter.INT_UNIFORM:
        return RangeParameterConfig(
            name=param.name,
            bounds=(param.config["min"], param.config["max"]),
            parameter_type="int",
        )

    elif param.type == HyperParameter.Q_UNIFORM:
        return RangeParameterConfig(
            name=param.name,
            bounds=(param.config["min"], param.config["max"]),
            parameter_type="float",
        )

    elif param.type in [HyperParameter.LOG_UNIFORM_V2, HyperParameter.Q_LOG_UNIFORM_V2]:
        return RangeParameterConfig(
            name=param.name,
            bounds=(param.config["min"], param.config["max"]),
            parameter_type="float",
            scaling="log",
        )

    elif param.type in [HyperParameter.CATEGORICAL, HyperParameter.CATEGORICAL_PROB]:
        values = param.config["values"]
        if len(values) == 0:
            raise ValueError(f"Parameter '{param.name}' has empty values list")

        first_value = values[0]
        if isinstance(first_value, bool):
            param_type = "bool"
        elif isinstance(first_value, int):
            param_type = "int"
        elif isinstance(first_value, float):
            param_type = "float"
        elif isinstance(first_value, str):
            param_type = "str"
        else:
            param_type = "str"  # Default to string

        return ChoiceParameterConfig(
            name=param.name,
            values=values,
            parameter_type=param_type,
        )

    else:
        raise ValueError(
            f"Parameter '{param.name}' has unknown or unsupported type '{param.type}'. "
            f"Please use a supported distribution type for ax method."
        )


def _create_ax_client_from_config(
    config: Union[dict, SweepConfig],
    params: HyperParameterSet,
    random_seed: int = 42,
) -> Client:
    """Create and configure an Ax Client based on sweep config."""
    ax_parameters = []
    for param in params.searchable_params:
        ax_param = _convert_parameter_to_ax_config(param)
        if ax_param is not None:
            ax_parameters.append(ax_param)

    if len(ax_parameters) == 0:
        raise ValueError(
            "No searchable parameters found for Ax optimization. "
            "At least one non-constant parameter is required."
        )

    logger.debug(f"Converted {len(ax_parameters)} parameters to Ax format")

    client = Client(random_seed=random_seed)
    client.configure_experiment(
        parameters=ax_parameters,
        name=config.get("name", "sweep"),
        description=config.get("description"),
    )

    # Single-objective optimization
    metric_name = config["metric"]["name"]
    goal = config["metric"]["goal"]
    if goal == "minimize":
        objective = f"-{metric_name}"
    else:
        objective = metric_name

    client.configure_optimization(
        objective=objective,
    )

    logger.debug(f"Configured Ax experiment with objective: {objective}")

    client.configure_generation_strategy(
        method="fast",
        initialization_random_seed=random_seed,
        initialize_with_center=False,
    )

    return client


def _extract_parameters_from_run(run: SweepRun, params: HyperParameterSet) -> dict:
    """Extract parameter values from a run's config."""
    parameters = {}

    for param in params.searchable_params:
        value = params._get_val_from_config(run.config, param.name)

        if value is None:
            raise ValueError(
                f"Parameter '{param.name}' not found in run config. "
                f"Run may be from a different sweep configuration."
            )

        parameters[param.name] = value

    return parameters


def _extract_metric_from_run(run: SweepRun, metric_name: str) -> float:
    """Extract the final metric value from a completed run."""
    try:
        return run.summary_metric(metric_name)
    except (KeyError, ValueError) as e:
        raise ValueError(f"Cannot extract metric '{metric_name}' from run: {e}")


def _extract_latest_metric_from_run(run: SweepRun, metric_name: str) -> float:
    """Extract the latest metric value from a running trial."""
    history = run.metric_history(metric_name, filter_invalid=True)

    if len(history) == 0:
        raise ValueError(f"No metric data available for '{metric_name}' in run history")

    return history[-1]


def _attach_historical_trials_to_client(
    client: Client,
    runs: List[SweepRun],
    params: HyperParameterSet,
    metric_name: str,
) -> dict:
    """Attach historical trial data from runs to the Ax Client.

    Maps RunState to Ax trial status (COMPLETED, FAILED, RUNNING, ABANDONED).
    Returns dict with counts: completed, failed, running, abandoned, skipped.
    """
    stats = {
        "completed": 0,
        "failed": 0,
        "running": 0,
        "abandoned": 0,
        "skipped": 0,
    }

    for run in runs:
        try:
            parameters = _extract_parameters_from_run(run, params)
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping run with invalid parameters: {e}")
            stats["skipped"] += 1
            continue

        trial_index = client.attach_trial(parameters=parameters)

        if run.state == RunState.finished:
            try:
                metric_value = _extract_metric_from_run(run, metric_name)
                metric_values = {metric_name: metric_value}

                client.complete_trial(trial_index=trial_index, raw_data=metric_values)
                stats["completed"] += 1

            except ValueError as e:
                logger.warning(
                    f"Trial {trial_index} missing metric(s), marking as FAILED: {e}"
                )
                client.mark_trial_failed(
                    trial_index=trial_index, failed_reason=f"Missing metric: {e}"
                )
                stats["failed"] += 1

        elif run.state in [RunState.failed, RunState.crashed, RunState.killed]:
            client.mark_trial_failed(
                trial_index=trial_index, failed_reason=f"Run {run.state}"
            )
            stats["failed"] += 1

        elif run.state == RunState.running:
            try:
                metric_value = _extract_latest_metric_from_run(run, metric_name)
                client.attach_data(
                    trial_index=trial_index,
                    raw_data={metric_name: metric_value},
                )
                logger.debug(
                    f"Attached partial data for running trial {trial_index}"
                )
            except ValueError:
                logger.debug(f"Trial {trial_index} is running but has no data yet")

            stats["running"] += 1

        else:
            client.mark_trial_abandoned(trial_index=trial_index)
            stats["abandoned"] += 1

    logger.debug(
        f"Attached trials - Completed: {stats['completed']}, "
        f"Failed: {stats['failed']}, Running: {stats['running']}, "
        f"Abandoned: {stats['abandoned']}, Skipped: {stats['skipped']}"
    )

    return stats


def _ax_params_to_sweep_config(ax_params: dict, params: HyperParameterSet) -> dict:
    """Convert Ax parameterization dict to sweeps config format."""
    for param in params:
        if param.type == HyperParameter.CONSTANT:
            continue
        elif param.name in ax_params:
            value = ax_params[param.name]
            if param.type == HyperParameter.INT_UNIFORM:
                value = int(value)
            param.value = value
        else:
            logger.warning(
                f"Parameter '{param.name}' not in Ax suggestion, "
                f"using default or previous value"
            )

    return params.to_config()


def _validate_config(config: dict) -> None:
    """Validate sweep config for ax method."""
    if "method" not in config:
        raise ValueError("Sweep config must contain 'method' section")

    if config["method"] != "ax":
        raise ValueError(
            f"Invalid sweep configuration for Ax search. "
            f"Expected method='ax', got method='{config['method']}'"
        )

    if "metric" not in config:
        raise ValueError(
            'Ax Bayesian search requires "metric" section in config'
        )

    if "name" not in config["metric"]:
        raise ValueError('Metric section must contain "name" field')

    if "goal" not in config["metric"]:
        raise ValueError(
            'Metric section must contain "goal" field (minimize or maximize)'
        )

    if config["metric"]["goal"] not in ["minimize", "maximize"]:
        raise ValueError(
            f"Metric goal must be 'minimize' or 'maximize', "
            f"got '{config['metric']['goal']}'"
        )

    if "parameters" not in config:
        raise ValueError('Ax Bayesian search requires "parameters" section in config')

    if not isinstance(config["parameters"], dict) or len(config["parameters"]) == 0:
        raise ValueError("Parameters section must be a non-empty dict")


def ax_search_next_runs(
    runs: List[SweepRun],
    config: Union[dict, SweepConfig],
    validate: bool = False,
    n: int = 1,
    random_seed: int = 42,
    **kwargs,
) -> List[SweepRun]:
    """Suggest runs using ax-platform Bayesian optimization.

    Supports single-objective optimization using the metric config.
    Handles trial status natively: completed, failed, running trials are all used.

    Args:
        runs: List of existing runs in the sweep
        config: Sweep configuration dict or SweepConfig
        validate: Whether to validate config against schema
        n: Number of new runs to generate
        random_seed: Random seed for reproducibility

    Returns:
        List of n SweepRun objects with suggested configurations.

    Raises:
        ValueError: For invalid config or unsupported parameter types
        RuntimeError: If Ax optimization fails
    """
    _check_ax_available()

    if validate:
        config = SweepConfig(config)

    _validate_config(config)

    params = HyperParameterSet.from_config(config["parameters"])
    metric_name = config["metric"]["name"]

    if len(params.searchable_params) == 0:
        raise ValueError(
            "No searchable parameters found. "
            "At least one non-constant parameter is required for Ax."
        )

    logger.info(
        f"Starting Ax single-objective optimization for {len(params.searchable_params)} "
        f"searchable parameters, {len(runs)} historical runs"
    )

    try:
        client = _create_ax_client_from_config(config, params, random_seed=random_seed)
    except Exception as e:
        raise RuntimeError(f"Failed to create Ax client: {e}") from e

    if len(runs) > 0:
        with _suppress_ax_logging():
            stats = _attach_historical_trials_to_client(
                client, runs, params, metric_name
            )
        logger.info(f"Trial attachment statistics: {stats}")
    else:
        logger.info("No historical runs - Ax will use initialization strategy")

    logger.info(f"Generating {n} new trial suggestions")

    try:
        next_parameterizations = client.get_next_trials(max_trials=n)
    except Exception as e:
        raise RuntimeError(
            f"Ax optimization failed: {e}. "
            "This may occur if the search space is fully explored, "
            "if there are no valid trials to learn from, or "
            "if the optimization cannot generate valid candidates. "
            "Consider checking your parameter ranges and metric data."
        ) from e

    suggested_runs = []

    for trial_index, ax_params in next_parameterizations.items():
        # Convert Ax parameterization to sweeps config format
        sweep_config = _ax_params_to_sweep_config(ax_params, params)

        # Add search metadata
        search_info = {
            "method": "ax",
            "ax_trial_index": trial_index,
        }

        suggested_runs.append(SweepRun(config=sweep_config, search_info=search_info))

    logger.info(f"Successfully generated {len(suggested_runs)} trial suggestions")

    return suggested_runs
