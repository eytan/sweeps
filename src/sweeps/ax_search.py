"""Bayesian optimization using Meta's ax-platform.

This module provides Bayesian optimization capabilities using the ax-platform
library, which offers more advanced optimization strategies compared to the
sklearn-based implementation in bayes_search.py.

Key Features:
- Leverages Ax's sophisticated trial status management (no manual imputation)
- Failed trials teach the model what parameter regions to avoid
- Running trials contribute as pending observations
- Advanced Bayesian optimization strategies from Meta Research

Supported parameter types:
- Continuous (min/max with uniform distribution)
- Integer (min/max with integer values)
- Categorical (values list)
- Log-scale (log_uniform_values distribution)

Not supported (use method='bayes' instead):
- Normal distributions (normal, q_normal, log_normal, q_log_normal)
- Beta distributions (beta, q_beta)
- Deprecated distributions (log_uniform, inv_log_uniform v1 variants)

Example usage:
    config = {
        'method': 'ax',
        'parameters': {
            'learning_rate': {'min': 0.001, 'max': 0.1, 'distribution': 'log_uniform_values'},
            'batch_size': {'min': 16, 'max': 128},
            'optimizer': {'values': ['adam', 'sgd', 'rmsprop']}
        },
        'metric': {'name': 'val_loss', 'goal': 'minimize'}
    }
    suggestions = ax_search_next_runs([], config, n=5)
"""

import logging
from contextlib import contextmanager
from typing import List, Union

# Sweeps imports
from .config.cfg import SweepConfig
from .params import HyperParameter, HyperParameterSet
from .run import RunState, SweepRun

# Ax imports (with clear error if not installed)
try:
    from ax.api.client import Client
    from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
except ImportError as e:
    raise ImportError(
        "ax method requires ax-platform. " "Install with: pip install sweeps[ax]"
    ) from e

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_ax_logging():
    """Context manager to temporarily suppress Ax client's info-level logging.

    This is used to silence verbose output during bulk trial operations like
    attaching historical trials, which can produce excessive log messages.
    """
    # Get the Ax client logger
    ax_logger = logging.getLogger("ax.api.client")

    # Save current level and set to WARNING
    original_level = ax_logger.level
    ax_logger.setLevel(logging.WARNING)

    try:
        yield
    finally:
        # Restore original level
        ax_logger.setLevel(original_level)


# Map of unsupported parameter types to their display names
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
    """
    Convert a sweeps HyperParameter to an Ax ParameterConfig.

    Args:
        param: HyperParameter to convert

    Returns:
        RangeParameterConfig for continuous/integer parameters,
        ChoiceParameterConfig for categorical parameters,
        or None for constants (which are not searchable)

    Raises:
        ValueError: If parameter type is not supported by ax method
    """
    # Skip constants - they are not part of the search space
    if param.type == HyperParameter.CONSTANT:
        return None

    # Check for unsupported types first
    if param.type in UNSUPPORTED_TYPES:
        raise ValueError(
            f"Parameter '{param.name}' uses distribution '{UNSUPPORTED_TYPES[param.type]}' "
            f"which is not supported by ax method. "
            f"Supported types: uniform, int_uniform, q_uniform, log_uniform_values, categorical. "
            f"Consider using method='bayes' for sklearn-based optimization with these distributions."
        )

    # Continuous parameters
    if param.type == HyperParameter.UNIFORM:
        return RangeParameterConfig(
            name=param.name,
            bounds=(param.config["min"], param.config["max"]),
            parameter_type="float",
        )

    # Integer parameters
    elif param.type == HyperParameter.INT_UNIFORM:
        return RangeParameterConfig(
            name=param.name,
            bounds=(param.config["min"], param.config["max"]),
            parameter_type="int",
        )

    # Quantized uniform (treat as float)
    elif param.type == HyperParameter.Q_UNIFORM:
        return RangeParameterConfig(
            name=param.name,
            bounds=(param.config["min"], param.config["max"]),
            parameter_type="float",
        )

    # Log-scale parameters
    elif param.type in [HyperParameter.LOG_UNIFORM_V2, HyperParameter.Q_LOG_UNIFORM_V2]:
        return RangeParameterConfig(
            name=param.name,
            bounds=(param.config["min"], param.config["max"]),
            parameter_type="float",
            scaling="log",
        )

    # Categorical parameters (both uniform and weighted)
    elif param.type in [HyperParameter.CATEGORICAL, HyperParameter.CATEGORICAL_PROB]:
        # Note: Ax treats all categorical parameters uniformly
        # If probabilities are specified in the config, they are ignored
        # Determine parameter type from first value
        values = param.config["values"]
        if len(values) == 0:
            raise ValueError(f"Parameter '{param.name}' has empty values list")

        # Infer parameter_type from the first value
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
        # Fallback for any other parameter type
        raise ValueError(
            f"Parameter '{param.name}' has unknown or unsupported type '{param.type}'. "
            f"Please use a supported distribution type for ax method."
        )


def _create_ax_client_from_config(
    config: Union[dict, SweepConfig],
    params: HyperParameterSet,
    random_seed: int = 42,
) -> Client:
    """
    Create and configure an Ax Client based on sweep config.

    Args:
        config: Sweep configuration
        params: HyperParameterSet from config["parameters"]

    Returns:
        Configured Client ready for trial generation

    Raises:
        ValueError: If no searchable parameters are found
    """
    # Convert all searchable parameters to Ax configs
    ax_parameters = []
    for param in params.searchable_params:
        ax_param = _convert_parameter_to_ax_config(param)
        if ax_param is not None:  # Skip constants
            ax_parameters.append(ax_param)

    if len(ax_parameters) == 0:
        raise ValueError(
            "No searchable parameters found for Ax optimization. "
            "At least one non-constant parameter is required."
        )

    logger.debug(f"Converted {len(ax_parameters)} parameters to Ax format")

    # Create client
    client = Client(random_seed=random_seed)

    # Configure experiment
    client.configure_experiment(
        parameters=ax_parameters,
        name=config.get("name", "sweep"),
        description=config.get("description"),
    )

    # Configure optimization
    metric_name = config["metric"]["name"]
    goal = config["metric"]["goal"]

    # Ax uses objective string format:
    # - "-metric_name" for minimization (negative to convert max to min)
    # - "metric_name" for maximization
    if goal == "minimize":
        objective = f"-{metric_name}"
    else:  # maximize
        objective = metric_name

    client.configure_optimization(objective=objective)

    # Configure generation strategy (default to "fast")
    # "fast" uses Bayesian optimization with good defaults
    client.configure_generation_strategy(
        method="fast",
        initialization_random_seed=random_seed,
        initialize_with_center=False,
    )

    logger.debug(f"Configured Ax experiment with objective: {objective}")

    return client


def _extract_parameters_from_run(run: SweepRun, params: HyperParameterSet) -> dict:
    """
    Extract parameter values from a run's config.

    Args:
        run: SweepRun with config
        params: HyperParameterSet for parameter mapping

    Returns:
        Dict mapping parameter names to values
        Example: {"learning_rate": 0.001, "batch_size": 32}

    Raises:
        ValueError: If required parameters are missing from run config
    """
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
    """
    Extract the final metric value from a completed run.

    Uses summary_metric to get the final (summary) metric value.

    Args:
        run: SweepRun with metric data
        metric_name: Name of the metric

    Returns:
        Metric value (float)

    Raises:
        ValueError: If metric cannot be extracted
    """
    try:
        return run.summary_metric(metric_name)
    except (KeyError, ValueError) as e:
        raise ValueError(f"Cannot extract metric '{metric_name}' from run: {e}")


def _extract_latest_metric_from_run(run: SweepRun, metric_name: str) -> float:
    """
    Extract the latest metric value from a running trial.

    Args:
        run: SweepRun with history
        metric_name: Name of the metric

    Returns:
        Latest metric value (float)

    Raises:
        ValueError: If no metric data available
    """
    history = run.metric_history(metric_name, filter_invalid=True)

    if len(history) == 0:
        raise ValueError(f"No metric data available for '{metric_name}' in run history")

    return history[-1]


def _attach_historical_trials_to_client(
    client: Client,
    runs: List[SweepRun],
    params: HyperParameterSet,
    metric_name: str,
    goal: str,
) -> dict:
    """
    Attach historical trial data from runs to the Ax Client using native Ax trial statuses.

    This function leverages Ax's sophisticated trial status management:
    - COMPLETED: Trials with valid metric data that Ax learns from
    - FAILED: Crashed/failed runs that teach Ax what regions to avoid
    - RUNNING: Currently executing trials treated as pending observations
    - ABANDONED: Trials that were stopped before completion

    No manual imputation is needed - Ax handles all trial states natively.

    Trial Status Mapping:
    - RunState.finished → attach_data + complete_trial (auto COMPLETED/FAILED)
    - RunState.failed/crashed/killed → mark_trial_failed()
    - RunState.running → attach_data if available, else left as RUNNING
    - RunState.pending/preempting/preempted → mark_trial_abandoned()

    Args:
        client: Configured Ax Client
        runs: Historical sweep runs
        params: HyperParameterSet for parameter extraction
        metric_name: Name of optimization metric
        goal: "minimize" or "maximize"

    Returns:
        Dict with statistics: {
            "completed": int,
            "failed": int,
            "running": int,
            "abandoned": int,
            "skipped": int
        }
    """
    stats = {
        "completed": 0,
        "failed": 0,
        "running": 0,
        "abandoned": 0,
        "skipped": 0,
    }

    for run in runs:
        # Extract parameters from run config
        try:
            parameters = _extract_parameters_from_run(run, params)
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping run with invalid parameters: {e}")
            stats["skipped"] += 1
            continue

        # Attach trial to Ax experiment to get trial_index
        trial_index = client.attach_trial(parameters=parameters)

        # Handle based on run state using Ax-native trial status
        if run.state == RunState.finished:
            # Finished run - try to extract metric and complete
            try:
                metric_value = _extract_metric_from_run(run, metric_name)

                # Attach data and complete trial
                # complete_trial will automatically mark as COMPLETED or FAILED
                # based on whether metric data is present
                client.complete_trial(
                    trial_index=trial_index, raw_data={metric_name: metric_value}
                )
                stats["completed"] += 1

            except ValueError as e:
                # No valid metric - mark as failed
                logger.warning(
                    f"Trial {trial_index} missing metric, marking as FAILED: {e}"
                )
                client.mark_trial_failed(
                    trial_index=trial_index, failed_reason=f"Missing metric: {e}"
                )
                stats["failed"] += 1

        elif run.state in [RunState.failed, RunState.crashed, RunState.killed]:
            # Failed/crashed run - mark as FAILED
            # Ax will learn to avoid similar parameter configurations
            client.mark_trial_failed(
                trial_index=trial_index, failed_reason=f"Run {run.state}"
            )
            stats["failed"] += 1

        elif run.state == RunState.running:
            # Running trial - try to attach partial data if available
            try:
                # Get latest metric value if available
                metric_value = _extract_latest_metric_from_run(run, metric_name)
                client.attach_data(
                    trial_index=trial_index, raw_data={metric_name: metric_value}
                )
                logger.debug(f"Attached partial data for running trial {trial_index}")
            except ValueError:
                # No data yet - leave trial as RUNNING
                # Ax will treat this as a pending observation
                logger.debug(f"Trial {trial_index} is running but has no data yet")

            stats["running"] += 1

        else:  # pending, preempting, preempted
            # These trials haven't started or were preempted - mark as abandoned
            # Ax will exclude them from optimization
            client.mark_trial_abandoned(trial_index=trial_index)
            stats["abandoned"] += 1

    logger.debug(
        f"Attached trials - Completed: {stats['completed']}, "
        f"Failed: {stats['failed']}, Running: {stats['running']}, "
        f"Abandoned: {stats['abandoned']}, Skipped: {stats['skipped']}"
    )

    return stats


def _ax_params_to_sweep_config(ax_params: dict, params: HyperParameterSet) -> dict:
    """
    Convert Ax parameterization dict to sweeps config format.

    Ax provides: {"param1": 0.5, "param2": 10, "param3": "choice_a"}
    Sweeps expects: {"param1": {"value": 0.5}, "param2": {"value": 10}, ...}

    Also handles:
    - Type conversions (ensure int params are ints)
    - Nested parameters (using NESTING_DELIMITER)
    - Constants (added from original params)

    Args:
        ax_params: Dict from Ax (parameter name -> value)
        params: Original HyperParameterSet for type info and constants

    Returns:
        Sweeps config dict format with "value" keys
    """
    # Set parameter values from Ax suggestion
    for param in params:
        if param.type == HyperParameter.CONSTANT:
            # Constants keep their original value
            continue
        elif param.name in ax_params:
            # Get value from Ax suggestion
            value = ax_params[param.name]

            # Type conversion for integer parameters
            if param.type == HyperParameter.INT_UNIFORM:
                value = int(value)

            # Set the value on the parameter object
            param.value = value
        else:
            logger.warning(
                f"Parameter '{param.name}' not in Ax suggestion, "
                f"using default or previous value"
            )

    # Convert to sweeps config format (with nested "value" keys)
    return params.to_config()


def _validate_config(config: dict) -> None:
    """Validate sweep config for ax method.

    Checks:
    - method == "ax"
    - "metric" section exists with "name" and "goal"
    - "parameters" section exists and is non-empty

    Args:
        config: Sweep configuration dict

    Raises:
        ValueError: With descriptive message if validation fails
    """
    if "method" not in config:
        raise ValueError("Sweep config must contain 'method' section")

    if config["method"] != "ax":
        raise ValueError(
            f"Invalid sweep configuration for Ax search. "
            f"Expected method='ax', got method='{config['method']}'"
        )

    if "metric" not in config:
        raise ValueError('Ax Bayesian search requires "metric" section in config')

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
    """
    Suggest runs using ax-platform Bayesian optimization.

    This implementation fully leverages Ax's native capabilities:
    - Sophisticated trial status management (COMPLETED, FAILED, RUNNING, ABANDONED)
    - No manual imputation needed - Ax handles incomplete data natively
    - Failed trials teach the model what parameter regions to avoid
    - Running trials contribute as pending observations
    - Advanced Bayesian optimization strategies from Meta Research

    Main workflow:
    1. Validate config and extract parameters
    2. Create Ax Client with experiment configuration
    3. Attach historical trial data with native Ax status handling
    4. Generate n new trials using Ax's optimization
    5. Convert Ax parameterizations to SweepRun format
    6. Return list of SweepRun objects

    Args:
        runs: List of existing runs in the sweep
        config: Sweep configuration dict or SweepConfig
        validate: Whether to validate config against schema
        n: Number of new runs to generate
        **kwargs: Additional arguments (reserved for future extensions)

    Returns:
        List of n SweepRun objects with suggested configurations

    Raises:
        ValueError: For invalid config or unsupported parameter types
        ImportError: If ax-platform is not installed
        RuntimeError: If Ax optimization fails

    Example:
        >>> config = {
        ...     'method': 'ax',
        ...     'parameters': {
        ...         'learning_rate': {'min': 0.001, 'max': 0.1, 'distribution': 'log_uniform_values'},
        ...         'batch_size': {'min': 16, 'max': 128}
        ...     },
        ...     'metric': {'name': 'loss', 'goal': 'minimize'}
        ... }
        >>> suggestions = ax_search_next_runs([], config, n=5)
        >>> len(suggestions)
        5
    """
    # 1. Validation
    if validate:
        config = SweepConfig(config)

    _validate_config(config)

    # 2. Extract configuration
    params = HyperParameterSet.from_config(config["parameters"])
    metric_name = config["metric"]["name"]
    goal = config["metric"]["goal"]

    if len(params.searchable_params) == 0:
        raise ValueError(
            "No searchable parameters found. "
            "At least one non-constant parameter is required for Ax."
        )

    logger.info(
        f"Starting Ax optimization for {len(params.searchable_params)} searchable parameters, "
        f"{len(runs)} historical runs"
    )

    # 3. Create Ax Client
    try:
        client = _create_ax_client_from_config(config, params, random_seed=random_seed)
    except Exception as e:
        raise RuntimeError(f"Failed to create Ax client: {e}") from e

    # 4. Attach historical data using Ax-native trial status
    if len(runs) > 0:
        # Suppress Ax's verbose info logging during bulk trial operations
        with _suppress_ax_logging():
            stats = _attach_historical_trials_to_client(
                client, runs, params, metric_name, goal
            )
        logger.info(f"Trial attachment statistics: {stats}")
    else:
        logger.info("No historical runs - Ax will use initialization strategy")

    # 5. Generate new trials
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

    # 6. Convert to SweepRun format
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
