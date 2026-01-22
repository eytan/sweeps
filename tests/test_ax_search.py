#!/usr/bin/env python3
"""Test script for ax_search implementation."""

from sweeps.ax_search import ax_search_next_runs
from sweeps.run import RunState, SweepRun

print("=" * 60)
print("Testing ax_search implementation")
print("=" * 60)
print()

# Test 1: Basic usage with empty sweep
print("Test 1: Basic usage with empty sweep")
print("-" * 60)
config = {
    "method": "ax",
    "parameters": {
        "learning_rate": {"min": 0.001, "max": 0.1},
        "batch_size": {"min": 16, "max": 128},
    },
    "metric": {"name": "loss", "goal": "minimize"},
}

try:
    suggestions = ax_search_next_runs([], config, n=3)
    print(f"✓ Generated {len(suggestions)} suggestions")
    for i, s in enumerate(suggestions):
        lr = s.config["learning_rate"]["value"]
        bs = s.config["batch_size"]["value"]
        print(f"  Suggestion {i+1}: lr={lr:.4f}, batch_size={bs}")
    print(f"  Search info: {suggestions[0].search_info}")
    print()
except Exception as e:
    import traceback

    print(f"✗ Error: {e}")
    traceback.print_exc()
    print()

# Test 2: With historical data
print("Test 2: With completed trials")
print("-" * 60)
runs = [
    SweepRun(
        state=RunState.finished,
        config={"learning_rate": {"value": 0.01}, "batch_size": {"value": 32}},
        summary_metrics={"loss": 0.5},
    ),
    SweepRun(
        state=RunState.finished,
        config={"learning_rate": {"value": 0.001}, "batch_size": {"value": 64}},
        summary_metrics={"loss": 0.3},
    ),
    SweepRun(
        state=RunState.finished,
        config={"learning_rate": {"value": 0.05}, "batch_size": {"value": 96}},
        summary_metrics={"loss": 0.8},
    ),
]

try:
    suggestions = ax_search_next_runs(runs, config, n=2)
    print(
        f"✓ Generated {len(suggestions)} suggestions with {len(runs)} historical trials"
    )
    for i, s in enumerate(suggestions):
        lr = s.config["learning_rate"]["value"]
        bs = s.config["batch_size"]["value"]
        print(f"  Suggestion {i+1}: lr={lr:.4f}, batch_size={bs}")
    print()
except Exception as e:
    import traceback

    print(f"✗ Error: {e}")
    traceback.print_exc()
    print()

# Test 3: Categorical parameters
print("Test 3: Categorical parameters")
print("-" * 60)
config_cat = {
    "method": "ax",
    "parameters": {
        "optimizer": {"values": ["adam", "sgd", "rmsprop"]},
        "lr": {"min": 0.001, "max": 0.1},
    },
    "metric": {"name": "acc", "goal": "maximize"},
}

try:
    suggestions = ax_search_next_runs([], config_cat, n=2)
    print(f"✓ Generated {len(suggestions)} suggestions with categorical params")
    for i, s in enumerate(suggestions):
        opt = s.config["optimizer"]["value"]
        lr = s.config["lr"]["value"]
        print(f"  Suggestion {i+1}: optimizer={opt}, lr={lr:.4f}")
    print()
except Exception as e:
    import traceback

    print(f"✗ Error: {e}")
    traceback.print_exc()
    print()

# Test 4: Log-scale parameters
print("Test 4: Log-scale parameters")
print("-" * 60)
config_log = {
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

try:
    suggestions = ax_search_next_runs([], config_log, n=3)
    print(f"✓ Generated {len(suggestions)} suggestions with log-scale params")
    for i, s in enumerate(suggestions):
        lr = s.config["learning_rate"]["value"]
        wd = s.config["weight_decay"]["value"]
        print(f"  Suggestion {i+1}: lr={lr:.6f}, weight_decay={wd:.6f}")
    print()
except Exception as e:
    import traceback

    print(f"✗ Error: {e}")
    traceback.print_exc()
    print()

# Test 5: Failed trials (Ax should learn to avoid)
# Test 5: With failed trials
print("Test 5: With failed trials")
print("-" * 60)
runs_with_failures = [
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

try:
    suggestions = ax_search_next_runs(runs_with_failures, config, n=2)
    print(
        f"✓ Generated {len(suggestions)} suggestions (Ax learned from 1 failed trial)"
    )
    for i, s in enumerate(suggestions):
        lr = s.config["learning_rate"]["value"]
        bs = s.config["batch_size"]["value"]
        print(f"  Suggestion {i+1}: lr={lr:.4f}, batch_size={bs}")
    print()
except Exception as e:
    import traceback

    print(f"✗ Error: {e}")
    traceback.print_exc()
    print()

# Test 6: Integer parameters
print("Test 6: Integer parameters")
print("-" * 60)
config_int = {
    "method": "ax",
    "parameters": {
        "num_layers": {"min": 1, "max": 5},
        "hidden_size": {"min": 64, "max": 512},
    },
    "metric": {"name": "loss", "goal": "minimize"},
}

try:
    suggestions = ax_search_next_runs([], config_int, n=3)
    print(f"✓ Generated {len(suggestions)} suggestions with integer params")
    for i, s in enumerate(suggestions):
        layers = s.config["num_layers"]["value"]
        hidden = s.config["hidden_size"]["value"]
        print(
            f"  Suggestion {i+1}: num_layers={layers} (type: {type(layers).__name__}), hidden_size={hidden} (type: {type(hidden).__name__})"
        )
    print()
except Exception as e:
    import traceback

    print(f"✗ Error: {e}")
    traceback.print_exc()
    print()

# Test 7: Multi-objective optimization (MOO) with 2 objectives
print("Test 7: Multi-objective optimization (2 objectives)")
print("-" * 60)
config_moo = {
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

try:
    suggestions = ax_search_next_runs([], config_moo, n=3)
    print(f"✓ Generated {len(suggestions)} MOO suggestions")
    for i, s in enumerate(suggestions):
        lr = s.config["learning_rate"]["value"]
        bs = s.config["batch_size"]["value"]
        print(f"  Suggestion {i+1}: lr={lr:.4f}, batch_size={bs}")
    print(f"  Search info keys: {list(suggestions[0].search_info.keys())}")
    assert suggestions[0].search_info.get("is_multi_objective") is True
    assert suggestions[0].search_info.get("objective_names") == ["accuracy", "latency"]
    print("  ✓ MOO metadata present in search_info")
    print()
except Exception as e:
    import traceback

    print(f"✗ Error: {e}")
    traceback.print_exc()
    print()

# Test 8: MOO with 3 objectives and thresholds
print("Test 8: MOO with 3 objectives and thresholds")
print("-" * 60)
config_moo_3 = {
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

try:
    suggestions = ax_search_next_runs([], config_moo_3, n=2)
    print(f"✓ Generated {len(suggestions)} suggestions with 3 objectives")
    print(f"  Objectives: {suggestions[0].search_info.get('objective_names')}")
    print()
except Exception as e:
    import traceback

    print(f"✗ Error: {e}")
    traceback.print_exc()
    print()

# Test 9: MOO with historical data
print("Test 9: MOO with historical data")
print("-" * 60)
moo_runs = [
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
    SweepRun(
        state=RunState.finished,
        config={"learning_rate": {"value": 0.05}, "batch_size": {"value": 96}},
        summary_metrics={"accuracy": 0.75, "latency": 30},
    ),
]

try:
    suggestions = ax_search_next_runs(moo_runs, config_moo, n=2)
    print(f"✓ Generated {len(suggestions)} MOO suggestions with {len(moo_runs)} historical trials")
    if suggestions[0].search_info.get("pareto_frontier"):
        print(f"  Pareto frontier contains {len(suggestions[0].search_info['pareto_frontier'])} points")
    else:
        print("  (Pareto frontier not yet available - may need more completed trials)")
    print()
except Exception as e:
    import traceback

    print(f"✗ Error: {e}")
    traceback.print_exc()
    print()

# Test 10: Outcome constraints (single-objective)
print("Test 10: Outcome constraints (single-objective)")
print("-" * 60)
config_constraints = {
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

try:
    suggestions = ax_search_next_runs([], config_constraints, n=2)
    print(f"✓ Generated {len(suggestions)} suggestions with outcome constraints")
    print()
except Exception as e:
    import traceback

    print(f"✗ Error: {e}")
    traceback.print_exc()
    print()

# Test 11: MOO + outcome constraints
print("Test 11: MOO + outcome constraints")
print("-" * 60)
config_moo_constraints = {
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

try:
    suggestions = ax_search_next_runs([], config_moo_constraints, n=2)
    print(f"✓ Generated {len(suggestions)} MOO suggestions with constraints")
    print()
except Exception as e:
    import traceback

    print(f"✗ Error: {e}")
    traceback.print_exc()
    print()

# Test 12: Validation error - both metric and metrics specified
print("Test 12: Validation error - both metric and metrics")
print("-" * 60)
config_both = {
    "method": "ax",
    "parameters": {"lr": {"min": 0.001, "max": 0.1}},
    "metric": {"name": "loss", "goal": "minimize"},
    "metrics": [
        {"name": "accuracy", "goal": "maximize"},
        {"name": "latency", "goal": "minimize"},
    ],
}

try:
    ax_search_next_runs([], config_both, n=1)
    print("✗ Should have raised an error")
except ValueError as e:
    if "Cannot specify both" in str(e):
        print(f"✓ Correctly rejected config with both metric and metrics")
        print(f"  Error: {e}")
    else:
        print(f"✗ Wrong error: {e}")
print()

# Test 13: Validation error - metrics with < 2 items
print("Test 13: Validation error - metrics with < 2 items")
print("-" * 60)
config_one_metric = {
    "method": "ax",
    "parameters": {"lr": {"min": 0.001, "max": 0.1}},
    "metrics": [
        {"name": "accuracy", "goal": "maximize"},
    ],
}

try:
    ax_search_next_runs([], config_one_metric, n=1)
    print("✗ Should have raised an error")
except ValueError as e:
    if "at least 2" in str(e):
        print(f"✓ Correctly rejected config with < 2 metrics")
        print(f"  Error: {e}")
    else:
        print(f"✗ Wrong error: {e}")
print()

# Test 14: Validation error - invalid constraint format
print("Test 14: Validation error - invalid constraint format")
print("-" * 60)
from sweeps.config.schema import parse_metric_constraint

invalid_constraints = [
    "latency > 100",  # Invalid operator
    "123metric <= 100",  # Invalid metric name
    "latency <=",  # Missing value
]

for constraint in invalid_constraints:
    try:
        parse_metric_constraint(constraint)
        print(f"✗ Should have rejected: '{constraint}'")
    except ValueError as e:
        print(f"✓ Correctly rejected: '{constraint}'")

# Test valid constraints
valid_constraints = [
    "latency <= 100",
    "accuracy >= 0.9",
    "memory_usage <= 512.5",
    "loss <= -0.5",  # Negative value
]

for constraint in valid_constraints:
    try:
        metric, op, bound = parse_metric_constraint(constraint)
        print(f"✓ Parsed: '{constraint}' -> metric={metric}, op={op}, bound={bound}")
    except ValueError as e:
        print(f"✗ Should have accepted: '{constraint}' - {e}")
print()

# Test 15: Validation error - early_terminate + metrics (MOO)
print("Test 15: Validation error - early_terminate + metrics")
print("-" * 60)
config_et_moo = {
    "method": "ax",
    "parameters": {"lr": {"min": 0.001, "max": 0.1}},
    "metrics": [
        {"name": "accuracy", "goal": "maximize"},
        {"name": "latency", "goal": "minimize"},
    ],
    "early_terminate": {"type": "hyperband", "min_iter": 3},
}

try:
    ax_search_next_runs([], config_et_moo, n=1)
    print("✗ Should have raised an error")
except ValueError as e:
    if "early_terminate is not supported" in str(e):
        print(f"✓ Correctly rejected config with early_terminate + MOO")
        print(f"  Error: {e}")
    else:
        print(f"✗ Wrong error: {e}")
print()

# Test 16: Backward compatibility - single metric still works
print("Test 16: Backward compatibility - single metric still works")
print("-" * 60)
config_single = {
    "method": "ax",
    "parameters": {
        "learning_rate": {"min": 0.001, "max": 0.1},
        "batch_size": {"min": 16, "max": 128},
    },
    "metric": {"name": "val_loss", "goal": "minimize"},
}

try:
    suggestions = ax_search_next_runs([], config_single, n=2)
    print(f"✓ Generated {len(suggestions)} suggestions using single metric (backward compat)")
    assert suggestions[0].search_info.get("is_multi_objective") is None
    print("  ✓ No MOO metadata in search_info (as expected)")
    print()
except Exception as e:
    import traceback

    print(f"✗ Error: {e}")
    traceback.print_exc()
    print()

print("=" * 60)
print("✅ All tests completed!")
print("=" * 60)
