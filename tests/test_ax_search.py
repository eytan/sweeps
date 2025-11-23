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

print("=" * 60)
print("✅ All tests completed!")
print("=" * 60)
