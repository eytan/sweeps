# Ax Benchmarks

Benchmarks validating the Ax integration with W&B Sweeps.

## Overview

This directory contains benchmarking tools for:
- **SOO benchmarks**: Single-objective optimization with Branin test function
- **MOO benchmarks**: Multi-objective optimization with hypervolume tracking
- **Integration tests**: Verifying JSON serialization roundtrip matches direct Ax

For a simple getting started example, see `examples/ax_getting_started.py`.

## Files

| File | Description |
|------|-------------|
| `ax_benchmarks.py` | Benchmark suite with SOO, MOO, and integration test modes |
| `plot_moo_results.py` | Generates hypervolume and Pareto front plots |
| `configs/` | YAML configs for benchmark problems |

## Quick Start

**Run SOO benchmark (Branin):**
```bash
python ax_benchmarks.py --mode soo --trials 30 --replications 5
```

**Run MOO benchmark (DTLZ2, C2DTLZ2):**
```bash
python ax_benchmarks.py --mode moo --trials 30 --replications 5
```

**Run integration test (direct Ax vs JSON roundtrip):**
```bash
python ax_benchmarks.py --mode integration --trials 20 --replications 3
```

## Benchmark Problems

### Single-Objective (SOO)
- **Branin** (2D): Classic optimization benchmark, optimal value ~0.398

### Multi-Objective (MOO)
- **DTLZ2** (6D, 2 objectives): Standard unconstrained MOO benchmark
- **C2DTLZ2** (6D, 2 objectives, 1 constraint): Constrained variant

## Benchmark Methodology

- **Hypervolume**: Standard MOO metric measuring Pareto front quality
- **Multiple replications**: Different random seeds for statistical robustness
- **Integration test**: Compares in-memory run history vs JSON-serialized
  roundtrip to verify data survives serialization cycle
