# Ax Benchmarks

Benchmarks validating the Ax integration with W&B Sweeps.

## Overview

This directory contains benchmarking tools for:
- **SOO benchmarks**: Comparing Ax against sklearn-based Bayesian optimization
- **MOO benchmarks**: Multi-objective optimization with hypervolume tracking
- **Integration tests**: Verifying W&B wrapper matches direct Ax API

For a simple getting started example, see `examples/ax_getting_started.py`.

## Files

| File | Description |
|------|-------------|
| `ax_native_benchmarks.py` | Unified benchmark suite with SOO, MOO, and integration test modes |
| `plot_moo_results.py` | Generates hypervolume and Pareto front plots from benchmark results |
| `configs/` | Sample YAML configs for constrained MOO problems |

## Quick Start

**Run SOO benchmark (Ax vs sklearn-bayes):**
```bash
python ax_native_benchmarks.py --mode soo --trials 24 --replications 3
```

**Run MOO benchmark:**
```bash
python ax_native_benchmarks.py --mode moo --trials 32 --replications 1
```

**Run integration test (W&B wrapper vs direct Ax):**
```bash
python ax_native_benchmarks.py --mode integration --trials 20 --replications 3
```

## Benchmark Problems

### Single-Objective (SOO)
- **Branin** (2D): Classic optimization benchmark
- **Hartmann6** (6D): Higher-dimensional test function

### Multi-Objective (MOO)
- **DTLZ2** (6D, 2 objectives): Standard unconstrained MOO benchmark
- **C2DTLZ2** (6D, 2 objectives, 1 constraint): Constrained variant
- **WeldedBeam** (4D, 2 objectives, 4 constraints): Engineering design problem

## Benchmark Methodology

- **Hypervolume**: Standard MOO metric for Pareto front quality
- **Multiple replications**: Different random seeds for statistical robustness
- **JSON output**: Per-replication curves, final Pareto fronts, aggregate statistics

Results are saved to `benchmark_results/` for analysis and plotting.
