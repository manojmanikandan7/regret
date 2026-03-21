"""Tests for experiment runner execution behavior."""

from pathlib import Path

import pytest

from regret.algorithms.local_search import RLS
from regret.experiments.runner import ExperimentRunner
from regret.problems.pseudo_boolean import OneMax


def test_run_experiment_parallel_preserves_seed_order(tmp_path: Path) -> None:
    """Parallel runs should return results aligned with seed order."""
    runner = ExperimentRunner(output_dir=str(tmp_path))
    problem = OneMax(n=10)

    results = runner.run_experiment(
        RLS,
        problem,
        budget=12,
        runs=6,
        mode="lite",
        parallel=True,
    )

    assert [r["seed"] for r in results] == list(range(6))


def test_run_single_trajectory_stride_downsamples_and_keeps_last_point() -> None:
    """Trajectory stride should reduce size while preserving terminal state."""
    runner = ExperimentRunner()
    problem = OneMax(n=10)

    result = runner.run_single(
        RLS,
        problem,
        budget=20,
        seed=0,
        mode="full",
        trajectory_stride=6,
    )

    trajectory = result["trajectory"]
    assert trajectory
    assert len(trajectory) < result["evaluations"]
    assert trajectory[-1][0] == result["evaluations"]


def test_run_experiment_rejects_invalid_trajectory_stride() -> None:
    """Stride less than one should fail fast with a clear error."""
    runner = ExperimentRunner()
    problem = OneMax(n=8)

    with pytest.raises(ValueError, match="trajectory_stride"):
        runner.run_experiment(
            RLS,
            problem,
            budget=10,
            runs=1,
            mode="full",
            trajectory_stride=0,
        )
