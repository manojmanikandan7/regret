"""Shared type definitions for the regret package.

This module provides centralized type aliases and TypedDicts used across
the regret package for experiment results, trajectories, and profile analysis.

Type Aliases:
    TrajectoryPoint: Single point in an optimization trajectory.
    Trajectory: Complete optimization trajectory.
    KeyedResults: Results dict keyed by (algorithm_name, budget) tuples.
    HistoryResults: Results dict keyed by algorithm name only.
    TimeGrid: 1D numpy array of evaluation time points.
    FitnessLevels: 1D numpy array of fitness thresholds.
    InverseRuntimeProfile: 2D numpy array P(\\tau_v <= T).
    EmpiricalCumulativeRegret: E[CR(T)] computed directly from trajectory averaging.
    ProfileCumulativeRegret: E[CR(T)] derived via tail-sum from (inverse) profiles.
    InverseProfiles: Dict mapping algorithm to inverse runtime profile.

TypedDicts:
    RunResult: Result dictionary from a single optimization run.
    TableStatistics: Summary statistics for simple regret values.
"""

from typing import NotRequired, TypeAlias, TypedDict

import numpy as np

# Trajectory Types

TrajectoryPoint: TypeAlias = tuple[int, float, float]
"""Single point in optimization trajectory: (evaluation_index, current_value, best_value)."""

Trajectory: TypeAlias = list[TrajectoryPoint]
"""Complete optimization trajectory as list of (eval, current, best) tuples."""


# Run Result Types


class RunResult(TypedDict):
    """Result dictionary from a single optimization run.

    Required keys (always present):
        regret: Simple regret = f_star - best_value.
        best_value: Best fitness value found during the run.
        optimal: Whether the optimum was found (|best_value - f_star| < 1e-9).
        evaluations: Number of fitness evaluations performed.
        seed: Random seed used for reproducibility.

    Optional keys (present in 'full' mode only):
        trajectory: Full optimization trajectory as list of
            (evaluation_index, current_value, best_value) tuples.
    """

    regret: float
    best_value: float
    optimal: bool
    evaluations: int
    seed: int
    trajectory: NotRequired[Trajectory]


# Result Collection Types

KeyedResults: TypeAlias = dict[tuple[str, int], list[RunResult]]
"""Results keyed by (algorithm_name, budget) tuples.

Each value is a list of run result dictionaries with keys:
    - regret (float): Simple regret = f_star - best_value
    - best_value (float): Best fitness value found
    - optimal (bool): Whether optimum was found
    - evaluations (int): Number of evaluations performed
    - seed (int): Random seed used
    - trajectory (list[tuple], optional): Full (eval, current, best) trajectory

Example:
    {
        ("RLS", 100): [
            {"regret": 0.0, "best_value": 10.0, "optimal": True,
             "evaluations": 100, "seed": 0, "trajectory": [(1, 5.0, 5.0), ...]},
            {"regret": 1.0, "best_value": 9.0, "optimal": False, ...},
        ],
        ("EA", 100): [...],
    }
"""

HistoryResults: TypeAlias = dict[str, list[RunResult]]
"""Results keyed by algorithm name only (single-budget view).

Each value is a list of run result dictionaries (same structure as KeyedResults
values). This is typically created by filtering KeyedResults to a specific budget.

Example:
    {
        "RLS": [{"regret": 0.0, "best_value": 10.0, ...}, ...],
        "EA": [{"regret": 1.0, "best_value": 9.0, ...}, ...],
    }
"""


# Profile Analysis Types

TimeGrid: TypeAlias = np.ndarray
"""1D array of evaluation time points, shape (T,)."""

FitnessLevels: TypeAlias = np.ndarray
"""1D array of fitness threshold levels, shape (F,)."""

InverseRuntimeProfile: TypeAlias = np.ndarray
"""2D array P(\\tau_v <= T), shape (n_fitness_levels, n_time_points).

Entry [f, t] gives the probability that fitness level fitness_levels[f]
was reached by time time_grid[t].
"""

EmpiricalCumulativeRegret: TypeAlias = dict[str, np.ndarray]
"""Empirical expected cumulative regret per algorithm: {algorithm_name: E[CR(T)] array}.

Computed directly by averaging per-run cumulative regret trajectories.
Each array has shape (T,) matching the time_grid, giving E[CR(T)] at each
time point.

This is the "ground truth" E[CR(T)] computed from actual trajectory data.
"""

ProfileCumulativeRegret: TypeAlias = dict[str, np.ndarray]
"""Profile-derived expected cumulative regret per algorithm: {algorithm_name: E[CR(T)] array}.

Derived via the tail-sum formula from inverse runtime profiles:
    E[CR(T)] = Sum_{v=1}^{f*} Sum_{t'=1}^{T} [1 - P(\\tau_v <= t')]
            {inverse profile: P(\\tau_v <= t); P(\\tau_v > t) = 1 - P(\\tau_v <= t)}

Each array has shape (T,) matching the time_grid. For integer-valued,
unit-increment fitness functions, this should match EmpiricalCumulativeRegret.
"""

InverseProfiles: TypeAlias = dict[str, InverseRuntimeProfile]
"""Inverse runtime profiles per algorithm: {algorithm_name: profile_array}."""


# Table Statistics Types


class TableStatistics(TypedDict):
    """Summary statistics computed for simple regret values.

    Attributes:
        mean: Mean simple regret.
        median: Median simple regret.
        std: Standard deviation.
        iqr: Interquartile range (Q75 - Q25).
        ci_lower: Lower bound of bootstrap CI for mean.
        ci_upper: Upper bound of bootstrap CI for mean.
        p_opt: Probability of finding optimum (regret < 1e-9).
        n_runs: Number of runs.
    """

    mean: float
    median: float
    std: float
    iqr: float
    ci_lower: float
    ci_upper: float
    p_opt: float
    n_runs: int


# Public API

__all__ = [
    # Trajectory types
    "TrajectoryPoint",
    "Trajectory",
    # Run result types
    "RunResult",
    # Result collection types
    "KeyedResults",
    "HistoryResults",
    # Profile analysis types
    "TimeGrid",
    "FitnessLevels",
    "InverseRuntimeProfile",
    "EmpiricalCumulativeRegret",
    "ProfileCumulativeRegret",
    "InverseProfiles",
    # Table statistics types
    "TableStatistics",
]
