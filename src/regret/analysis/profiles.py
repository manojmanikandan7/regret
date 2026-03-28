"""Runtime profile analysis for algorithm comparison.

This module provides functions to compute and analyze inverse runtime profiles,
which capture the probability of reaching fitness thresholds within given
evaluation budgets. The tail-sum formula relates these profiles to expected
cumulative regret.

Key concepts:
    - **Inverse Runtime Profile**: P(\\tau_v <= T) = probability of reaching fitness v by time T.
    - **Tail-sum formula for expectations**: E[CR(T)] = Sum_{v=1}^{f*} Sum_{t'=1}^{T} [1 - P(\\tau_v <= t')]
        for integer-valued fitness functions with unit increments (Work-in-progress for other functions).

Type Aliases (imported from regret._types):
    HistoryResults: Results dict keyed by algorithm name.
    TimeGrid: 1D numpy array of evaluation time points.
    FitnessLevels: 1D numpy array of fitness thresholds.
    InverseProfiles: Dict with key algorithm name, value 2D numpy array P(\\tau_v <= T), shape (F, T).
    EmpiricalCumulativeRegret: E[CR(T)] computed directly from trajectory averaging.
    ProfileCumulativeRegret: E[CR(T)] derived via layer-cake from inverse profiles.

Functions:
    run_profile_analysis: Compute profiles and verify tail-sum relationship.
"""

import numpy as np

from regret._types import (
    EmpiricalCumulativeRegret,
    FitnessLevels,
    HistoryResults,
    InverseProfiles,
    ProfileCumulativeRegret,
    TimeGrid,
)
from regret.core.metrics import (
    compute_inv_runtime_profile,
    cumulative_regret,
    inv_profile_to_expected_cumulative_regret,
)


def run_profile_analysis(
    results: HistoryResults,
    f_star: float,
    budget: int,
    max_time_grid_points: int = 10000,
) -> tuple[TimeGrid, FitnessLevels, InverseProfiles, EmpiricalCumulativeRegret, ProfileCumulativeRegret]:
    """Analyze runtime profile and cumulative regret relationships.

    Computes inverse runtime profiles P(\\tau_v <= T) for all fitness levels,
    then derives expected cumulative regret via the tail-sum representation.

    Args:
        results: Results keyed by algorithm name. Each value is a list of run
            result dicts with 'trajectory' key containing a list of
            (evaluation_index, current_value, best_value) tuples.
        f_star: Global optimum fitness value.
        budget: Maximum evaluation budget used to build the time grid.
        max_time_grid_points: Maximum number of time grid points for large budgets.
            Higher values improve accuracy but increase memory usage. For budgets
            smaller than this value, every evaluation is included in the grid.

    Returns:
        Tuple of (time_grid, fitness_levels, inv_profiles, empirical_ecr, profile_ecr):
            - time_grid: 1D array of evaluation time points, shape (T,).
            - fitness_levels: 1D array of fitness thresholds, shape (F,).
            - inv_profiles: Dict mapping algorithm name to inverse runtime profile
              array of shape (F, T). Entry [f, t] = P(\\tau_v <= T), the probability
              that fitness level fitness_levels[f] was reached by time_grid[t].
            - empirical_ecr: Dict mapping algorithm name to E[CR(T)] array of
              shape (T,), computed by direct averaging of cumulative regrets.
            - profile_ecr: Dict mapping algorithm name to E[CR(T)] array of
              shape (T,), derived from inverse runtime profile via tail-sum formula.
    """
    # Build grids
    # Time grid: every evaluation from 1 to budget
    # For large budgets, subsample to keep memory reasonable
    # FIXME: Subsampling reduces accuracy; find a balanced subsampling measure
    max_points = 5000
    if budget <= max_points:
        time_grid = np.arange(1, budget + 1, dtype=float)
    else:
        time_grid = np.unique(
            np.concatenate(
                [
                    np.arange(1, max_points + 1),  # dense at start
                    np.geomspace(max_points, budget, max_points).astype(int),
                ]
            )
        ).astype(float)

    # Fitness levels: integers from 1 to f_star
    # if f_star <= 200:
    fitness_levels = np.arange(1, int(f_star) + 1, dtype=float)
    # else:
    #     fitness_levels = np.unique(
    #         np.concatenate(
    #             [
    #                 np.arange(1, 50),
    #                 np.geomspace(50, f_star, 150).astype(int),
    #             ]
    #         )
    #     ).astype(float)

    # Fitness levels are integer-spaced for integer-valued objectives and
    # dense linear for normalized/continuous objectives (e.g., NK in [0, 1]).
    eps = 1e-12
    is_integer_scale = abs(f_star - round(f_star)) <= eps and f_star >= 1.0
    if is_integer_scale:
        fitness_levels = np.arange(1.0, float(int(round(f_star))) + 1.0, dtype=float)
    else:
        # HACK: Not well grounded for verification, but enables support for gathering inverse runtime profile plots
        n_levels = int(budget)
        hi = max(float(f_star), eps)
        fitness_levels = np.linspace(0.0, hi, num=n_levels, dtype=float)

    inv_profiles: dict[str, np.ndarray] = {}
    empirical_ecr: dict[str, np.ndarray] = {}
    profile_ecr: dict[str, np.ndarray] = {}

    for alg_name, runs in results.items():
        trajectories = [r["trajectory"] for r in runs if "trajectory" in r]
        if not trajectories:
            continue

        # Compute inverse runtime profile P(\\tau_v <= T) for each fitness level and time
        inv_profile = compute_inv_runtime_profile(trajectories, fitness_levels, time_grid)
        inv_profiles[alg_name] = inv_profile

        # Derive E[CR(T)] from inverse runtime profile via tail-sum representation
        profile_ecr[alg_name] = inv_profile_to_expected_cumulative_regret(inv_profile, fitness_levels, time_grid)

        # Compute E[CR(T)] directly from per-run cumulative regrets
        # (with track_incumbent=True for best-so-far solutions)
        # Interpolate each run's CR onto the shared time_grid
        cr_matrix = []
        for traj in trajectories:
            cr_points = cumulative_regret(traj, f_star, track_incumbent=True)
            if not cr_points:
                continue
            t_vals = np.array([t for t, _ in cr_points])
            cr_vals = np.array([v for _, v in cr_points])
            # Interpolate onto time_grid (left-fill for t < first evaluation)
            interpolated = np.interp(time_grid, t_vals, cr_vals, left=0.0, right=cr_vals[-1])
            cr_matrix.append(interpolated)
        if cr_matrix:
            # Averages cumulative regret across runs (empirical mean at each eval point t)
            empirical_ecr[alg_name] = np.mean(cr_matrix, axis=0)

    return time_grid, fitness_levels, inv_profiles, empirical_ecr, profile_ecr
