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

import logging

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

logger = logging.getLogger(__name__)


def _build_adaptive_time_grid(budget: int, max_points: int = 5000) -> np.ndarray:
    """Build an adaptive time grid that is denser at early evaluations.

    Uses a three-phase approach:
    1. Dense coverage for early evaluations (where fitness changes rapidly)
    2. Moderate density for mid-range evaluations
    3. Sparse coverage for late evaluations (where profiles typically plateau)

    This balances accuracy with memory usage for large budgets.

    Args:
        budget: Maximum evaluation count.
        max_points: Target maximum number of grid points (actual may be slightly higher
            due to ensuring endpoint inclusion).

    Returns:
        Sorted array of unique evaluation time points from 1 to budget.
    """
    if budget <= max_points:
        return np.arange(1, budget + 1, dtype=float)

    # Adaptive grid with three phases:
    # Phase 1: Dense linear coverage for first 10% of budget (most fitness improvement)
    # Phase 2: Moderate geometric spacing for 10-50% of budget
    # Phase 3: Sparse geometric spacing for 50-100% of budget

    phase1_end = max(100, budget // 10)  # At least 100 points, or 10% of budget
    phase2_end = budget // 2

    # Allocate points: 40% to phase 1, 35% to phase 2, 25% to phase 3
    n_phase1 = int(max_points * 0.4)
    n_phase2 = int(max_points * 0.35)
    n_phase3 = max_points - n_phase1 - n_phase2

    # Phase 1: Dense linear (every point if possible, otherwise evenly spaced)
    if phase1_end <= n_phase1:
        phase1 = np.arange(1, phase1_end + 1)
    else:
        phase1 = np.linspace(1, phase1_end, n_phase1)

    # Phase 2: Geometric spacing from phase1_end to phase2_end
    if phase2_end > phase1_end:
        phase2 = np.geomspace(phase1_end, phase2_end, n_phase2)
    else:
        phase2 = np.array([])

    # Phase 3: Sparser geometric spacing from phase2_end to budget
    if budget > phase2_end:
        phase3 = np.geomspace(phase2_end, budget, n_phase3)
    else:
        phase3 = np.array([])

    # Combine, convert to int (evaluation counts), and deduplicate
    combined = np.concatenate([phase1, phase2, phase3])
    time_grid = np.unique(np.round(combined).astype(int)).astype(float)

    # Ensure endpoints are included
    if time_grid[0] != 1.0:
        time_grid = np.concatenate([[1.0], time_grid])
    if time_grid[-1] != float(budget):
        time_grid = np.concatenate([time_grid, [float(budget)]])

    return time_grid


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
    # Build adaptive time grid - denser at early evaluations where fitness changes rapidly
    time_grid = _build_adaptive_time_grid(budget, max_points=max_time_grid_points)

    # Build fitness level grid based on objective value type.
    #
    # For integer-valued objectives (OneMax, LeadingOnes, etc.), use integer spacing
    # which aligns with the tail-sum representation for exact E[CR(T)] computation.
    #
    # For continuous objectives (NKLandscape in [0,1]), use linear spacing as an
    # approximation - the tail-sum verification may show minor discrepancies.
    eps = 1e-12
    is_integer_scale = abs(f_star - round(f_star)) <= eps and f_star >= 1.0
    if is_integer_scale:
        fitness_levels = np.arange(1.0, float(int(round(f_star))) + 1.0, dtype=float)
    else:
        # For continuous/normalized objectives: use linear spacing.
        # NOTE: This is an approximation. The tail-sum identity E[CR(T)] = sum of
        # survival probabilities assumes discrete fitness levels. For continuous
        # objectives, this produces approximate results suitable for visualization
        # but may show small discrepancies in cr_profile_verification plots.

        logger.warning(
            "Non-integer f_star=%.4f detected. Using linear fitness level "
            "spacing which provides approximate (not exact) E[CR(T)] computation. "
            "The cr_profile_verification plot may show minor discrepancies.",
            f_star,
            stacklevel=2,
        )
        n_levels = min(int(budget), 1000)  # Cap levels to avoid memory issues
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
