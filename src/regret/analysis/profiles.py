import numpy as np

from regret.core.metrics import (
    compute_inv_runtime_profile,
    cumulative_regret,
    inv_profile_to_expected_cumulative_regret,
)


def run_profile_analysis(
    results: dict[str, list[dict]],
    f_star: float,
    budget: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Analyze runtime profile and cumulative regret relationships.

    Computes runtime inverse profiles P(\\tau_v <= T) for all fitness levels,
    gets runtime profiles then derives expected cumulative regret via
    layer-cake representation.

    Args:
        results: Algorithm name -> list of run dicts (must include 'trajectory' key).
        f_star: Global optimum fitness value.
        budget: Maximum evaluation budget used to build the time grid.

    Returns:
        Tuple of (time_grid, fitness_levels, profiles, empirical_ecr, profile_ecr)
        where:
        - time_grid: Evaluation grid used for profile and regret calculations.
        - fitness_levels: Fitness thresholds used for runtime profiles.
        - inv_profiles: alg_name -> runtime inverse profile array [inv_profiles[f,t] = P(\\tau_f <= t)] (F, T)
        - empirical_ecr: alg_name -> E[CR(T)] from direct cumulative regret
        - profile_ecr: alg_name -> E[CR(T)] derived from inverse runtime profile
    """
    # Build grids
    # Time grid: every evaluation from 1 to budget
    # For large budgets, subsample to keep memory reasonable
    max_points = 500
    if budget <= max_points:
        time_grid = np.arange(1, budget + 1, dtype=float)
    else:
        time_grid = np.unique(
            np.concatenate(
                [
                    np.arange(1, min(200, budget) + 1),  # dense at start
                    np.geomspace(200, budget, max_points - 200).astype(int),
                ]
            )
        ).astype(float)

    # Fitness levels: integers from 1 to f_star
    # For large f_star (e.g. BinVal), subsample
    if f_star <= 200:
        fitness_levels = np.arange(1, int(f_star) + 1, dtype=float)
    else:
        fitness_levels = np.unique(
            np.concatenate(
                [
                    np.arange(1, 50),
                    np.geomspace(50, f_star, 150).astype(int),
                ]
            )
        ).astype(float)

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

        # Derive E[CR(T)] from inverse runtime profile via layer-cake representation
        profile_ecr[alg_name] = inv_profile_to_expected_cumulative_regret(inv_profile, fitness_levels, time_grid)

        # Compute E[CR(T)] directly from per-run cumulative regrets (with use_best=True, for best-so-far solutions)
        # Interpolate each run's CR onto the shared time_grid
        cr_matrix = []
        for traj in trajectories:
            cr_points = cumulative_regret(traj, f_star, use_best=True)
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
