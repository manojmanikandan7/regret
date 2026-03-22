import numpy as np

Trajectory = list[tuple[int, float, float]]


def compute_statistics(regrets: np.ndarray) -> dict:
    """Compute summary statistics for regrets.

    Args:
        regrets: Array of regret values.

    Returns:
        Mapping with basic summary statistics.
    """
    return {
        "mean": float(np.mean(regrets)),
        "median": float(np.median(regrets)),
        "std": float(np.std(regrets)),
        "min": float(np.min(regrets)),
        "max": float(np.max(regrets)),
        "q25": float(np.percentile(regrets, 25)),
        "q75": float(np.percentile(regrets, 75)),
    }


def probability_optimal(regrets: np.ndarray, tolerance: float = 1e-9) -> float:
    """Estimate probability of reaching the optimum.

    Args:
        regrets: Array of regret values.
        tolerance: Threshold for considering a run optimal.

    Returns:
        Fraction of runs with regret within the tolerance.
    """
    return float(np.mean(regrets <= tolerance))


def history_current_series(trajectory: Trajectory) -> list[tuple[int, float]]:
    """Extract current-value series from a trajectory.

    Args:
        trajectory: Sequence of (evaluations, current_value, best_value) tuples.

    Returns:
        List of (evaluations, current_value) pairs.
    """
    return [(t, current_value) for t, current_value, _ in trajectory]


def history_best_series(trajectory: Trajectory) -> list[tuple[int, float]]:
    """Extract best-value series from a trajectory.

    Args:
        trajectory: Sequence of (evaluations, current_value, best_value) tuples.

    Returns:
        List of (evaluations, best_value) pairs.
    """
    return [(t, best_value) for t, _, best_value in trajectory]


# REGRET CALCULATIONS


def simple_regret(solution_value: float, f_star: float) -> float:
    """Compute simple regret for a final solution value.

    Args:
        solution_value: Objective value of the returned solution.
        f_star: Global optimum value.

    Returns:
        Difference between the optimum and the solution value.
    """
    return f_star - solution_value


def instantaneous_regret(
    trajectory: Trajectory, f_star: float, use_best: bool = False
) -> list[tuple[int, float]]:
    """Compute instantaneous regret series for a trajectory at each evaluation (time) point.

    Args:
        trajectory: Sequence of (evaluations, current_value, best_value) tuples.
        f_star: Global optimum value.
        use_best: If True, use best_value; otherwise use current_value.

    Returns:
        List of (evaluation, instantaneous regret) pairs.
    """
    if use_best:
        return [(t, f_star - best_value) for t, _, best_value in trajectory]
    return [(t, f_star - current_value) for t, current_value, _ in trajectory]


def cumulative_regret(
    trajectory: Trajectory, f_star: float, use_best: bool = False
) -> list[tuple[int, float]]:
    """Compute cumulative regret series for a trajectory.
    cumulative regret is the running sum of instantaneous regrets using a left-hold
    approximation over the evaluation grid. At each time point t, the
    cumulative regret is the integral of instantaneous regret from
    time 0 to t.

    Args:
        trajectory: Sequence of (evaluations, current_value, best_value) tuples.
        f_star: Global optimum value.
        use_best: If True, use best value obtained so far (`best_value`); otherwise use current_value.

    Returns:
        List of (evaluation, cumulative regret) pairs using left-hold integration.
    """
    if len(trajectory) < 2:
        return [(trajectory[0][0], 0.0)] if trajectory else []

    inst_regrets = instantaneous_regret(trajectory, f_star, use_best)
    result = [(inst_regrets[0][0], 0.0)]
    cumulative = 0.0

    for i in range(1, len(inst_regrets)):
        t_prev, r_prev = inst_regrets[i - 1]
        t_curr, _ = inst_regrets[i]
        if t_curr < t_prev:
            raise ValueError("Trajectory evaluations must be non-decreasing.")
        cumulative += (t_curr - t_prev) * r_prev
        result.append((t_curr, cumulative))

    return result


def ttfo(trajectory: Trajectory, f_star: float, tolerance: float = 1e-9) -> int | None:
    """Compute time to first optimum (TTFO).

    Args:
        trajectory: Sequence of (evaluations, current_value, best_value) tuples.
        f_star: Global optimum value.
        tolerance: Allowed deviation from the optimum.

    Returns:
        Evaluation index of first optimum, or None if not reached.
    """
    for t, _, best_value in trajectory:
        if abs(best_value - f_star) <= tolerance:
            return t
    return None


def first_hitting_time_fitness_level(
    trajectory: Trajectory,
    fitness_level: float,
) -> int | None:
    """Return first evaluation at which best fitness reaches a threshold (i.e., best so far >= fitness_level).

    Args:
        trajectory: Sequence of (evaluations, current_value, best_value) tuples.
        fitness_level: Target fitness threshold.

    Returns:
        Evaluation index when fitness_level is first reached, or None if never reached.
    """
    for t, _, best in trajectory:
        if best >= fitness_level:
            return int(t)
    return None


def runtime_profile_single_run(
    trajectory: Trajectory,
    fitness_levels: np.ndarray,
) -> np.ndarray:
    """Compute hitting times for each fitness level in a single run.

    Args:
        trajectory: Sequence of (evaluations, current_value, best_value) tuples.
        fitness_levels: Sorted array of target fitness thresholds.

    Returns:
        Array of shape (len(fitness_levels),) with hitting times; np.inf where level unreached.
    """
    hitting_times = np.full(len(fitness_levels), np.inf)
    # Walk the trajectory once; best is monotone so we can sweep levels
    level_idx = 0
    n_levels = len(fitness_levels)
    for t, _, best in trajectory:
        while level_idx < n_levels and best >= fitness_levels[level_idx]:
            hitting_times[level_idx] = float(t)
            level_idx += 1
        if level_idx >= n_levels:
            break
    return hitting_times


def compute_runtime_profile(
    trajectories: list[Trajectory],
    fitness_levels: np.ndarray,
    time_grid: np.ndarray,
) -> np.ndarray:
    """Compute runtime profile: P(\\tau_v <= T) for all fitness levels and time points.

    Args:
        trajectories: List of trajectories, one per independent run.
        fitness_levels: Sorted array of fitness thresholds (shape F,).
        time_grid: Evaluation counts at which to evaluate the profile (shape T,).

    Returns:
        Array of shape (F, T) where element [i, j] is the fraction of runs
        that reached fitness_levels[i] by time_grid[j].
    """
    n_runs = len(trajectories)
    n_levels = len(fitness_levels)
    n_times = len(time_grid)

    # hitting_times[run, level] = first evaluation reaching that level (inf if not)
    hitting_times = np.full((n_runs, n_levels), np.inf)
    for r, traj in enumerate(trajectories):
        hitting_times[r] = runtime_profile_single_run(traj, fitness_levels)

    # profile[level, time] = fraction of runs where hitting_time <= time_grid[t]
    # Shape: (F, T)
    # Outer broadcast: (n_runs, n_levels, 1) <= (1, 1, n_times)
    profile = np.mean(
        hitting_times[:, :, np.newaxis] <= time_grid[np.newaxis, np.newaxis, :],
        axis=0,
    )  # shape (F, T)
    return profile


def profile_to_expected_cumulative_regret(
    profile: np.ndarray,
    fitness_levels: np.ndarray,
    time_grid: np.ndarray,
) -> np.ndarray:
    """Derive E[CR(T)] from runtime profile via layer-cake identity.

    Computes expected cumulative regret using
    E[CR(T)] = Sum_{v=1}^{f*} Sum_{t'=1}^{T} [1 - P(\\tau_v <= t')]
    via integration over levels and time.

    Args:
        profile: Runtime profile array of shape (F, T) from compute_runtime_profile.
        fitness_levels: Fitness thresholds of shape (F,); used for level spacing weights.
        time_grid: Evaluation counts of shape (T,) corresponding to profile columns.

    Returns:
        Array of shape (T,) containing expected cumulative regret at each time point in time_grid.
    """
    # survival: P(\\tau_v > t) = 1 − profile
    survival = 1.0 - profile  # (F, T)

    # FIXME: Level spacing (\delta(v)): for integer fitness this is 1, but handle non-unit spacing
    if len(fitness_levels) > 1:
        delta_v = np.diff(fitness_levels, prepend=fitness_levels[0])
    else:
        delta_v = np.ones(1)

    # Weighted cumulative sum along the time axis, then sum over levels
    # For each level v: contribution at time T = Σₜ'₌₁ᵀ survival[v, t'] * Δt
    # \delta(t) = 1 for unit-spaced evaluations; use np.cumsum with time spacing
    dt = np.diff(time_grid, prepend=time_grid[0])  # (T,)

    # Weighted area under survival curve up to each T
    # cumulative sum of (survival * dt) along time axis
    area_per_level = np.cumsum(survival * dt[np.newaxis, :], axis=1)  # (F, T)

    # Sum over levels, weighted by delta_v
    ecr = np.einsum("f,ft->t", delta_v, area_per_level)
    return ecr
