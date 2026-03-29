"""Regret metrics for evaluating optimization algorithm performance.

This module computes various types of regret metrics:

**Simple Regret (SR)**: The gap between the global optimum f* and the best
solution found at termination: SR(T) = f* - f(x_best_T). This measures
the quality of the final recommendation.

**Instantaneous Regret (IR)**: The gap at each evaluation step t.
- IR_current(t) = f* - f(x_t): regret of the solution evaluated at time t
  (track_incumbent=False)
- IR_incumbent(t) = f* - f(x_best_t): regret of the best solution found so far
  (track_incumbent=True). This is equivalent to simple regret at time t and is
  monotonically non-increasing.

**Cumulative Regret (CR)**: The integral of instantaneous regret over time:
CR(T) = ∫₀ᵀ IR(t) dt. This measures total cost incurred during optimization
(e.g., total error accumulated across all iterations).

**Normalized Regret (NR)**: Simple regret scaled to [0, 1] by the fitness range:
NR = (f* - f(x)) / (f* - f_worst). Enables comparison across problems with
different scales.

**Expected Simple Regret (E[SR])**: The mean simple regret across multiple
independent runs. This is the standard metric for comparing fixed-budget
optimization algorithms.

All formulas assume maximization problems, where f* is the global maximum.
"""

import numpy as np

Trajectory = list[tuple[int, float, float]]


def compute_statistics(regrets: np.ndarray) -> dict:
    """Compute summary statistics for regrets.

    Args:
        regrets: Array of regret values.

    Returns:
        Mapping with basic summary statistics.
    """
    regrets_arr = np.asarray(regrets, dtype=float)
    if regrets_arr.ndim != 1 or regrets_arr.size == 0:
        raise ValueError("regrets must be a non-empty 1D array.")

    return {
        "mean": float(np.mean(regrets_arr)),
        "median": float(np.median(regrets_arr)),
        "std": float(np.std(regrets_arr)),
        "min": float(np.min(regrets_arr)),
        "max": float(np.max(regrets_arr)),
        "q25": float(np.percentile(regrets_arr, 25)),
        "q75": float(np.percentile(regrets_arr, 75)),
    }


def probability_optimal(regrets: np.ndarray, tolerance: float = 1e-9) -> float:
    """Estimate probability of reaching the optimum.

    Args:
        regrets: Array of regret values.
        tolerance: Threshold for considering a run optimal.

    Returns:
        Fraction of runs with regret within the tolerance.
    """
    regrets_arr = np.asarray(regrets, dtype=float)
    if regrets_arr.ndim != 1 or regrets_arr.size == 0:
        raise ValueError("regrets must be a non-empty 1D array.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    return float(np.mean(regrets_arr <= tolerance))


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
    trajectory: Trajectory, f_star: float, track_incumbent: bool = False
) -> list[tuple[int, float]]:
    """Compute instantaneous regret series for a trajectory at each evaluation (time) point.

    Instantaneous regret measures the gap from the optimum at each evaluation step.

    When track_incumbent=False (default):
        IR(t) = f* - f(x_t)
        This is the "true" instantaneous regret of the solution evaluated at time t.

    When track_incumbent=True:
        IR(t) = f* - f(x_best_t)
        This tracks the regret of the best-so-far (incumbent) solution, which is
        equivalent to simple regret at time t. This variant is monotonically
        non-increasing.

    Args:
        trajectory: Sequence of (evaluations, current_value, best_value) tuples.
        f_star: Global optimum value.
        track_incumbent: If True, use best-so-far value; otherwise use current value.

    Returns:
        List of (evaluation, instantaneous regret) pairs.
    """
    if track_incumbent:
        return [(t, f_star - best_value) for t, _, best_value in trajectory]
    return [(t, f_star - current_value) for t, current_value, _ in trajectory]


def cumulative_regret(trajectory: Trajectory, f_star: float, track_incumbent: bool = False) -> list[tuple[int, float]]:
    """Compute cumulative regret series for a trajectory.

    Cumulative regret is the running integral of instantaneous regret using a left-hold
    approximation over the evaluation grid: CR(T) = \\sum_{1}^{t} IR(t).

    This measures the total "cost" incurred during optimization, as opposed to
    simple regret which only measures final solution quality.

    Args:
        trajectory: Sequence of (evaluations, current_value, best_value) tuples.
        f_star: Global optimum value.
        track_incumbent: If True, use best-so-far value; otherwise use current value.
            See instantaneous_regret() for details on this parameter.

    Returns:
        List of (evaluation, cumulative regret) pairs using left-hold integration.
    """
    if len(trajectory) < 2:
        return [(trajectory[0][0], 0.0)] if trajectory else []

    inst_regrets = instantaneous_regret(trajectory, f_star, track_incumbent)
    # Include the missing initial rectangle from time 0 to the first evaluation
    # Under left-hold integration: cumulative_regret(t) = (t - 0) * regret_at_0
    # We approximate regret_at_0 as the regret at the first evaluation
    first_t, first_r = inst_regrets[0]
    result = [(first_t, first_t * first_r)]
    cumulative = first_t * first_r

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


def time_to_target(
    trajectory: Trajectory,
    f_star: float,
    target_regret: float,
    tolerance: float = 1e-9,
) -> int | None:
    """Compute time to reach a target regret threshold.

    This generalizes TTFO (time-to-first-optimum):
    - ttfo(traj, f_star) ≡ time_to_target(traj, f_star, target_regret=0.0)

    Args:
        trajectory: Sequence of (evaluations, current_value, best_value) tuples.
        f_star: Global optimum value.
        target_regret: Regret threshold to achieve (e.g., 1.0 for "within 1 of optimum").
        tolerance: Floating-point tolerance for comparisons.

    Returns:
        First evaluation where regret <= target_regret, or None if never reached.
    """
    target_fitness = f_star - target_regret
    for t, _, best_value in trajectory:
        if best_value >= target_fitness - tolerance:
            return t
    return None


def normalized_regret(
    solution_value: float,
    f_star: float,
    f_worst: float,
) -> float:
    """Compute normalized regret scaled to [0, 1].

    Normalized regret allows comparison across problems with different
    fitness scales: NR = (f* - f(x)) / (f* - f_worst).

    Args:
        solution_value: Objective value of the solution.
        f_star: Global optimum value.
        f_worst: Worst possible fitness value.

    Returns:
        Normalized regret in [0, 1], where 0 means optimal and 1 means worst.

    Raises:
        ValueError: If f_star == f_worst (degenerate problem).
    """
    if f_star == f_worst:
        # Degenerate case: all solutions have the same fitness
        return 0.0 if solution_value >= f_star else 1.0
    return (f_star - solution_value) / (f_star - f_worst)


def expected_simple_regret(regrets: np.ndarray) -> float:
    """Compute expected (mean) simple regret E[SR(T)] across runs.

    This is the standard metric for fixed-budget optimization comparison.
    Equivalent to compute_statistics(regrets)["mean"] but semantically explicit.

    Args:
        regrets: Array of simple regret values from independent runs.

    Returns:
        Mean simple regret.

    Raises:
        ValueError: If regrets is empty or not 1D.
    """
    regrets_arr = np.asarray(regrets, dtype=float)
    if regrets_arr.ndim != 1 or regrets_arr.size == 0:
        raise ValueError("regrets must be a non-empty 1D array.")
    return float(np.mean(regrets_arr))


def inv_runtime_profile_single_run(
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
    levels = np.asarray(fitness_levels, dtype=float)
    if levels.ndim != 1:
        raise ValueError("fitness_levels must be a 1D array.")
    if levels.size and np.any(np.diff(levels) < 0):
        raise ValueError("fitness_levels must be sorted in non-decreasing order.")

    hitting_times = np.full(levels.size, np.inf)
    # Walk the trajectory once; best-so-far values are monotone so we can sweep levels
    level_idx = 0
    n_levels = levels.size
    for t, _, best in trajectory:
        while level_idx < n_levels and best >= levels[level_idx]:  # If best at t >= to a level k, record the t
            hitting_times[level_idx] = float(t)
            level_idx += 1
        if level_idx >= n_levels:
            break
    return hitting_times


def compute_inv_runtime_profile(
    trajectories: list[Trajectory],
    fitness_levels: np.ndarray,
    time_grid: np.ndarray,
) -> np.ndarray:
    """Compute inverse runtime profile: P(\\tau_v <= T) for all fitness levels and time points.

    Args:
        trajectories: List of trajectories, one per independent run.
        fitness_levels: Sorted array of fitness thresholds (shape F,).
        time_grid: Evaluation counts at which to evaluate the inverse profile (shape T,).

    Returns:
        Array of shape (F, T) where element [i, j] is the fraction of runs
        that reached fitness_levels[i] by time_grid[j].
    """
    levels = np.asarray(fitness_levels, dtype=float)
    times = np.asarray(time_grid, dtype=float)
    if levels.ndim != 1 or times.ndim != 1:
        raise ValueError("fitness_levels and time_grid must be 1D arrays.")
    if levels.size and np.any(np.diff(levels) < 0):
        raise ValueError("fitness_levels must be sorted in non-decreasing order.")
    if times.size and np.any(np.diff(times) < 0):
        raise ValueError("time_grid must be sorted in non-decreasing order.")
    if len(trajectories) == 0:
        raise ValueError("trajectories must contain at least one trajectory.")

    n_runs = len(trajectories)
    n_levels = levels.size

    # hitting_times[run, level] = first evaluation reaching that level (inf if not)
    hitting_times = np.full((n_runs, n_levels), np.inf)
    for r, traj in enumerate(trajectories):
        hitting_times[r] = inv_runtime_profile_single_run(traj, levels)

    # inverse profile[level, time] = fraction of runs where hitting_time <= time_grid[t]
    # Shape: (F, T)
    # Outer broadcast: (n_runs, n_levels, 1) <= (1, 1, n_times)
    inv_profile = np.mean(
        hitting_times[:, :, np.newaxis] <= times[np.newaxis, np.newaxis, :],
        axis=0,
    )  # shape (F, T)
    return inv_profile


def inv_profile_to_expected_cumulative_regret(
    inv_profile: np.ndarray,
    fitness_levels: np.ndarray,
    time_grid: np.ndarray,
) -> np.ndarray:
    """Derive E[CR(T)] from runtime profile via tail-sum formula.

    Computes expected cumulative regret using
    E[CR(T)] = Sum_{v=1}^{f*} Sum_{t'=1}^{T} [1 - P(\\tau_v <= t')]
    via integration over levels and time.

    Args:
        inv_profile: Runtime inverse profile array of shape (F, T) from compute_inv_runtime_profile.
        fitness_levels: Fitness thresholds of shape (F,); used for level spacing weights.
        time_grid: Evaluation counts of shape (T,) corresponding to profile columns.

    Returns:
        Array of shape (T,) containing expected cumulative regret at each time point in time_grid.
    """
    inv_profile_arr = np.asarray(inv_profile, dtype=float)
    levels = np.asarray(fitness_levels, dtype=float)
    times = np.asarray(time_grid, dtype=float)

    if inv_profile_arr.ndim != 2:
        raise ValueError("(inverse) profile must be a 2D array of shape (F, T).")
    if levels.ndim != 1 or times.ndim != 1:
        raise ValueError("fitness_levels and time_grid must be 1D arrays.")
    if inv_profile_arr.shape != (levels.size, times.size):
        raise ValueError("(inverse) profile shape must match (len(fitness_levels), len(time_grid)).")
    if levels.size and np.any(np.diff(levels) < 0):
        raise ValueError("fitness_levels must be sorted in non-decreasing order.")
    if times.size and np.any(np.diff(times) < 0):
        raise ValueError("time_grid must be sorted in non-decreasing order.")

    # survival: P(\tau_v > t) = 1 - profile [profile: P(\tau_v <= t)]
    profiles = 1.0 - inv_profile_arr  # (F, T)

    # Level spacing \delta(v) from origin to include first level weight.
    delta_v = np.diff(levels, prepend=0.0)  # (F,)

    ## Weighted cumulative sum along the time axis, then sum over levels

    # Time spacing \delta(t) from origin to include [0, time_grid[0]] interval.
    dt = np.diff(times, prepend=0.0)  # (T,)

    # Weighted area under survival curve up to each T
    # cumulative sum of (survival * dt) along time axis (The Sum_{t=1}^T part)
    area_per_level = np.cumsum(profiles * dt[np.newaxis, :], axis=1)  # (F, T)

    # Sum over levels, weighted by \delta(v) (The Sum_{v=0}^f* part)
    # ecr = np.einsum("f,ft->t", delta_v, area_per_level)
    expected_cumulative_regret = delta_v @ area_per_level
    return expected_cumulative_regret
