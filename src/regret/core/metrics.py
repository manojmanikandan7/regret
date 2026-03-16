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
