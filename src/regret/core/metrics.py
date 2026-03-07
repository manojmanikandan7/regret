import numpy as np

Trajectory = list[tuple[int, float, float]]


def compute_statistics(regrets: np.ndarray) -> dict:
    """Compute summary statistics for regret values."""
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
    """Compute probability that optimum was found."""
    return float(np.mean(regrets <= tolerance))


def history_current_series(trajectory: Trajectory) -> list[tuple[int, float]]:
    """
    Return (evaluations, current_value) pairs from a trajectory.
    """
    return [(t, current_value) for t, current_value, _ in trajectory]


def history_best_series(trajectory: Trajectory) -> list[tuple[int, float]]:
    """
    Return (evaluations, best_value) pairs from a trajectory.
    """
    return [(t, best_value) for t, _, best_value in trajectory]


# REGRET CALCULATIONS


def simple_regret(solution_value: float, f_star: float) -> float:
    """
    Compute simple regret according to the final solution value
    Could be the best value so far or the current value.
    """
    return f_star - solution_value


def instantaneous_regret(
    trajectory: Trajectory, f_star: float, use_best: bool = False
) -> list[tuple[int, float]]:
    """
    Return (evaluations, instantaneous regret) pairs from a trajectory
    at each evaluation (time) point.

    By default uses current_value; set use_best=True to use best_value.
    """
    if use_best:
        return [(t, f_star - best_value) for t, _, best_value in trajectory]
    return [(t, f_star - current_value) for t, current_value, _ in trajectory]


def cumulative_regret(
    trajectory: Trajectory, f_star: float, use_best: bool = False
) -> list[tuple[int, float]]:
    """
    Return (evaluations, cumulative regret) pairs from a trajectory.

    Computes the running sum of instantaneous regrets using a left-hold
    approximation over the evaluation grid. At each time point t, the
    cumulative regret is the integral of instantaneous regret from
    time 0 to t.

    By default uses current_value; set use_best=True to use best_value.
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
    """
    Time to first optimum (TTFO) based on best_value in the trajectory.

    Returns the first evaluation index at which the optimum is reached,
    or None if not found within the trajectory.
    """
    for t, _, best_value in trajectory:
        if abs(best_value - f_star) <= tolerance:
            return t
    return None
