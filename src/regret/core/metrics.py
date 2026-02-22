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

def simple_regret(solution_value: float, f_star: float) -> float:
    """
    Compute simple regret according the solution value 
    Could be the best value so far or the current value.
    """
    return f_star - solution_value 

def instantaneous_regret(
    trajectory: Trajectory, f_star: float, use_best: bool = False
) -> list[tuple[int, float]]:
    """
    Compute instantaneous regret based on values at each time step by default.
    
    Returns a list of (evaluations, regret) pairs.
    """
    return [(t, f_star - best_value) if use_best else (t, f_star - current_value) for t, current_value, best_value in trajectory]


def cumulative_regret(trajectory: Trajectory, f_star: float, use_best: bool = False) -> float:
    """
    Compute cumulative regret using values at each time step by default.

    Uses a left-hold approximation over the evaluation grid.
    """
    if len(trajectory) < 2:
        return 0.0

    regrets = instantaneous_regret(trajectory, f_star, use_best)
    cumulative = 0.0
    for (t_i, r_i), (t_next, _) in zip(regrets[:-1], regrets[1:]):
        if t_next < t_i:
            raise ValueError("Trajectory evaluations must be non-decreasing.")
        cumulative += (t_next - t_i) * r_i
    return float(cumulative)

def ttfo(trajectory: Trajectory, f_star: float, tolerance: float = 1e-9) -> int | None:
    """
    Time to first optimum (TTFO).

    Returns the first evaluation index at which the optimum is reached,
    or None if not found within the trajectory.
    """
    for t, _, best_value in trajectory:
        if abs(best_value - f_star) <= tolerance:
            return t
    return None
