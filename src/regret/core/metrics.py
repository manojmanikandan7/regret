import numpy as np


def simple_regret(best_value: float, f_star: float) -> float:
    """Compute simple regret."""
    return f_star - best_value


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
