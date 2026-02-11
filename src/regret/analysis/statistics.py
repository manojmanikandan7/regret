import numpy as np
from scipy import stats


def mann_whitney_test(
    regrets1: np.ndarray, regrets2: np.ndarray
) -> tuple[float, float]:
    """Perform Mann-Whitney U test."""
    statistic, pvalue = stats.mannwhitneyu(regrets1, regrets2, alternative="two-sided")
    return statistic, pvalue


def wilcoxon_test(regrets1: np.ndarray, regrets2: np.ndarray) -> tuple[float, float]:
    """Perform Wilcoxon signed-rank test."""
    statistic, pvalue = stats.wilcoxon(regrets1, regrets2)
    return statistic, pvalue


def effect_size_cohens_d(regrets1: np.ndarray, regrets2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(regrets1), len(regrets2)
    var1, var2 = np.var(regrets1, ddof=1), np.var(regrets2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(regrets1) - np.mean(regrets2)) / pooled_std


def bootstrap_confidence_interval(
    regrets: np.ndarray, n_bootstrap: int = 10000, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for mean regret."""
    rng = np.random.default_rng(42)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(regrets, size=len(regrets), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return float(lower), float(upper)
