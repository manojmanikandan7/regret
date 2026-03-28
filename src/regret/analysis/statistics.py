"""Statistical tests and effect size measures for algorithm comparison.

This module provides non-parametric statistical tests and effect size measures
commonly used when comparing optimization algorithm performance. These methods
are robust to non-normal distributions typical of regret values.

Functions:
    mann_whitney_test: Unpaired comparison between two independent samples.
    wilcoxon_test: Paired comparison between matched samples.
    effect_size_cohens_d: Standardized mean difference effect size.
    bootstrap_confidence_interval: Bootstrap CI for the mean.
"""

import numpy as np
from scipy import stats


def mann_whitney_test(regrets1: np.ndarray, regrets2: np.ndarray) -> tuple[float, float]:
    """Perform Mann-Whitney U test for independent samples.

    Non-parametric test comparing two independent samples. Tests whether
    one distribution tends to have larger values than the other.

    Args:
        regrets1: First array of regret values.
        regrets2: Second array of regret values.

    Returns:
        Tuple of (U statistic, two-sided p-value).
    """
    statistic, pvalue = stats.mannwhitneyu(regrets1, regrets2, alternative="two-sided")
    return statistic, pvalue


def wilcoxon_test(regrets1: np.ndarray, regrets2: np.ndarray) -> tuple[float, float]:
    """Perform Wilcoxon signed-rank test for paired samples.

    Non-parametric test comparing two related samples (e.g., same random seeds).
    Tests whether the distribution of differences is symmetric around zero.

    Args:
        regrets1: First array of regret values.
        regrets2: Second array of regret values (must have same length).

    Returns:
        Tuple of (test statistic, two-sided p-value).
    """
    return_ob = stats.wilcoxon(regrets1, regrets2)
    statistic, pvalue = return_ob.statistic, return_ob.pvalue
    return statistic, pvalue


def effect_size_cohens_d(regrets1: np.ndarray, regrets2: np.ndarray) -> float:
    """Compute Cohen's d effect size.

    Args:
        regrets1: First array of regret values.
        regrets2: Second array of regret values.

    Returns:
        Cohen's d effect size. Returns 0.0 if both samples have zero variance
        and equal means. Returns inf/-inf if samples have zero variance but
        different means.
    """
    n1, n2 = len(regrets1), len(regrets2)
    var1, var2 = np.var(regrets1, ddof=1), np.var(regrets2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    mean_diff = np.mean(regrets1) - np.mean(regrets2)

    # Handle zero variance case to avoid divide-by-zero warning
    if pooled_std == 0.0:
        if mean_diff == 0.0:
            return 0.0
        return float("inf") if mean_diff > 0 else float("-inf")

    return mean_diff / pooled_std


def bootstrap_confidence_interval(
    regrets: np.ndarray, n_bootstrap: int = 10000, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for mean regret.

    Uses percentile bootstrap method to estimate confidence interval for
    the sample mean. Uses a fixed seed for reproducibility.

    Args:
        regrets: Array of regret values.
        n_bootstrap: Number of bootstrap resamples (default: 10000).
        confidence: Confidence level, e.g., 0.95 for 95% CI (default: 0.95).

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval.
    """
    rng = np.random.default_rng(42)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(regrets, size=len(regrets), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return float(lower), float(upper)
