"""Plotting utilities for experiment result visualization.

This module provides functions to visualize optimization algorithm performance
through various plot types including regret curves, boxplots, heatmaps, and
runtime profiles. All plots use matplotlib with standard defaults.

Plot types:
    - Regret curves: Mean regret vs budget with uncertainty bands.
    - Boxplots: Distribution comparison at specific budgets.
    - Convergence probability: P(optimal) vs budget.
    - Performance profiles: CDF of regrets across algorithms.
    - History plots: Fitness trajectory over evaluations.
    - TTFO distribution: Time-to-first-optimum scatter plots.
    - Runtime profiles: Inverse runtime profile surfaces and curves.

Statistical annotations include pairwise tests (Mann-Whitney U, Wilcoxon)
and effect sizes (Cohen's d) for algorithm comparison.

Type Aliases (imported from regret._types):
    KeyedResults: Results dict keyed by (algorithm_name, budget) tuples.
    HistoryResults: Results dict keyed by algorithm name only.
    FitnessLevels: 1D numpy array of intermediate fitness levels.
    TimeGrid: 1D numpy array of evaluation time points.
    InverseRuntimeProfile: 2D numpy array P(\\tau_v <= T).
    EmpiricalCumulativeRegret: E[CR(T)] computed directly from trajectory averaging.
    ProfileCumulativeRegret: E[CR(T)] derived via layer-cake from inverse profiles.
"""

import logging
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from regret._types import (
    EmpiricalCumulativeRegret,
    FitnessLevels,
    HistoryResults,
    InverseRuntimeProfile,
    KeyedResults,
    ProfileCumulativeRegret,
    TimeGrid,
)
from regret.analysis.statistics import (
    bootstrap_confidence_interval,
    effect_size_cohens_d,
    mann_whitney_test,
    wilcoxon_test,
)
from regret.core.metrics import (
    cumulative_regret,
    history_best_series,
    history_current_series,
    instantaneous_regret,
)

logger = logging.getLogger(__name__)

matplotlib.use("Agg")  # Use non-interactive backend to avoid tkinter threading issues


# Set publication-quality defaults
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 400


def _clean_numeric(values: np.ndarray | list[float]) -> np.ndarray:
    """Return finite values as a 1D float array.

    Args:
        values: Input array or list of numeric values.

    Returns:
        1D numpy array containing only finite (non-NaN, non-inf) values.
    """
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _safe_sem(values: np.ndarray) -> float:
    """Compute standard error of the mean robustly for small samples.

    Args:
        values: Array of numeric values.

    Returns:
        SEM value, or 0.0 if sample size is <= 1.
    """
    n = len(values)
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))


def _safe_sd(values: np.ndarray) -> float:
    """Compute sample standard deviation robustly for small samples.

    Args:
        values: Array of numeric values.

    Returns:
        Sample SD, or 0.0 if sample size is <= 1.
    """
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _summary_interval(
    samples: np.ndarray,
    spread: str = "sem",
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
) -> tuple[float, float, float]:
    """Compute center and uncertainty interval for 1D samples.

    Args:
        samples: 1D array of sample values.
        spread: Uncertainty type - one of "none", "sem", "sd", "bootstrap_ci", "iqr".
        confidence: Confidence level for bootstrap CI (default: 0.95).
        n_bootstrap: Number of bootstrap resamples (default: 2000).

    Returns:
        Tuple of (center, lower, upper) values. For "none", all three are equal.

    Raises:
        ValueError: If spread is not a recognized option.
    """
    vals = _clean_numeric(samples)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan

    mode = spread.strip().lower()
    if mode == "none":
        c = float(np.mean(vals))
        return c, c, c
    if mode == "sem":
        c = float(np.mean(vals))
        d = _safe_sem(vals)
        return c, c - d, c + d
    if mode == "sd":
        c = float(np.mean(vals))
        d = _safe_sd(vals)
        return c, c - d, c + d
    if mode == "bootstrap_ci":
        c = float(np.mean(vals))
        lo, hi = bootstrap_confidence_interval(vals, n_bootstrap=n_bootstrap, confidence=confidence)
        return c, lo, hi
    if mode == "iqr":
        c = float(np.median(vals))
        lo, hi = np.percentile(vals, [25, 75])
        return c, float(lo), float(hi)

    raise ValueError("spread must be one of {'none', 'sem', 'sd', 'bootstrap_ci', 'iqr'}.")


def _format_pvalue(pvalue: float) -> str:
    """Format a p-value for display in annotations.

    Args:
        pvalue: Statistical p-value to format.

    Returns:
        Formatted string: "nan" if non-finite, "<1e-4" if very small,
        otherwise 4 decimal places.
    """
    if not np.isfinite(pvalue):
        return "nan"
    if pvalue < 1e-4:
        return "<1e-4"
    return f"{pvalue:.4f}"


def _safe_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size with safeguards for degenerate cases.

    Args:
        a: First sample array.
        b: Second sample array.

    Returns:
        Cohen's d value, or NaN if either sample has fewer than 2 values
        or if the result is non-finite.
    """
    a_vals = _clean_numeric(a)
    b_vals = _clean_numeric(b)
    if len(a_vals) < 2 or len(b_vals) < 2:
        return np.nan

    d = float(effect_size_cohens_d(a_vals, b_vals))
    if not np.isfinite(d):
        return np.nan
    return d


def _pairwise_stats_text(
    distributions: dict[str, np.ndarray],
    reference: str | None = None,
    paired: bool = False,
) -> str:
    """Build compact pairwise significance and effect-size annotation text.

    Args:
        distributions: Mapping of algorithm name to regret/metric arrays.
        reference: Reference algorithm for comparisons (defaults to first sorted).
        paired: If True, use Wilcoxon signed-rank test; otherwise Mann-Whitney U.

    Returns:
        Multi-line annotation string with test results, or empty string if
        insufficient data.
    """
    if not distributions:
        return ""

    names = sorted(distributions.keys())
    if len(names) < 2:
        return ""

    ref = reference if reference in distributions else names[0]
    ref_vals = _clean_numeric(distributions[ref])
    if len(ref_vals) == 0:
        return ""

    lines = [f"Reference: {ref}"]
    for name in names:
        if name == ref:
            continue

        vals = _clean_numeric(distributions[name])
        if len(vals) == 0:
            continue

        if paired and len(vals) == len(ref_vals):
            stat, pval = wilcoxon_test(ref_vals, vals)
            test_name = "Wilcoxon"
        else:
            stat, pval = mann_whitney_test(ref_vals, vals)
            test_name = "MWU"

        d = _safe_cohens_d(ref_vals, vals)
        lines.append(f"{name}: {test_name} p={_format_pvalue(float(pval))}, d={d:.3f}, stat={float(stat):.2f}")

    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _draw_pairwise_annotation(ax, text: str) -> None:
    """Draw pairwise statistics annotation box on axes.

    Args:
        ax: Matplotlib axes to annotate.
        text: Annotation text to display.
    """
    if not text:
        return
    ax.text(
        0.99,
        0.01,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )


def _time_series_band(
    values: np.ndarray,
    spread: str = "sem",
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute center and uncertainty band across runs for each time index.

    Args:
        values: 2D array of shape (n_runs, n_time) containing time series.
        spread: Uncertainty type - one of "none", "sem", "sd", "bootstrap_ci", "iqr".
        confidence: Confidence level for bootstrap CI.
        n_bootstrap: Number of bootstrap resamples.

    Returns:
        Tuple of (centers, lowers, uppers) arrays, each of shape (n_time,).
    """
    n_time = values.shape[1]
    centers = np.full(n_time, np.nan, dtype=float)
    lowers = np.full(n_time, np.nan, dtype=float)
    uppers = np.full(n_time, np.nan, dtype=float)

    for idx in range(n_time):
        samples = values[:, idx]
        c, lo, hi = _summary_interval(
            samples,
            spread=spread,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
        )
        centers[idx] = c
        lowers[idx] = lo
        uppers[idx] = hi

    return centers, lowers, uppers


def _finalize_figure(
    fig: Figure,
    save_path: str | None = None,
    show: bool = True,
):
    """Save and/or render figure with consistent behavior.

    Args:
        fig: Matplotlib Figure to finalize.
        save_path: Path to save the figure. Creates parent directories if needed.
        show: Whether to display the figure interactively.
    """
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _ttfo_for_trajectory(
    trajectory: list[tuple[int, float, float]],
    f_star: float | None,
    tolerance: float = 1e-9,
) -> int | None:
    """Find first evaluation index where best-so-far reaches the optimum.

    Args:
        trajectory: List of (evaluations, current_value, best_value) tuples.
        f_star: Global optimum value. If None, returns None immediately.
        tolerance: Tolerance for considering optimum reached.

    Returns:
        Evaluation index at which TTFO occurs, or None if never reached.
    """
    if f_star is None:
        return None

    for eval_idx, _current, best in trajectory:
        if abs(best - f_star) <= tolerance:
            return int(eval_idx)
    return None


def plot_simple_regret_curves(
    results: KeyedResults,
    save_path: str | None = None,
    log_scale: bool = True,
    title: str | None = None,
    show: bool = True,
    spread: str = "sem",
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    annotate_pairwise: bool = False,
    comparison_budget: int | None = None,
    paired_runs: bool = False,
) -> None:
    """Plot mean simple regret vs budget for multiple algorithms.

    Creates a line plot showing mean simple regret on the y-axis versus
    evaluation budget on the x-axis, with one line per algorithm. Uncertainty
    bands can be added around each line.

    Args:
        results: Results keyed by (algorithm_name, budget) tuples. Each value
            is a list of run result dicts with keys:
            - regret (float): Simple regret = f_star - best_value
            - best_value (float): Best fitness found
            - optimal (bool): Whether optimum was found
            - evaluations (int): Number of evaluations
            - seed (int): Random seed
        save_path: Path to save the figure. If None, figure is not saved.
        log_scale: Use log-log scale for axes (default True).
        title: Plot title. If None, no title is shown.
        show: Display the figure interactively (default True).
        spread: Uncertainty band type. Options:
            - "sem": Standard error of the mean
            - "sd": Standard deviation
            - "bootstrap_ci": Bootstrap confidence interval
            - "iqr": Interquartile range
            - "none": No uncertainty band
        confidence: Confidence level for bootstrap CI (default 0.95).
        n_bootstrap: Number of bootstrap resamples (default 2000).
        annotate_pairwise: Add pairwise statistical comparison annotations.
        comparison_budget: Budget at which to compute pairwise comparisons.
            If None, uses the maximum budget.
        paired_runs: Use paired statistical tests (Wilcoxon vs Mann-Whitney).
    """

    algorithms = sorted(set(alg for alg, _ in results.keys()))
    budgets = sorted(set(budget for _, budget in results.keys()))

    fig, ax = plt.subplots()
    budget_distributions: dict[int, dict[str, np.ndarray]] = {}

    for alg in algorithms:
        center_regrets = []
        lower_regrets = []
        upper_regrets = []

        for budget in budgets:
            if (alg, budget) in results:
                regrets = np.array([r["regret"] for r in results[(alg, budget)]], dtype=float)
                center, lower, upper = _summary_interval(
                    regrets,
                    spread=spread,
                    confidence=confidence,
                    n_bootstrap=n_bootstrap,
                )
                center_regrets.append(center)
                lower_regrets.append(lower)
                upper_regrets.append(upper)
                budget_distributions.setdefault(int(budget), {})[alg] = regrets
            else:
                center_regrets.append(np.nan)
                lower_regrets.append(np.nan)
                upper_regrets.append(np.nan)

        center_regrets = np.array(center_regrets, dtype=float)
        lower_regrets = np.array(lower_regrets, dtype=float)
        upper_regrets = np.array(upper_regrets, dtype=float)

        if log_scale:
            ax.loglog(budgets, center_regrets, marker="o", label=alg, linewidth=2)
        else:
            ax.plot(budgets, center_regrets, marker="o", label=alg, linewidth=2)

        if spread.strip().lower() != "none":
            if log_scale:
                lower_regrets = np.where(lower_regrets > 0, lower_regrets, np.nan)
            ax.fill_between(budgets, lower_regrets, upper_regrets, alpha=0.2)

    ax.set_xlabel("Budget (evaluations)")
    ax.set_ylabel("Mean Simple Regret")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if annotate_pairwise and budget_distributions:
        if comparison_budget is None:
            comparison_budget = max(budget_distributions.keys())
        if comparison_budget in budget_distributions:
            dist_map = budget_distributions[comparison_budget]
            if dist_map:
                reference = min(
                    dist_map.keys(),
                    key=lambda name: float(np.nanmean(_clean_numeric(dist_map[name]))),
                )
                text = _pairwise_stats_text(dist_map, reference=reference, paired=paired_runs)
                if text:
                    _draw_pairwise_annotation(
                        ax,
                        f"At budget={comparison_budget}\n{text}",
                    )

    _finalize_figure(fig, save_path=save_path, show=show)


def plot_simple_regret_boxplots(
    results: KeyedResults,
    budget: int,
    save_path: str | None = None,
    title: str | None = None,
    show: bool = True,
    show_points: bool = True,
    annotate_pairwise: bool = False,
    reference_algorithm: str | None = None,
    paired_runs: bool = False,
) -> None:
    """Create boxplots comparing algorithm regret distributions at a specific budget.

    Displays side-by-side boxplots showing the distribution of simple regret
    values for each algorithm at a specified evaluation budget.

    Args:
        results: Results keyed by (algorithm_name, budget) tuples. Each value
            is a list of run result dicts with 'regret' key (float).
        budget: The budget value to filter results by.
        save_path: Path to save the figure. If None, figure is not saved.
        title: Plot title. If None, a default title is generated.
        show: Display the figure interactively (default True).
        show_points: Overlay individual data points on boxplots (default True).
        annotate_pairwise: Add pairwise statistical comparison annotations.
        reference_algorithm: Reference algorithm for pairwise comparisons.
            If None, the algorithm with lowest mean regret is used.
        paired_runs: Use paired statistical tests (Wilcoxon vs Mann-Whitney).
    """

    algorithms = sorted([alg for alg, b in results.keys() if b == budget])
    data = [np.array([r["regret"] for r in results[(alg, budget)]], dtype=float) for alg in algorithms]

    fig, ax = plt.subplots()
    bp = ax.boxplot(data, tick_labels=algorithms, patch_artist=True, notch=True)

    colors = cm.get_cmap("Set3")(np.linspace(0, 1, len(algorithms)))
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)

    if show_points:
        rng = np.random.default_rng(123)
        for i, vals in enumerate(data, start=1):
            x = i + rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(x, vals, s=12, alpha=0.35, color="black", zorder=3)

    ax.set_ylabel("Simple Regret")
    ax.set_xlabel("Algorithm")
    ax.set_title(title if title else f"Regret Distribution at Budget = {budget}")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    if annotate_pairwise and algorithms:
        distributions = {alg: vals for alg, vals in zip(algorithms, data, strict=False)}
        ref = reference_algorithm
        if ref is None and distributions:
            ref = min(
                distributions.keys(),
                key=lambda name: float(np.mean(distributions[name])),
            )
        text = _pairwise_stats_text(distributions, reference=ref, paired=paired_runs)
        _draw_pairwise_annotation(ax, text)

    _finalize_figure(fig, save_path=save_path, show=show)


def plot_convergence_probability(
    results: KeyedResults,
    save_path: str | None = None,
    title: str | None = None,
    show: bool = True,
    show_confidence_band: bool = True,
    confidence: float = 0.95,
) -> None:
    """Plot probability of finding optimum vs budget.

    Creates a line plot showing the fraction of runs that found the global
    optimum (regret < 1e-9) at each budget level.

    Args:
        results: Results keyed by (algorithm_name, budget) tuples. Each value
            is a list of run result dicts with 'regret' key (float).
        save_path: Path to save the figure. If None, figure is not saved.
        title: Plot title. If None, no title is shown.
        show: Display the figure interactively (default True).
        show_confidence_band: Show bootstrap confidence band around probability.
        confidence: Confidence level for bootstrap CI (default 0.95).
    """

    algorithms = sorted(set(alg for alg, _ in results.keys()))
    budgets = sorted(set(budget for _, budget in results.keys()))

    fig, ax = plt.subplots()

    for alg in algorithms:
        probs = []
        lowers = []
        uppers = []
        for budget in budgets:
            if (alg, budget) in results:
                regrets = np.array([r["regret"] for r in results[(alg, budget)]], dtype=float)
                successes = regrets < 1e-9
                p = float(np.mean(successes))
                probs.append(p)
                if show_confidence_band:
                    _, lo, hi = _summary_interval(
                        successes.astype(float),
                        spread="bootstrap_ci",
                        confidence=confidence,
                        n_bootstrap=2000,
                    )
                    lowers.append(lo)
                    uppers.append(hi)
            else:
                probs.append(np.nan)
                lowers.append(np.nan)
                uppers.append(np.nan)

        ax.semilogx(budgets, probs, marker="o", label=alg, linewidth=2)
        if show_confidence_band:
            ax.fill_between(budgets, lowers, uppers, alpha=0.2)

    ax.set_xlabel("Budget (evaluations)")
    ax.set_ylabel("P(optimal)")
    ax.set_ylim((-0.05, 1.05))
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    _finalize_figure(fig, save_path=save_path, show=show)


def plot_comparison_heatmap(
    results: KeyedResults,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Create heatmap of mean regrets (log10 scale) across algorithms and budgets.

    Displays a 2D heatmap with algorithms on the y-axis and budgets on the
    x-axis, with cell colors indicating log10(mean simple regret).

    Args:
        results: Results keyed by (algorithm_name, budget) tuples. Each value
            is a list of run result dicts with 'regret' key (float).
        save_path: Path to save the figure. If None, figure is not saved.
        show: Display the figure interactively (default True).
        save_path: Path to save the figure.
        show: Display the figure interactively.
    """

    algorithms = sorted(set(alg for alg, _ in results.keys()))
    budgets = sorted(set(budget for _, budget in results.keys()))
    data = np.zeros((len(algorithms), len(budgets)))

    for i, alg in enumerate(algorithms):
        for j, budget in enumerate(budgets):
            if (alg, budget) in results:
                regrets = np.array([r["regret"] for r in results[(alg, budget)]], dtype=float)
                data[i, j] = np.log10(np.mean(regrets) + 1e-10)
            else:
                data[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(budgets)))
    ax.set_yticks(range(len(algorithms)))
    ax.set_xticklabels([f"{b:.0e}" for b in budgets], rotation=45, ha="right")
    ax.set_yticklabels(algorithms)

    ax.set_xlabel("Budget")
    ax.set_ylabel("Algorithm")
    ax.set_title("Mean Regret (log10 scale)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log10(mean regret)")

    _finalize_figure(fig, save_path=save_path, show=show)


def plot_performance_profile(
    results: KeyedResults,
    budget: int,
    save_path: str | None = None,
    title: str | None = None,
    show: bool = True,
    annotate_pairwise: bool = False,
    reference_algorithm: str | None = None,
    paired_runs: bool = False,
) -> None:
    """Create performance profile (CDF of regrets) at a specific budget.

    Displays the cumulative distribution function (CDF) of simple regret
    values for each algorithm, showing what fraction of runs achieved
    regret below each threshold.

    Args:
        results: Results keyed by (algorithm_name, budget) tuples. Each value
            is a list of run result dicts with 'regret' key (float).
        budget: The budget value to filter results by.
        save_path: Path to save the figure. If None, figure is not saved.
        title: Plot title. If None, a default title is generated.
        show: Display the figure interactively (default True).
        annotate_pairwise: Add pairwise statistical comparison annotations.
        reference_algorithm: Reference algorithm for pairwise comparisons.
            If None, the algorithm with lowest mean regret is used.
        paired_runs: Use paired statistical tests (Wilcoxon vs Mann-Whitney).
    """

    algorithms = sorted([alg for alg, b in results.keys() if b == budget])
    distributions: dict[str, np.ndarray] = {}

    fig, ax = plt.subplots()

    for alg in algorithms:
        regrets = np.array([r["regret"] for r in results[(alg, budget)]], dtype=float)
        distributions[alg] = regrets
        regrets = np.maximum(regrets, 1e-12)
        sorted_regrets = np.sort(regrets)
        cdf = np.arange(1, len(sorted_regrets) + 1) / len(sorted_regrets)
        ax.plot(sorted_regrets, cdf, label=alg, linewidth=2)

    ax.set_xlabel("Simple Regret")
    ax.set_ylabel("Cumulative Probability")
    ax.set_xscale("log", nonpositive="clip")
    ax.set_title(title if title else f"Performance Profile (Budget = {budget})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if annotate_pairwise and distributions:
        ref = reference_algorithm
        if ref is None:
            ref = min(
                distributions.keys(),
                key=lambda name: float(np.mean(distributions[name])),
            )
        text = _pairwise_stats_text(distributions, reference=ref, paired=paired_runs)
        _draw_pairwise_annotation(ax, text)

    _finalize_figure(fig, save_path=save_path, show=show)


def plot_history(
    results: HistoryResults,
    f_star: float,
    series: str = "current",
    log_x: bool = False,
    log_y: bool = False,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
    ttfo_tolerance: float = 1e-9,
    show_ttfo_markers: bool = True,
    spread: str = "sem",
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> None:
    """Plot fitness history series vs evaluations.

    Displays fitness trajectory over evaluation count for each algorithm,
    showing either the current evaluated value or the best-so-far value.

    Args:
        results: Results keyed by algorithm name. Each value is a list of run
            result dicts with 'trajectory' key containing a list of
            (evaluation_index, current_value, best_value) tuples.
        f_star: Global optimum value (shown as horizontal reference line).
        series: Series type to plot:
            - "current": Value of the solution evaluated at each step
            - "best": Best value found so far (monotonically increasing)
        log_x: Use logarithmic x-axis (default False).
        log_y: Use logarithmic y-axis (default False).
        title: Plot title. If None, no title is shown.
        save_path: Path to save the figure. If None, figure is not saved.
        show: Display the figure interactively (default True).
        ttfo_tolerance: Tolerance for TTFO (time-to-first-optimum) detection.
        show_ttfo_markers: Show vertical lines at mean TTFO with error bars.
        spread: Uncertainty band type ("sem", "sd", "bootstrap_ci", "iqr", "none").
        confidence: Confidence level for bootstrap CI (default 0.95).
        n_bootstrap: Number of bootstrap resamples (default 1000).
    """
    fig, ax = plt.subplots()

    for alg, runs in results.items():
        n_total = len(runs)
        trajectories = [r["trajectory"] for r in runs if "trajectory" in r]
        n_with_trajectory = len(trajectories)

        if n_with_trajectory < n_total:
            n_skipped = n_total - n_with_trajectory

            logger.warning(
                "plot_history: Algorithm '%s' has %d/%d runs without trajectory data (skipped)",
                alg,
                n_skipped,
                n_total,
            )

        if not trajectories:
            continue

        series_list = []
        ttfo_values = []

        for trajectory in trajectories:
            if series == "current":
                series_points = history_current_series(trajectory)
            elif series == "best":
                series_points = history_best_series(trajectory)
            else:
                raise ValueError("series must be 'current' or 'best'.")

            series_list.append(series_points)

            ttfo = _ttfo_for_trajectory(trajectory, f_star, tolerance=ttfo_tolerance)
            if ttfo is not None:
                ttfo_values.append(ttfo)

        if not series_list:
            continue

        times = [t for t, _ in series_list[0]]
        values = []
        for series_points in series_list:
            series_map = {t: v for t, v in series_points}
            values.append([series_map.get(t, np.nan) for t in times])

        values = np.array(values, dtype=float)
        mean_vals, lower_vals, upper_vals = _time_series_band(
            values,
            spread=spread,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
        )

        if log_x and log_y:
            (line,) = ax.loglog(times, mean_vals, marker=None, label=alg, linewidth=2)
        elif log_x:
            (line,) = ax.semilogx(times, mean_vals, marker=None, label=alg, linewidth=2)
        elif log_y:
            (line,) = ax.semilogy(times, mean_vals, marker=None, label=alg, linewidth=2)
        else:
            (line,) = ax.plot(times, mean_vals, marker=None, label=alg, linewidth=2)

        if spread.strip().lower() != "none":
            band_lo = lower_vals
            if log_y:
                band_lo = np.where(band_lo > 0, band_lo, np.nan)
            ax.fill_between(times, band_lo, upper_vals, alpha=0.2)

        if show_ttfo_markers and ttfo_values:
            mean_ttfo = float(np.mean(ttfo_values))
            sem_ttfo = float(np.std(ttfo_values) / np.sqrt(len(ttfo_values)))

            # Interpolate y-value on mean curve for TTFO marker
            times_arr = np.asarray(times, dtype=float)
            mean_arr = np.asarray(mean_vals, dtype=float)

            # Guard against NaNs for interpolation
            valid = np.isfinite(times_arr) & np.isfinite(mean_arr)
            if np.any(valid):
                x_valid = times_arr[valid]
                y_valid = mean_arr[valid]
                order = np.argsort(x_valid)
                x_valid = x_valid[order]
                y_valid = y_valid[order]

                y_at_ttfo = float(
                    np.interp(
                        mean_ttfo,
                        x_valid,
                        y_valid,
                        left=y_valid[0],
                        right=y_valid[-1],
                    )
                )

                ax.axvline(
                    mean_ttfo,
                    color=line.get_color(),
                    linestyle=":",
                    linewidth=1.4,
                    alpha=0.9,
                    label=f"Mean time to first optimum ({alg})",
                )
                ax.scatter(
                    [mean_ttfo],
                    [y_at_ttfo],
                    color=line.get_color(),
                    marker="o",
                    s=35,
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=4,
                )
                ax.errorbar(
                    [mean_ttfo],
                    [y_at_ttfo],
                    xerr=[[sem_ttfo], [sem_ttfo]],
                    fmt="none",
                    ecolor=line.get_color(),
                    elinewidth=1.2,
                    capsize=4,
                    alpha=0.8,
                )

    ax.axhline(
        f_star,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="f* (Global Optimum)",
    )
    ax.set_xlabel("Evaluations")
    if series == "best":
        ax.set_ylabel("Best value so far")
    else:
        ax.set_ylabel("Current value")

    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    _finalize_figure(fig, save_path=save_path, show=show)


def plot_regret_curves(
    results: HistoryResults,
    f_star: float,
    series: str = "instantaneous",
    track_incumbent: bool = False,
    log_x: bool = False,
    log_y: bool = False,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
    ttfo_tolerance: float = 1e-9,
    show_ttfo_markers: bool = True,
    spread: str = "sem",
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> None:
    """Plot regret series vs evaluations.

    Displays instantaneous or cumulative regret trajectory over evaluation
    count for each algorithm.

    Args:
        results: Results keyed by algorithm name. Each value is a list of run
            result dicts with 'trajectory' key containing a list of
            (evaluation_index, current_value, best_value) tuples.
        f_star: Global optimum value.
        series: Regret series type:
            - "instantaneous": Regret at each evaluation step
            - "cumulative": Running sum of instantaneous regret
        track_incumbent: If True, compute regret based on best-so-far (incumbent)
            value; otherwise use current evaluated value. When True, instantaneous
            regret is monotonically non-increasing.
        log_x: Use logarithmic x-axis (default False).
        log_y: Use logarithmic y-axis (default False).
        title: Plot title. If None, no title is shown.
        save_path: Path to save the figure. If None, figure is not saved.
        show: Display the plot interactively (default True).
        ttfo_tolerance: Tolerance for TTFO (time-to-first-optimum) detection.
        show_ttfo_markers: Show vertical lines at mean TTFO.
        spread: Uncertainty band type ("sem", "sd", "bootstrap_ci", "iqr", "none").
        confidence: Confidence level for bootstrap CI (default 0.95).
        n_bootstrap: Number of bootstrap samples (default 1000).
    """
    fig, ax = plt.subplots()

    for alg, runs in results.items():
        n_total = len(runs)
        trajectories = [r["trajectory"] for r in runs if "trajectory" in r]
        n_with_trajectory = len(trajectories)

        if n_with_trajectory < n_total:
            n_skipped = n_total - n_with_trajectory
            logger.warning(
                "plot_regret_curves: Algorithm '%s' has %d/%d runs without trajectory data (skipped)",
                alg,
                n_skipped,
                n_total,
            )

        if not trajectories:
            continue

        series_list = []
        ttfo_values = []

        for trajectory in trajectories:
            if series == "instantaneous":
                series_points = instantaneous_regret(trajectory, f_star, track_incumbent=track_incumbent)
            elif series == "cumulative":
                series_points = cumulative_regret(trajectory, f_star, track_incumbent=track_incumbent)
            else:
                raise ValueError("series must be 'instantaneous' or 'cumulative'.")

            series_list.append(series_points)

            ttfo = _ttfo_for_trajectory(trajectory, f_star, tolerance=ttfo_tolerance)
            if ttfo is not None:
                ttfo_values.append(ttfo)

        if not series_list:
            continue

        times = [t for t, _ in series_list[0]]
        values = []
        for series_points in series_list:
            series_map = {t: v for t, v in series_points}
            values.append([series_map.get(t, np.nan) for t in times])

        values = np.array(values, dtype=float)
        mean_vals, lower_vals, upper_vals = _time_series_band(
            values,
            spread=spread,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
        )

        if log_x and log_y:
            (line,) = ax.loglog(times, mean_vals, marker=None, label=alg, linewidth=2)
        elif log_x:
            (line,) = ax.semilogx(times, mean_vals, marker=None, label=alg, linewidth=2)
        elif log_y:
            (line,) = ax.semilogy(times, mean_vals, marker=None, label=alg, linewidth=2)
        else:
            (line,) = ax.plot(times, mean_vals, marker=None, label=alg, linewidth=2)

        if spread.strip().lower() != "none":
            band_lo = lower_vals
            if log_y:
                band_lo = np.where(band_lo > 0, band_lo, np.nan)
            ax.fill_between(times, band_lo, upper_vals, alpha=0.2)

        if show_ttfo_markers and ttfo_values:
            mean_ttfo = float(np.mean(ttfo_values))
            sem_ttfo = float(np.std(ttfo_values) / np.sqrt(len(ttfo_values)))

            # Interpolate y-value on mean curve for TTFO marker
            times_arr = np.asarray(times, dtype=float)
            mean_arr = np.asarray(mean_vals, dtype=float)

            # Guard against NaNs for interpolation
            valid = np.isfinite(times_arr) & np.isfinite(mean_arr)
            if np.any(valid):
                x_valid = times_arr[valid]
                y_valid = mean_arr[valid]
                order = np.argsort(x_valid)
                x_valid = x_valid[order]
                y_valid = y_valid[order]

                y_at_ttfo = float(
                    np.interp(
                        mean_ttfo,
                        x_valid,
                        y_valid,
                        left=y_valid[0],
                        right=y_valid[-1],
                    )
                )

                ax.axvline(
                    mean_ttfo,
                    color=line.get_color(),
                    linestyle=":",
                    linewidth=1.4,
                    alpha=0.9,
                    label=f"Mean time to first optimum ({alg})",
                )
                ax.scatter(
                    [mean_ttfo],
                    [y_at_ttfo],
                    color=line.get_color(),
                    marker="o",
                    s=35,
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=4,
                )
                ax.errorbar(
                    [mean_ttfo],
                    [y_at_ttfo],
                    xerr=[[sem_ttfo], [sem_ttfo]],
                    fmt="none",
                    ecolor=line.get_color(),
                    elinewidth=1.2,
                    capsize=4,
                    alpha=0.8,
                )

    ax.set_xlabel("Evaluations")
    if series == "instantaneous":
        ax.set_ylabel("Instantaneous regret")
    else:
        ax.set_ylabel("Cumulative regret")

    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    _finalize_figure(fig, save_path=save_path, show=show)


def plot_ttfo_distribution(
    results: HistoryResults,
    f_star: float | None,
    save_path: str | None = None,
    title: str | None = None,
    show: bool = True,
    tolerance: float = 1e-9,
    show_median: bool = True,
    annotate_pairwise: bool = False,
    reference_algorithm: str | None = None,
    paired_runs: bool = False,
) -> None:
    """Plot TTFO (time-to-first-optimum) distribution across algorithms.

    Creates a scatter plot showing individual TTFO samples for each algorithm,
    with optional median TTFO vertical lines.

    Args:
        results: Results keyed by algorithm name. Each value is a list of run
            result dicts with 'trajectory' key containing a list of
            (evaluation_index, current_value, best_value) tuples.
        f_star: Known optimum value. If None, the function returns without plotting.
        save_path: Path to save the figure. If None, figure is not saved.
        title: Plot title. If None, no title is shown.
        show: Whether to display the plot interactively (default True).
        tolerance: Tolerance for considering optimum reached (default 1e-9).
        show_median: Whether to show median TTFO as vertical dashed lines.
        annotate_pairwise: Add pairwise statistical comparison annotations.
        reference_algorithm: Reference algorithm for pairwise comparisons.
        paired_runs: Use paired statistical tests (Wilcoxon vs Mann-Whitney).
    """
    if f_star is None:
        return

    fig, ax = plt.subplots()

    algorithms = sorted(results.keys())
    colors = cm.get_cmap("tab10")(np.linspace(0, 1, len(algorithms)))
    ttfo_map: dict[str, np.ndarray] = {}

    for idx, (alg, color) in enumerate(zip(algorithms, colors, strict=False), start=1):
        runs = results[alg]
        n_total = len(runs)
        n_without_trajectory = 0
        ttfos = []

        for run in runs:
            trajectory = run.get("trajectory", [])
            if not trajectory:
                n_without_trajectory += 1
                continue

            ttfo = _ttfo_for_trajectory(trajectory, f_star, tolerance=tolerance)
            if ttfo is not None:
                ttfos.append(ttfo)

        if n_without_trajectory > 0:
            logger.warning(
                "plot_ttfo_distribution: Algorithm '%s' has %d/%d runs without trajectory data (skipped)",
                alg,
                n_without_trajectory,
                n_total,
            )

        if not ttfos:
            continue

        ttfo_map[alg] = np.asarray(ttfos, dtype=float)

        # Place samples on algorithm rows with light jitter to show multiplicity.
        rng = np.random.default_rng(123 + idx)
        y = np.full(len(ttfos), fill_value=idx, dtype=float) + rng.uniform(-0.12, 0.12, size=len(ttfos))
        ax.scatter(
            ttfos,
            y,
            s=24,
            alpha=0.6,
            color=color,
            label=f"{alg} TTFO samples",
        )

        if show_median:
            med = float(np.median(ttfos))
            ax.vlines(
                x=med,
                ymin=idx - 0.22,
                ymax=idx + 0.22,
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                label=f"{alg} Median",
            )

    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Algorithm")
    ax.set_yticks(np.arange(1, len(algorithms) + 1))
    ax.set_yticklabels(algorithms)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if annotate_pairwise and ttfo_map:
        ref = reference_algorithm
        if ref is None:
            ref = min(ttfo_map.keys(), key=lambda name: float(np.mean(ttfo_map[name])))
        text = _pairwise_stats_text(ttfo_map, reference=ref, paired=paired_runs)
        _draw_pairwise_annotation(ax, text)

    _finalize_figure(fig, save_path=save_path, show=show)


def plot_inverse_runtime_profile_surface(
    inv_profile: InverseRuntimeProfile,
    fitness_levels: FitnessLevels,
    time_grid: TimeGrid,
    f_star: float,
    save_path: str | None = None,
    show: bool = True,
    title: str | None = None,
    n_contours: int = 6,
) -> None:
    """Visualize inverse runtime profile P(\\tau_v <= T) as a 2D heatmap with contours.

    Displays the probability of reaching each fitness level by each time point
    as a heatmap, with contour lines showing iso-probability curves.

    Args:
        inv_profile: Inverse runtime profile array of shape (F, T) where entry
            [f, t] gives P(\\tau_v <= T), the probability that fitness level
            fitness_levels[f] was reached by time time_grid[t].
        fitness_levels: 1D array of fitness level thresholds, shape (F,).
        time_grid: 1D array of evaluation time points, shape (T,).
        f_star: Global optimum fitness value (marked as reference line).
        save_path: Path to save figure. If None, not saved.
        show: Whether to display the figure (default True).
        title: Optional title for the plot.
        n_contours: Number of contour levels to draw (default 6).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Heatmap
    im = ax.imshow(
        inv_profile,
        origin="lower",
        aspect="auto",
        extent=(time_grid[0], time_grid[-1], fitness_levels[0], fitness_levels[-1]),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(im, ax=ax, label="$P(\\tau_v \\leq T)$")

    # Contour lines at evenly spaced probability values
    levels = np.linspace(0.1, 0.9, n_contours)
    T_grid_2d, V_grid_2d = np.meshgrid(time_grid, fitness_levels)
    ax.contour(
        T_grid_2d,
        V_grid_2d,
        inv_profile,
        levels=levels,
        colors="white",
        linewidths=0.8,
        alpha=0.6,
    )

    ax.axhline(f_star, color="red", linestyle="--", linewidth=2, label="f*")
    ax.set_xlabel("Evaluations (T)")
    ax.set_ylabel("Fitness level (v)")
    if title:
        ax.set_title(title)
    ax.legend()

    _finalize_figure(fig, save_path=save_path, show=show)


def plot_inverse_runtime_profile_curves(
    inv_profiles: dict[str, InverseRuntimeProfile],
    fitness_levels: FitnessLevels,
    time_grid: TimeGrid,
    selected_levels: list[float],
    f_star: float,
    save_path: str | None = None,
    show: bool = True,
    title: str | None = None,
) -> None:
    """Compare inverse runtime profiles across algorithms at selected fitness levels.

    Creates a multi-panel plot showing P(\\tau_v <= T) vs T for each algorithm
    at representative fitness thresholds, allowing comparison of how quickly
    different algorithms reach various fitness levels.

    Args:
        inv_profiles: Mapping of algorithm name to inverse runtime profile array
            of shape (F, T).
        fitness_levels: 1D array of fitness level thresholds, shape (F,).
        time_grid: 1D array of evaluation time points, shape (T,).
        selected_levels: List of fitness levels to plot (subselected from
            fitness_levels). One subplot is created per level.
        f_star: Global optimum fitness value.
        save_path: Path to save figure. If None, not saved.
        show: Whether to display the figure (default True).
        title: Optional title for the plot.

    Returns:
        None. Saves and/or displays the figure based on save_path and show.
    """
    fig, axes = plt.subplots(1, len(selected_levels), figsize=(5 * len(selected_levels), 5), sharey=True)
    if len(selected_levels) == 1:
        axes = [axes]

    for ax, level in zip(axes, selected_levels, strict=False):
        # Find closest index in fitness_levels
        idx = int(np.argmin(np.abs(fitness_levels - level)))
        actual_level = fitness_levels[idx]

        for alg_name, inv_profile in inv_profiles.items():
            ax.plot(time_grid, inv_profile[idx, :], label=alg_name, linewidth=2)

        ax.set_title(f"v = {actual_level:.3g}")
        ax.set_xlabel("Evaluations (T)")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("$P(\\tau_v \\leq T)$")
    axes[-1].legend()

    if title:
        fig.suptitle(title)

    _finalize_figure(fig, save_path=save_path, show=show)


logger = logging.getLogger(__name__)

matplotlib.use("Agg")  # Use non-interactive backend to avoid tkinter threading issues


def plot_cr_profile_verification(
    empirical_ecr: EmpiricalCumulativeRegret,
    profile_ecr: ProfileCumulativeRegret,
    time_grid: TimeGrid,
    save_path: str | None = None,
    show: bool = True,
    title: str | None = None,
) -> None:
    """Verify the tail-sum formula by comparing E[CR(T)] from two derivations.

    Plots expected cumulative regret computed both directly (from per-run CR)
    and via the inverse runtime profile. These should match for integer-valued,
    unit-increment fitness functions; any gap indicates an error in trajectory
    recording or metric computation.

    Direct derivation:
        Mean of per-run cumulative regret (track_incumbent=True)

    Profile derivation (tail-sum formula):
        Sum_{v=1}^{f*} Sum_{t'=1}^{T} [1 - P(\\tau_v <= t')]
        {inverse profile: P(\\tau_v <= t); P(\\tau_v > t) = 1 - P(\\tau_v <= t)}

    Note (Work-in-progress): The tail-sum formula only holds exactly for integer-valued,
    unit-increment fitness functions (For now). For other fitness functions (e.g., BinVal
    with exponential spacing), the verification may show discrepancies.

    Args:
        empirical_ecr: Dict mapping algorithm name to E[CR(T)] array computed
            by direct averaging of cumulative regrets. Each array has shape (T,).
        profile_ecr: Dict mapping algorithm name to E[CR(T)] array derived from
            inverse runtime profile via tail-sum formula. Shape (T,).
        time_grid: 1D array of evaluation time points, shape (T,).
        save_path: Path to save figure. If None, not saved.
        show: Whether to display the figure (default True).
        title: Optional title for the plot.
    """
    fig, ax = plt.subplots()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, alg in enumerate(empirical_ecr):
        c = colors[i % len(colors)]
        ax.plot(time_grid, empirical_ecr[alg], color=c, linewidth=2, label=f"{alg} (direct)")
        ax.plot(
            time_grid,
            profile_ecr[alg],
            color=c,
            linewidth=2,
            linestyle="--",
            label=f"{alg} (profile)",
        )

    ax.set_xlabel("Evaluations (T)")
    ax.set_ylabel("E[CR(T)] - incumbent regret")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    _finalize_figure(fig, save_path=save_path, show=show)
