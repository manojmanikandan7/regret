from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to avoid tkinter threading issues

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from regret.core.metrics import (
    cumulative_regret,
    history_best_series,
    history_current_series,
    instantaneous_regret,
)
from regret.analysis.statistics import (
    bootstrap_confidence_interval,
    effect_size_cohens_d,
    mann_whitney_test,
    wilcoxon_test,
)

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
    """Return finite values as a 1D float array."""
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _safe_sem(values: np.ndarray) -> float:
    """Compute SEM robustly for small samples."""
    n = len(values)
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))


def _safe_sd(values: np.ndarray) -> float:
    """Compute sample standard deviation robustly for small samples."""
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _summary_interval(
    samples: np.ndarray,
    spread: str = "sem",
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
) -> tuple[float, float, float]:
    """
    Return center, lower, upper uncertainty values for 1D samples.

    spread options:
        - "none": no interval
        - "sem": mean +/- SEM
        - "sd": mean +/- SD
        - "bootstrap_ci": bootstrap CI on the mean
        - "iqr": interquartile range around the median
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
        lo, hi = bootstrap_confidence_interval(
            vals, n_bootstrap=n_bootstrap, confidence=confidence
        )
        return c, lo, hi
    if mode == "iqr":
        c = float(np.median(vals))
        lo, hi = np.percentile(vals, [25, 75])
        return c, float(lo), float(hi)

    raise ValueError(
        "spread must be one of {'none', 'sem', 'sd', 'bootstrap_ci', 'iqr'}."
    )


def _format_pvalue(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "nan"
    if pvalue < 1e-4:
        return "<1e-4"
    return f"{pvalue:.4f}"


def _safe_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d with safeguards for degenerate variance."""
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
    """Build compact pairwise significance/effect-size annotation text."""
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
        lines.append(
            f"{name}: {test_name} p={_format_pvalue(float(pval))}, d={d:.3f}, stat={float(stat):.2f}"
        )

    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _draw_pairwise_annotation(ax, text: str) -> None:
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
    """Compute center/lower/upper across runs for each time index."""
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
    """Save and/or render figure with consistent behavior."""
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
    """
    Return first evaluation index where best_so_far reaches optimum (TTFO), else None.
    trajectory tuples are expected as (evaluations, current_value, best_value).
    """
    if f_star is None:
        return None

    for eval_idx, _current, best in trajectory:
        if abs(best - f_star) <= tolerance:
            return int(eval_idx)
    return None


def plot_simple_regret_curves(
    results: dict[tuple, list[dict]],
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
):
    """Plot mean regret vs budget for multiple algorithms."""

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
                regrets = np.array(
                    [r["regret"] for r in results[(alg, budget)]], dtype=float
                )
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
                text = _pairwise_stats_text(
                    dist_map, reference=reference, paired=paired_runs
                )
                if text:
                    _draw_pairwise_annotation(
                        ax,
                        f"At budget={comparison_budget}\n{text}",
                    )

    _finalize_figure(fig, save_path=save_path, show=show)


def plot_simple_regret_boxplots(
    results: dict[tuple, list[dict]],
    budget: int,
    save_path: str | None = None,
    title: str | None = None,
    show: bool = True,
    show_points: bool = True,
    annotate_pairwise: bool = False,
    reference_algorithm: str | None = None,
    paired_runs: bool = False,
):
    """Create boxplots comparing algorithms at a specific budget."""

    algorithms = sorted([alg for alg, b in results.keys() if b == budget])
    data = [
        np.array([r["regret"] for r in results[(alg, budget)]], dtype=float)
        for alg in algorithms
    ]

    fig, ax = plt.subplots()
    bp = ax.boxplot(data, tick_labels=algorithms, patch_artist=True, notch=True)

    colors = cm.get_cmap("Set3")(np.linspace(0, 1, len(algorithms)))
    for patch, color in zip(bp["boxes"], colors):
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
        distributions = {alg: vals for alg, vals in zip(algorithms, data)}
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
    results: dict[tuple, list[dict]],
    save_path: str | None = None,
    title: str | None = None,
    show: bool = True,
    show_confidence_band: bool = True,
    confidence: float = 0.95,
):
    """Plot probability of finding optimum vs budget."""

    algorithms = sorted(set(alg for alg, _ in results.keys()))
    budgets = sorted(set(budget for _, budget in results.keys()))

    fig, ax = plt.subplots()

    for alg in algorithms:
        probs = []
        lowers = []
        uppers = []
        for budget in budgets:
            if (alg, budget) in results:
                regrets = np.array(
                    [r["regret"] for r in results[(alg, budget)]], dtype=float
                )
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
    results: dict[tuple, list[dict]],
    save_path: str | None = None,
    show: bool = True,
):
    """Create heatmap of mean regrets across algorithms and budgets."""

    algorithms = sorted(set(alg for alg, _ in results.keys()))
    budgets = sorted(set(budget for _, budget in results.keys()))
    data = np.zeros((len(algorithms), len(budgets)))

    for i, alg in enumerate(algorithms):
        for j, budget in enumerate(budgets):
            if (alg, budget) in results:
                regrets = np.array(
                    [r["regret"] for r in results[(alg, budget)]], dtype=float
                )
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
    results: dict[tuple, list[dict]],
    budget: int,
    save_path: str | None = None,
    title: str | None = None,
    show: bool = True,
    annotate_pairwise: bool = False,
    reference_algorithm: str | None = None,
    paired_runs: bool = False,
):
    """Create performance profile (CDF of regrets)."""

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
    results: dict[str, list[dict]],
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
):
    """
    Plot history series vs evaluations.

    results maps algorithm name -> list of run dicts that include "trajectory".
    series: "current" or "best"

    TTFO helper markers:
        * vertical line at mean TTFO for each algorithm
        * point marker at corresponding mean series value
    """
    fig, ax = plt.subplots()

    for alg, runs in results.items():
        trajectories = [r["trajectory"] for r in runs if "trajectory" in r]
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
    results: dict[str, list[dict]],
    f_star: float,
    series: str = "instantaneous",
    use_best: bool = False,
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
):
    """
    Plot regret series vs evaluations.

    results maps algorithm name -> list of run dicts that include "trajectory".
    series: "instantaneous" or "cumulative"

    TTFO helper markers:
        * vertical line at mean TTFO for each algorithm
        * point marker at corresponding mean series value
    """
    fig, ax = plt.subplots()

    for alg, runs in results.items():
        trajectories = [r["trajectory"] for r in runs if "trajectory" in r]
        if not trajectories:
            continue

        series_list = []
        ttfo_values = []

        for trajectory in trajectories:
            if series == "instantaneous":
                series_points = instantaneous_regret(
                    trajectory, f_star, use_best=use_best
                )
            elif series == "cumulative":
                series_points = cumulative_regret(trajectory, f_star, use_best=use_best)
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
    results: dict[str, list[dict]],
    f_star: float | None,
    save_path: str | None = None,
    title: str | None = None,
    show: bool = True,
    tolerance: float = 1e-9,
    show_median: bool = True,
    annotate_pairwise: bool = False,
    reference_algorithm: str | None = None,
    paired_runs: bool = False,
):
    """
    Plot TTFO (time-to-first-optimum) distribution across algorithms.

    Creates a scatter plot showing individual TTFO samples for each algorithm,
    with optional median TTFO vertical lines.

    Args:
        results: Maps algorithm name -> list of run dicts containing "trajectory".
        f_star: Known optimum value. If None, the function returns without plotting.
        save_path: Path to save the figure. If None, figure is not saved.
        title: Plot title.
        show: Whether to display the plot interactively.
        tolerance: Tolerance for considering optimum reached.
        show_median: Whether to show median TTFO as vertical dashed lines.
    """
    if f_star is None:
        return

    fig, ax = plt.subplots()

    algorithms = sorted(results.keys())
    colors = cm.get_cmap("tab10")(np.linspace(0, 1, len(algorithms)))
    ttfo_map: dict[str, np.ndarray] = {}

    for idx, (alg, color) in enumerate(zip(algorithms, colors), start=1):
        runs = results[alg]
        ttfos = []

        for run in runs:
            trajectory = run.get("trajectory", [])
            if not trajectory:
                continue

            ttfo = _ttfo_for_trajectory(trajectory, f_star, tolerance=tolerance)
            if ttfo is not None:
                ttfos.append(ttfo)

        if not ttfos:
            continue

        ttfo_map[alg] = np.asarray(ttfos, dtype=float)

        # Place samples on algorithm rows with light jitter to show multiplicity.
        rng = np.random.default_rng(123 + idx)
        y = np.full(len(ttfos), fill_value=idx, dtype=float) + rng.uniform(
            -0.12, 0.12, size=len(ttfos)
        )
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
