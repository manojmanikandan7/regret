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

# Set publication-quality defaults
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 400


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
):
    """Plot mean regret vs budget for multiple algorithms."""

    algorithms = sorted(set(alg for alg, _ in results.keys()))
    budgets = sorted(set(budget for _, budget in results.keys()))

    fig, ax = plt.subplots()

    for alg in algorithms:
        mean_regrets = []
        std_regrets = []

        for budget in budgets:
            if (alg, budget) in results:
                regrets = np.array(
                    [r["regret"] for r in results[(alg, budget)]], dtype=float
                )
                mean_regrets.append(np.mean(regrets))
                std_regrets.append(np.std(regrets))
            else:
                mean_regrets.append(np.nan)
                std_regrets.append(np.nan)

        mean_regrets = np.array(mean_regrets, dtype=float)
        std_regrets = np.array(std_regrets, dtype=float)

        if log_scale:
            ax.loglog(budgets, mean_regrets, marker="o", label=alg, linewidth=2)
        else:
            ax.plot(budgets, mean_regrets, marker="o", label=alg, linewidth=2)

        ax.fill_between(
            budgets,
            mean_regrets - std_regrets,
            mean_regrets + std_regrets,
            alpha=0.2,
        )

    ax.set_xlabel("Budget (evaluations)")
    ax.set_ylabel("Mean Simple Regret")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    _finalize_figure(fig, save_path=save_path, show=show)


def plot_simple_regret_boxplots(
    results: dict[tuple, list[dict]],
    budget: int,
    save_path: str | None = None,
    title: str | None = None,
    show: bool = True,
):
    """Create boxplots comparing algorithms at a specific budget."""

    algorithms = sorted([alg for alg, b in results.keys() if b == budget])
    data = [
        np.array([r["regret"] for r in results[(alg, budget)]], dtype=float)
        for alg in algorithms
    ]

    fig, ax = plt.subplots()
    bp = ax.boxplot(data, tick_labels=algorithms, patch_artist=True)

    colors = cm.get_cmap("Set3")(np.linspace(0, 1, len(algorithms)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Simple Regret")
    ax.set_xlabel("Algorithm")
    ax.set_title(title if title else f"Regret Distribution at Budget = {budget}")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    _finalize_figure(fig, save_path=save_path, show=show)


def plot_convergence_probability(
    results: dict[tuple, list[dict]],
    save_path: str | None = None,
    title: str | None = None,
    show: bool = True,
):
    """Plot probability of finding optimum vs budget."""

    algorithms = sorted(set(alg for alg, _ in results.keys()))
    budgets = sorted(set(budget for _, budget in results.keys()))

    fig, ax = plt.subplots()

    for alg in algorithms:
        probs = []
        for budget in budgets:
            if (alg, budget) in results:
                regrets = np.array(
                    [r["regret"] for r in results[(alg, budget)]], dtype=float
                )
                probs.append(float(np.mean(regrets < 1e-9)))
            else:
                probs.append(np.nan)

        ax.semilogx(budgets, probs, marker="o", label=alg, linewidth=2)

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
):
    """Create performance profile (CDF of regrets)."""

    algorithms = sorted([alg for alg, b in results.keys() if b == budget])

    fig, ax = plt.subplots()

    for alg in algorithms:
        regrets = np.array([r["regret"] for r in results[(alg, budget)]], dtype=float)
        sorted_regrets = np.sort(regrets)
        cdf = np.arange(1, len(sorted_regrets) + 1) / len(sorted_regrets)
        ax.plot(sorted_regrets, cdf, label=alg, linewidth=2)

    ax.set_xlabel("Simple Regret")
    ax.set_ylabel("Cumulative Probability")
    ax.set_xscale("log")
    ax.set_title(title if title else f"Performance Profile (Budget = {budget})")
    ax.legend()
    ax.grid(True, alpha=0.3)

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
        mean_vals = np.nanmean(values, axis=0)
        std_vals = np.nanstd(values, axis=0)

        if log_x and log_y:
            (line,) = ax.loglog(times, mean_vals, marker=None, label=alg, linewidth=2)
        elif log_x:
            (line,) = ax.semilogx(times, mean_vals, marker=None, label=alg, linewidth=2)
        elif log_y:
            (line,) = ax.semilogy(times, mean_vals, marker=None, label=alg, linewidth=2)
        else:
            (line,) = ax.plot(times, mean_vals, marker=None, label=alg, linewidth=2)

        ax.fill_between(times, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)

        if show_ttfo_markers and ttfo_values:
            mean_ttfo = float(np.mean(ttfo_values))

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
        mean_vals = np.nanmean(values, axis=0)
        std_vals = np.nanstd(values, axis=0)

        if log_x and log_y:
            (line,) = ax.loglog(times, mean_vals, marker=None, label=alg, linewidth=2)
        elif log_x:
            (line,) = ax.semilogx(times, mean_vals, marker=None, label=alg, linewidth=2)
        elif log_y:
            (line,) = ax.semilogy(times, mean_vals, marker=None, label=alg, linewidth=2)
        else:
            (line,) = ax.plot(times, mean_vals, marker=None, label=alg, linewidth=2)

        ax.fill_between(times, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)

        if show_ttfo_markers and ttfo_values:
            mean_ttfo = float(np.mean(ttfo_values))

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
):
    """
    Plot TTFO (time-to-first-optimum) distribution across algorithms.

    Creates a scatter plot showing individual TTFO samples for each algorithm,
    with optional median TTFO vertical lines.

    Parameters
    ----------
    results : dict[str, list[dict]]
        Maps algorithm name -> list of run dicts containing "trajectory".
    f_star : float | None
        Known optimum value. If None, the function returns without plotting.
    save_path : str | None
        Path to save the figure. If None, figure is not saved.
    title : str | None
        Plot title.
    show : bool
        Whether to display the plot interactively.
    tolerance : float
        Tolerance for considering optimum reached.
    show_median : bool
        Whether to show median TTFO as vertical dashed lines.
    """
    if f_star is None:
        return

    fig, ax = plt.subplots()

    algorithms = sorted(results.keys())
    colors = cm.get_cmap("tab10")(np.linspace(0, 1, len(algorithms)))

    for alg, color in zip(algorithms, colors):
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

        # Scatter TTFO samples at y = f_star
        y = np.full(len(ttfos), fill_value=f_star)
        ax.scatter(
            ttfos,
            y,
            s=24,
            alpha=0.6,
            color=color,
            label=f"{alg} TTFO samples",
        )

        if show_median:
            ax.axvline(
                x=float(np.median(ttfos)),
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
            )

    ax.axhline(
        f_star, color="black", linestyle=":", linewidth=1.0, alpha=0.8, label="f*"
    )
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Objective value")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    _finalize_figure(fig, save_path=save_path, show=show)
