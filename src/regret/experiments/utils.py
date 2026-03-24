"""
Some utilities used for parsing the config YAML files, and handling plotting, etc.

Note: These utilities respect the schema given by `./schema.py` for parsing the configuration.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast
from regret.core.base import Problem, Algorithm
from regret.analysis.plotting import (
    plot_comparison_heatmap,
    plot_convergence_probability,
    plot_history,
    plot_performance_profile,
    plot_regret_curves,
    plot_runtime_profile_curves,
    plot_runtime_profile_surface,
    plot_cr_profile_verification,
    plot_simple_regret_boxplots,
    plot_simple_regret_curves,
    plot_ttfo_distribution,
)
from . import PROBLEM_REGISTRY, ALGORITHM_REGISTRY, COOLING_REGISTRY


@dataclass(frozen=True)
class ProblemSpec:
    """Specification for a problem instance."""

    name: str
    class_name: str
    params: dict[str, Any]
    budget_for_plots: int


@dataclass(frozen=True)
class AlgorithmSpec:
    """Specification for an algorithm configuration."""

    name: str
    class_name: str
    args: dict[str, Any]


def safe_slug(text: str) -> str:
    """Convert text to filesystem-safe slug."""
    replacements = [
        (" ", "_"),
        ("+", "plus"),
        ("(", ""),
        (")", ""),
        ("/", "_"),
        ("-", "_"),
    ]
    result = text.strip().lower()
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def is_plot_enabled(
    plots_cfg: dict[str, Any],
    plot_key: str,
    default: bool = True,
) -> bool:
    """Return whether a plot is enabled in config with a default fallback."""
    if plot_key not in plots_cfg:
        return default
    cfg = plots_cfg[plot_key]
    if not isinstance(cfg, dict):
        return default
    return bool(cfg.get("enabled", default))


def get_plot_filename(
    plots_cfg: dict[str, Any],
    plot_key: str,
    default: str,
) -> str:
    """Return configured filename for a plot key or a safe default."""
    if plot_key not in plots_cfg:
        return default
    cfg = plots_cfg[plot_key]
    if not isinstance(cfg, dict):
        return default
    value = cfg.get("filename", default)
    return value if isinstance(value, str) else default


def merge_plot_title(prefix: str | None, subject: str) -> str:
    """Compose a plot title from optional prefix and subject."""
    if not prefix:
        return subject
    if not subject:
        return prefix
    return f"{prefix}: {subject}"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_cooling(spec: Any) -> Callable[[int], float]:
    """Resolve cooling schedule specification to a callable T(t) function."""
    if callable(spec):
        # Patch: prevents linter warnings
        return cast(Callable[[int], float], spec)

    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in COOLING_REGISTRY:
            return COOLING_REGISTRY[key]()
        raise ValueError(
            f"Unknown cooling schedule: {spec!r}. "
            f"Available: {sorted(COOLING_REGISTRY.keys())}"
        )

    if isinstance(spec, dict):
        kind = str(spec.get("type", "logarithmic")).strip().lower()
        params = spec.get("params", {})
        if kind in COOLING_REGISTRY:
            return COOLING_REGISTRY[kind](**params)
        raise ValueError(
            f"Unknown cooling type: {spec.get('type')!r}. "
            f"Available: {sorted(COOLING_REGISTRY.keys())}"
        )

    raise ValueError(f"Unsupported cooling specification: {spec!r}")


def get_algorithm_class(class_name: str) -> type[Algorithm]:
    """Look up algorithm class from registry."""
    if class_name not in ALGORITHM_REGISTRY:
        raise KeyError(f"Unknown algorithm class: {class_name}")
    return ALGORITHM_REGISTRY[class_name]


def get_problem_class(class_name: str) -> type[Problem]:
    """Look up problem class from registry."""
    if class_name not in PROBLEM_REGISTRY:
        raise KeyError(f"Unknown problem class: {class_name}")
    return PROBLEM_REGISTRY[class_name]


def instantiate_problem(spec: ProblemSpec) -> Problem:
    """Instantiate a problem from its specification."""
    cls = get_problem_class(spec.class_name)
    return cls(**spec.params)


def parse_problems(config: dict[str, Any]) -> list[ProblemSpec]:
    """Parse problem specifications from config.

    Args:
        config: Validated config dictionary with top-level 'problems' key.

    Returns:
        List of ProblemSpec objects.
    """
    problems_cfg = config.get("problems", [])
    global_plot_budget = config["plotting"].get("budget_for_plots")
    max_budget = max(config["suite"]["budgets"])

    if not problems_cfg:
        raise ValueError("No problems configured in config")

    prob_specs = []
    for p in problems_cfg:
        budget_for_plots = int(
            p.get("budget_for_plots") or global_plot_budget or max_budget
        )
        prob_specs.append(
            ProblemSpec(
                name=p["name"],
                class_name=p["class"],
                params=p.get("params", {}),
                budget_for_plots=budget_for_plots,
            )
        )

    return prob_specs


def parse_algorithms(config: dict[str, Any]) -> list[AlgorithmSpec]:
    """Parse algorithm specifications from config.

    Args:
        config: Validated config dictionary with top-level 'algorithms' key.

    Returns:
        List of AlgorithmSpec objects.
    """
    algorithms_cfg = config.get("algorithms", [])
    if not algorithms_cfg:
        raise ValueError("No algorithms configured in config")

    specs = []
    for a in algorithms_cfg:
        args_cfg = a.get("args", {})
        if not isinstance(args_cfg, dict):
            raise ValueError(
                f"Algorithm '{a.get('name', '<unknown>')}' args must be a mapping"
            )
        specs.append(
            AlgorithmSpec(
                name=a["name"],
                class_name=a["class"],
                args={
                    "defaults": args_cfg.get("defaults", {}),
                    "by_problem": args_cfg.get("by_problem", {}),
                },
            )
        )
    return specs


def resolve_algorithm_args(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Process algorithm arguments, resolving cooling schedules."""
    args = raw_args.copy()

    # Normalize "cooling" key to "T_func"
    if "cooling" in args and "T_func" not in args:
        args["T_func"] = args.pop("cooling")

    if "T_func" in args:
        args["T_func"] = resolve_cooling(args["T_func"])

    return args


def resolve_alg_kwargs(spec: AlgorithmSpec, problem_name: str) -> dict[str, Any]:
    """Resolve algorithm kwargs by merging defaults with problem-specific overrides."""
    defaults = spec.args.get("defaults", {})
    override = spec.args.get("by_problem", {}).get(problem_name, {})
    return resolve_algorithm_args(deep_merge(defaults, override))


def history_view_at_budget(
    keyed_results: dict[tuple[str, int], list[dict[str, Any]]], budget: int
) -> dict[str, list[dict[str, Any]]]:
    """Extract algorithm-to-runs mapping for one budget."""
    return {alg: runs for (alg, b), runs in keyed_results.items() if b == budget}


def write_runtime_profile_csv(
    output_dir: Path,
    time_grid: Any,
    empirical_ecr: dict[str, Any],
    profile_ecr: dict[str, Any],
) -> None:
    """Persist runtime-profile CR comparison series to CSV files.

    One CSV is written per algorithm with columns:
    evaluation, expected_cumulative_regret, mean_cumulative_regret.
    """
    for alg_name in sorted(set(empirical_ecr).intersection(profile_ecr)):
        csv_path = output_dir / f"cr_profile_data_{safe_slug(alg_name)}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "evaluation",
                    "expected_cumulative_regret",
                    "mean_cumulative_regret",
                ]
            )
            for t, expected_cr, mean_cr in zip(
                time_grid,
                profile_ecr[alg_name],
                empirical_ecr[alg_name],
            ):
                writer.writerow([float(t), float(expected_cr), float(mean_cr)])


def export_runtime_profile_data(
    suite_name: str,
    problem_name: str,
    n: int,
    f_star: float | None,
    results: dict[tuple[str, int], list[dict[str, Any]]],
    budget_for_plots: int,
    plotting_config: dict[str, Any],
    raw_output_dir: Path | str = "results/raw",
    figures_output_dir: Path | str | None = None,
) -> None:
    """Export runtime-profile CSV artifacts and optional profile plots.

    CSV files are written under:
    <raw_output_dir>/<suite>/<problem>/n<n>/<profile_subdir>/cr_profile_data_<alg>.csv

    If figures_output_dir is provided, runtime-profile plots are also saved under:
    <figures_output_dir>/<suite>/<problem>/n<n>/<profile_subdir>/
    """
    if f_star is None:
        return

    history_results = history_view_at_budget(results, budget_for_plots)
    if not history_results:
        return

    layout_cfg = plotting_config.get("layout", {})
    structure = layout_cfg.get("structure", {})
    per_problem_dir = bool(layout_cfg.get("per_problem_dir", True))
    include_n_subdir = bool(layout_cfg.get("include_n_subdir", True))
    plots_cfg = plotting_config.get("plots", {})
    titles_cfg = plotting_config.get("titles", {})

    include_problem_name = bool(titles_cfg.get("include_problem_name", True))
    include_problem_size = bool(titles_cfg.get("include_problem_size", True))
    title_prefix = " - ".join(
        part
        for part in [
            problem_name if include_problem_name else "",
            f"n={n}" if include_problem_size else "",
        ]
        if part
    )

    def _profile_dir(base_output_dir: Path | str) -> Path:
        base_dir = Path(base_output_dir) / safe_slug(suite_name)
        if per_problem_dir:
            base_dir = base_dir / safe_slug(problem_name)
        if include_n_subdir:
            base_dir = base_dir / f"n{n}"
        return base_dir / structure.get("profile", "profiles")

    raw_profile_dir = _profile_dir(raw_output_dir)
    raw_profile_dir.mkdir(parents=True, exist_ok=True)

    from regret.analysis.profiles import run_profile_analysis

    profile_plot_keys = [
        "runtime_profile_surface",
        "runtime_profile_curves",
        "cr_profile_verification",
    ]
    profile_cfg = {k: plots_cfg.get(k, {}) for k in profile_plot_keys}

    time_grid, fitness_levels, profiles, empirical_ecr, profile_ecr = (
        run_profile_analysis(
            results=history_results,
            f_star=f_star,
            budget=budget_for_plots,
        )
    )

    if empirical_ecr and profile_ecr:
        write_runtime_profile_csv(
            output_dir=raw_profile_dir,
            time_grid=time_grid,
            empirical_ecr=empirical_ecr,
            profile_ecr=profile_ecr,
        )

    if figures_output_dir is None:
        return

    figures_profile_dir = _profile_dir(figures_output_dir)
    figures_profile_dir.mkdir(parents=True, exist_ok=True)

    if is_plot_enabled(profile_cfg, "runtime_profile_surface"):
        for alg_name, profile in profiles.items():
            plot_runtime_profile_surface(
                profile=profile,
                fitness_levels=fitness_levels,
                time_grid=time_grid,
                f_star=f_star,
                save_path=str(
                    figures_profile_dir
                    / get_plot_filename(
                        profile_cfg,
                        "runtime_profile_surface",
                        "runtime_profile_surface_{algorithm}.pdf",
                    ).format(algorithm=alg_name),
                ),
                show=False,
                title=merge_plot_title(
                    title_prefix,
                    f"{alg_name}: runtime profile surface",
                ),
            )

    if is_plot_enabled(profile_cfg, "runtime_profile_curves") and profiles:
        selected_levels = [f_star * q for q in [0.25, 0.5, 0.75, 0.95]]
        plot_runtime_profile_curves(
            profiles=profiles,
            fitness_levels=fitness_levels,
            time_grid=time_grid,
            selected_levels=selected_levels,
            f_star=f_star,
            save_path=str(
                figures_profile_dir
                / get_plot_filename(
                    profile_cfg,
                    "runtime_profile_curves",
                    "runtime_profile_curves.pdf",
                ),
            ),
            show=False,
            title=merge_plot_title(title_prefix, "$P(\\tau_v \\leq T)$ by algorithm"),
        )

    if (
        empirical_ecr
        and profile_ecr
        and is_plot_enabled(profile_cfg, "cr_profile_verification")
    ):
        plot_cr_profile_verification(
            empirical_ecr=empirical_ecr,
            profile_ecr=profile_ecr,
            time_grid=time_grid,
            save_path=str(
                figures_profile_dir
                / get_plot_filename(
                    profile_cfg,
                    "cr_profile_verification",
                    "cr_profile_verification.pdf",
                ),
            ),
            show=False,
            title=merge_plot_title(
                title_prefix,
                "E[CR(T)]: direct vs profile identity",
            ),
        )


def generate_plots(
    suite_name: str,
    problem_name: str,
    n: int,
    f_star: float | None,
    results: dict[tuple[str, int], list[dict[str, Any]]],
    budget_for_plots: int,
    plotting_config: dict[str, Any],
    output_dir: Path | str = "results/figures",
) -> None:
    """Generate plots for a problem's results based on config.

    Args:
        suite_name: Name of the experiment suite.
        problem_name: Name of the problem.
        n: Problem size.
        f_star: Known optimum value, or None if unknown.
        results: Mapping of (algorithm, budget) -> list of run result dicts.
        max_budget: Maximum budget to use for budget-specific plots.
        budget_for_plots: Optional budget for budget-specific plots. If None,
            uses max_budget.
        output_dir: Base directory for saving figures.
        plotting_config: Optional plotting configuration dict from YAML.
            If None, generates all plots with default settings.
    """

    available_budgets = {int(b) for _, b in results.keys()}
    if budget_for_plots not in available_budgets:
        raise ValueError(
            "Selected plotting budget is not available in results: "
            f"budget={budget_for_plots}, available={sorted(available_budgets)}"
        )

    # Extract plot/layout/title configs with safe defaults.
    plots_cfg = plotting_config.get("plots", {})
    layout_cfg = plotting_config.get("layout", {})
    titles_cfg = plotting_config.get("titles", {})

    per_problem_dir = bool(layout_cfg.get("per_problem_dir", True))
    include_n_subdir = bool(layout_cfg.get("include_n_subdir", True))
    include_problem_name = bool(titles_cfg.get("include_problem_name", True))
    include_problem_size = bool(titles_cfg.get("include_problem_size", True))

    TITLE_PREFIX = " - ".join(
        part
        for part in [
            problem_name if include_problem_name else "",
            f"n={n}" if include_problem_size else "",
        ]
        if part
    )

    base_dir = Path(output_dir) / safe_slug(suite_name)
    if per_problem_dir:
        base_dir = base_dir / safe_slug(problem_name)
    if include_n_subdir:
        base_dir = base_dir / f"n{n}"

    # Define output directories from config or defaults
    structure = layout_cfg.get("structure", {})
    dirs = {
        "aggregate": base_dir / structure.get("aggregate", "aggregate"),
        "history": base_dir / structure.get("history", "history"),
        "distribution": base_dir
        / structure.get("distribution", f"budget_{budget_for_plots}"),
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    def _optional_kwargs(cfg: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        """Forward only explicitly configured plotting kwargs."""
        return {k: cfg[k] for k in keys if k in cfg}

    # Aggregate plots (across budgets)
    if is_plot_enabled(plots_cfg, "regret_curves"):
        cfg = plots_cfg["regret_curves"]
        plot_simple_regret_curves(
            results,
            save_path=str(
                dirs["aggregate"]
                / get_plot_filename(plots_cfg, "regret_curves", "regret_curves.pdf")
            ),
            title=merge_plot_title(TITLE_PREFIX, "Mean Simple Regret vs Budget"),
            log_scale=cfg.get("log_scale", True),
            **_optional_kwargs(
                cfg,
                [
                    "spread",
                    "confidence",
                    "n_bootstrap",
                    "annotate_pairwise",
                    "comparison_budget",
                    "paired_runs",
                ],
            ),
            show=False,
        )

    if is_plot_enabled(plots_cfg, "convergence_probability"):
        cfg = plots_cfg["convergence_probability"]
        plot_convergence_probability(
            results,
            save_path=str(
                dirs["aggregate"]
                / get_plot_filename(
                    plots_cfg, "convergence_probability", "convergence_probability.pdf"
                )
            ),
            title=merge_plot_title(TITLE_PREFIX, "Probability of Optimum vs Budget"),
            **_optional_kwargs(cfg, ["show_confidence_band", "confidence"]),
            show=False,
        )

    if is_plot_enabled(plots_cfg, "comparison_heatmap"):
        plot_comparison_heatmap(
            results,
            save_path=str(
                dirs["aggregate"]
                / get_plot_filename(
                    plots_cfg,
                    "comparison_heatmap",
                    "comparison_heatmap.pdf",
                )
            ),
            show=False,
        )

    # Budget-specific plots
    if is_plot_enabled(plots_cfg, "regret_boxplots"):
        cfg = plots_cfg["regret_boxplots"]
        filename = cfg.get("filename", "regret_boxplot_b{budget}.pdf").format(
            budget=budget_for_plots
        )
        plot_simple_regret_boxplots(
            results,
            budget=budget_for_plots,
            save_path=str(dirs["distribution"] / filename),
            title=merge_plot_title(
                TITLE_PREFIX, f"Regret Distribution at Budget={budget_for_plots}"
            ),
            **_optional_kwargs(
                cfg,
                [
                    "show_points",
                    "annotate_pairwise",
                    "reference_algorithm",
                    "paired_runs",
                ],
            ),
            show=False,
        )

    if is_plot_enabled(plots_cfg, "performance_profile"):
        cfg = plots_cfg["performance_profile"]
        filename = cfg.get("filename", "performance_profile_b{budget}.pdf").format(
            budget=budget_for_plots
        )
        plot_performance_profile(
            results,
            budget=budget_for_plots,
            save_path=str(dirs["distribution"] / filename),
            title=merge_plot_title(
                TITLE_PREFIX, f"Performance Profile at Budget={budget_for_plots}"
            ),
            **_optional_kwargs(
                cfg,
                ["annotate_pairwise", "reference_algorithm", "paired_runs"],
            ),
            show=False,
        )

    # History plots (require trajectory data)
    history_results = history_view_at_budget(results, budget_for_plots)
    if not history_results:
        return

    # Value history plots (plot_history function)
    # These plot "current" or "best" series
    history_plot_keys = ["history_current", "history_best"]
    for plot_key in history_plot_keys:
        if is_plot_enabled(plots_cfg, plot_key) and f_star is not None:
            cfg = plots_cfg[plot_key]
            series = cfg.get("series", "current" if "current" in plot_key else "best")
            default_filename = f"{plot_key}.pdf"
            plot_history(
                history_results,
                f_star=f_star,
                save_path=str(
                    dirs["history"]
                    / get_plot_filename(plots_cfg, plot_key, default_filename)
                ),
                title=merge_plot_title(
                    TITLE_PREFIX,
                    f"{series.title()} Value Trajectory at Budget={budget_for_plots}",
                ),
                series=series,
                log_x=cfg.get("log_x", False),
                log_y=cfg.get("log_y", False),
                show_ttfo_markers=cfg.get("show_ttfo_markers", True),
                **_optional_kwargs(cfg, ["spread", "confidence", "n_bootstrap"]),
                show=False,
            )

    # Regret plots (plot_regret_curves function)
    # These plot "instantaneous" or "cumulative" regret series
    regret_plot_keys = [
        "regret_instantaneous",
        "regret_cumulative",
        "regret_instantaneous_best",
        "regret_cumulative_best",
    ]
    for plot_key in regret_plot_keys:
        if is_plot_enabled(plots_cfg, plot_key) and f_star is not None:
            cfg = plots_cfg[plot_key]
            # Infer series from key name
            if "instantaneous" in plot_key:
                default_series = "instantaneous"
            else:
                default_series = "cumulative"
            series = cfg.get("series", default_series)
            use_best = cfg.get("use_best", "best" in plot_key)
            default_filename = f"history_{plot_key}.pdf"

            title_suffix = f"{series.title()} Regret"
            if use_best:
                title_suffix += " (best)"

            plot_regret_curves(
                history_results,
                f_star=f_star,
                save_path=str(
                    dirs["history"]
                    / get_plot_filename(plots_cfg, plot_key, default_filename)
                ),
                title=merge_plot_title(
                    TITLE_PREFIX,
                    f"{title_suffix} Trajectory at Budget={budget_for_plots}",
                ),
                series=series,
                use_best=use_best,
                log_x=cfg.get("log_x", False),
                log_y=cfg.get("log_y", True),
                show_ttfo_markers=cfg.get("show_ttfo_markers", True),
                **_optional_kwargs(cfg, ["spread", "confidence", "n_bootstrap"]),
                show=False,
            )

    # TTFO distribution plot
    if is_plot_enabled(plots_cfg, "ttfo_distribution"):
        cfg = plots_cfg["ttfo_distribution"]
        plot_ttfo_distribution(
            results=history_results,
            f_star=f_star,
            save_path=str(
                dirs["history"]
                / get_plot_filename(
                    plots_cfg,
                    "ttfo_distribution",
                    "history_ttfo_markers.pdf",
                )
            ),
            title=merge_plot_title(
                TITLE_PREFIX, f"TTFO Samples at Budget={budget_for_plots}"
            ),
            **_optional_kwargs(
                cfg,
                [
                    "tolerance",
                    "show_median",
                    "annotate_pairwise",
                    "reference_algorithm",
                    "paired_runs",
                ],
            ),
            show=False,
        )
