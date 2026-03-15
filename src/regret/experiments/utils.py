"""
Some utilities used for parsing the config YAML files, and handling plotting, etc.

Note: These utilities respect the schema given by `./schema.py` for parsing the configuration.
"""

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
    if not problems_cfg:
        raise ValueError("No problems configured in config")

    return [
        ProblemSpec(
            name=p.get("name", p["class"]),
            class_name=p["class"],
            params=p.get("params", {}),
        )
        for p in problems_cfg
    ]


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
                f"Algorithm '{a.get('name', a.get('class', '<unknown>'))}' args must be a mapping"
            )
        specs.append(
            AlgorithmSpec(
                name=a.get("name", a["class"]),
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


def generate_plots(
    suite_name: str,
    problem_name: str,
    n: int,
    f_star: float | None,
    results: dict[tuple[str, int], list[dict[str, Any]]],
    max_budget: int,
    output_dir: Path | str = "results/figures",
    plotting_config: dict[str, Any] | None = None,
) -> None:
    """Generate plots for a problem's results based on config.

    Args:
        suite_name: Name of the experiment suite.
        problem_name: Name of the problem.
        n: Problem size.
        f_star: Known optimum value, or None if unknown.
        results: Mapping of (algorithm, budget) -> list of run result dicts.
        max_budget: Maximum budget to use for budget-specific plots.
        output_dir: Base directory for saving figures.
        plotting_config: Optional plotting configuration dict from YAML.
            If None, generates all plots with default settings.
    """
    base_dir = (
        Path(output_dir) / safe_slug(suite_name) / safe_slug(problem_name) / f"n{n}"
    )

    # Extract plot configs, defaulting to enabled if not specified
    plots_cfg = (plotting_config or {}).get("plots", {})
    layout_cfg = (plotting_config or {}).get("layout", {})

    # Define output directories from config or defaults
    structure = layout_cfg.get("structure", {})
    dirs = {
        "aggregate": base_dir / structure.get("aggregate", "aggregate"),
        "history": base_dir / structure.get("history", "history"),
        "budget": base_dir / f"budget_{max_budget}",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    def _is_enabled(plot_key: str) -> bool:
        """Check if a plot is enabled in config (default: True)."""
        return plots_cfg.get(plot_key, {}).get("enabled", True)

    def _get_filename(plot_key: str, default: str) -> str:
        """Get filename from config or use default."""
        return plots_cfg.get(plot_key, {}).get("filename", default)

    def _format_title(problem: str, budget: int, template: str | None = None) -> str:
        """Format title with problem name and budget."""
        if template:
            return template.format(problem=problem, budget=budget)
        return f"{problem} at Budget={budget}"

    # Aggregate plots (across budgets)
    if _is_enabled("regret_curves"):
        cfg = plots_cfg.get("regret_curves", {})
        plot_simple_regret_curves(
            results,
            save_path=str(
                dirs["aggregate"] / _get_filename("regret_curves", "regret_curves.pdf")
            ),
            title=f"{problem_name}: Mean Simple Regret vs Budget",
            log_scale=cfg.get("log_scale", True),
            show=False,
        )

    if _is_enabled("convergence_probability"):
        plot_convergence_probability(
            results,
            save_path=str(
                dirs["aggregate"]
                / _get_filename(
                    "convergence_probability", "convergence_probability.pdf"
                )
            ),
            title=f"{problem_name}: Probability of Optimum vs Budget",
            show=False,
        )

    if _is_enabled("comparison_heatmap"):
        plot_comparison_heatmap(
            results,
            save_path=str(
                dirs["aggregate"]
                / _get_filename("comparison_heatmap", "comparison_heatmap.pdf")
            ),
            show=False,
        )

    # Budget-specific plots
    if _is_enabled("regret_boxplots"):
        cfg = plots_cfg.get("regret_boxplots", {})
        filename = cfg.get("filename", "regret_boxplot_b{budget}.pdf").format(
            budget=max_budget
        )
        plot_simple_regret_boxplots(
            results,
            budget=max_budget,
            save_path=str(dirs["budget"] / filename),
            title=f"{problem_name}: Regret Distribution at Budget={max_budget}",
            show=False,
        )

    if _is_enabled("performance_profile"):
        cfg = plots_cfg.get("performance_profile", {})
        filename = cfg.get("filename", "performance_profile_b{budget}.pdf").format(
            budget=max_budget
        )
        plot_performance_profile(
            results,
            budget=max_budget,
            save_path=str(dirs["budget"] / filename),
            title=f"{problem_name}: Performance Profile at Budget={max_budget}",
            show=False,
        )

    # History plots (require trajectory data)
    history_results = history_view_at_budget(results, max_budget)
    if not history_results:
        return

    # Value history plots (plot_history function)
    # These plot "current" or "best" series
    history_plot_keys = ["history_current", "history_best"]
    for plot_key in history_plot_keys:
        if _is_enabled(plot_key) and f_star is not None:
            cfg = plots_cfg.get(plot_key, {})
            series = cfg.get("series", "current" if "current" in plot_key else "best")
            default_filename = f"{plot_key}.pdf"
            plot_history(
                history_results,
                f_star=f_star,
                save_path=str(
                    dirs["history"] / _get_filename(plot_key, default_filename)
                ),
                title=f"{problem_name}: {series.title()} Value Trajectory at Budget={max_budget}",
                series=series,
                log_x=cfg.get("log_x", False),
                log_y=cfg.get("log_y", False),
                show_ttfo_markers=cfg.get("show_ttfo_markers", True),
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
        if _is_enabled(plot_key) and f_star is not None:
            cfg = plots_cfg.get(plot_key, {})
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
                    dirs["history"] / _get_filename(plot_key, default_filename)
                ),
                title=f"{problem_name}: {title_suffix} Trajectory at Budget={max_budget}",
                series=series,
                use_best=use_best,
                log_x=cfg.get("log_x", False),
                log_y=cfg.get("log_y", True),
                show_ttfo_markers=cfg.get("show_ttfo_markers", True),
                show=False,
            )

    # TTFO distribution plot
    if _is_enabled("ttfo_distribution"):
        cfg = plots_cfg.get("ttfo_distribution", {})
        plot_ttfo_distribution(
            results=history_results,
            f_star=f_star,
            save_path=str(
                dirs["history"]
                / _get_filename("ttfo_distribution", "history_ttfo_markers.pdf")
            ),
            title=f"{problem_name}: TTFO Samples at Budget={max_budget}",
            show=False,
        )
