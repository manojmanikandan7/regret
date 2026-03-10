"""
Orchestration module for executing validated experiment configurations.

Assumes config is valid and handles only execution logic.
"""

from pathlib import Path
from typing import Any

from .utils import (
    get_algorithm_class,
    instantiate_problem,
    parse_algorithms,
    parse_problems,
    resolve_alg_kwargs,
    safe_slug,
    generate_plots,
)

from regret.experiments.runner import ExperimentRunner


def plan_execution(config: dict[str, Any]) -> dict[str, Any]:
    """Build execution plan summary from validated config.

    Args:
        config: Validated config dictionary.

    Returns:
        Dictionary with execution plan details:
        - suite_name: Name of the suite
        - runs_per_combo: Number of runs per configuration
        - budgets: List of budgets
        - problems: List of problem names
        - algorithms: List of algorithm names
        - combination_count: Total combinations
        - total_runs: Total number of runs
    """
    suite_cfg = config["suite"]
    problem_specs = parse_problems(config)
    algorithm_specs = parse_algorithms(config)
    budgets = [int(b) for b in suite_cfg["budgets"]]
    runs = int(suite_cfg["runs"])

    combo_count = len(problem_specs) * len(algorithm_specs) * len(budgets)
    total_runs = combo_count * runs

    return {
        "suite_name": suite_cfg["name"],
        "runs_per_combo": runs,
        "budgets": budgets,
        "problems": [p.name for p in problem_specs],
        "algorithms": [a.name for a in algorithm_specs],
        "combination_count": combo_count,
        "total_runs": total_runs,
        "mode": suite_cfg["mode"],
        "parallel": suite_cfg.get("parallel", True),
    }


def execute_experiments(config: dict[str, Any], plot: bool = True) -> None:
    """Execute experiments from validated config.

    Args:
        config: Validated config dictionary.
        plot: Whether to generate plots after execution (default: True).
    """
    suite_cfg = config["suite"]
    suite_name = suite_cfg["name"]
    runs = int(suite_cfg["runs"])
    mode = suite_cfg["mode"]
    parallel = suite_cfg.get("parallel", True)
    budgets = [int(b) for b in suite_cfg["budgets"]]

    problem_specs = parse_problems(config)
    algorithm_specs = parse_algorithms(config)
    runner = ExperimentRunner()

    for problem_spec in problem_specs:
        problem = instantiate_problem(problem_spec)
        all_results = {}

        for alg_spec in algorithm_specs:
            alg_class = get_algorithm_class(alg_spec.class_name)
            alg_kwargs = resolve_alg_kwargs(alg_spec, problem_spec.name)

            for budget in budgets:
                exp_name = "/".join(
                    [
                        safe_slug(suite_name),
                        safe_slug(problem_spec.name),
                        safe_slug(alg_spec.name),
                        f"n{problem.n}",
                        f"b{budget}",
                    ]
                )
                print(
                    f"[run] problem={problem_spec.name} alg={alg_spec.name} "
                    f"n={problem.n} budget={budget} runs={runs}"
                )

                results = runner.run_experiment(
                    alg_class,
                    problem,
                    budget=budget,
                    runs=runs,
                    mode=mode,
                    name=exp_name,
                    parallel=parallel,
                    **alg_kwargs,
                )
                all_results[(alg_spec.name, budget)] = results

        # Generate plots if requested
        if plot and config.get("plotting", {}).get("enabled", True):
            figures_root = suite_cfg.get("output", {}).get(
                "figures_root", "results/figures"
            )
            generate_plots(
                suite_name=suite_name,
                problem_name=problem_spec.name,
                n=problem.n,
                f_star=getattr(problem, "f_star", None),
                results=all_results,
                max_budget=max(budgets),
                output_dir=figures_root,
                plotting_config=config.get("plotting"),
            )


def analyze_results(config: dict[str, Any], results_dir: Path | None = None) -> None:
    """Regenerate plots from existing experiment results.

    Args:
        config: Validated config dictionary.
        results_dir: Optional path to existing results directory.
                     If None, uses default from config.

    Raises:
        NotImplementedError: Analysis-only mode not yet implemented.
    """
    # TODO: Implement standalone analysis from saved results
    raise NotImplementedError("Analysis-only mode will be implemented in the future.")
