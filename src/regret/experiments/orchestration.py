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
                budget_for_plots=config.get("plotting", {}).get("budget_for_plots"),
                output_dir=figures_root,
                plotting_config=config.get("plotting"),
            )


def analyze_results(config: dict[str, Any]) -> None:
    """Regenerate plots from existing experiment results.

    Args:
        config: Validated config dictionary.

    Raises:
        FileNotFoundError: If results directory or files not found.
    """
    import json
    from collections import defaultdict

    # Determine results directory
    results_dir = Path(config["suite"].get("output", {}).get("raw_root", "results/raw"))

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    suite_name = config["suite"]["name"]
    problem_specs = parse_problems(config)
    algorithm_specs = parse_algorithms(config)

    # Map slugs back to configured display names to preserve plot legends/labels.
    problem_slug_to_name = {safe_slug(p.name): p.name for p in problem_specs}
    algorithm_slug_to_name = {safe_slug(a.name): a.name for a in algorithm_specs}

    # Scan for JSON files and group by (problem_name, problem_size)
    results_by_problem = defaultdict(dict)
    f_star_by_problem = {}
    json_files = list(results_dir.glob("**/*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON results found in {results_dir}")

    print(f"[analyze] Found {len(json_files)} result files in {results_dir}")

    # Load and parse all results
    for json_file in json_files:
        # Get path components for determining algorithm names
        rel_parts = json_file.relative_to(results_dir).parts

        with open(json_file, "r") as f:
            data = json.load(f)

        metadata = data["metadata"]
        statistics = data["statistics"]
        results = data["results"]

        # Path shape:
        #   <suite>/<problem>/<algorithm>/nX/bY.json
        problem_slug = rel_parts[1]
        algorithm_slug = rel_parts[2]

        problem_name = problem_slug_to_name.get(problem_slug, metadata["problem"])
        algorithm_name = algorithm_slug_to_name.get(
            algorithm_slug, metadata["algorithm"]
        )
        n = int(metadata["problem_size"])
        budget = int(metadata["budget"])

        problem_key = (problem_name, n)
        results_by_problem[problem_key][(algorithm_name, budget)] = results
        f_star_by_problem[problem_key] = statistics.get("global_optimum")

    # For each problem group, regenerate plots
    for (problem_name, n), alg_budget_results in sorted(results_by_problem.items()):
        print(f"\n[analyze] Generating plots for {problem_name} (n={n})")

        # Use saved optimum from raw results when available.
        f_star = f_star_by_problem.get((problem_name, n))

        # Find max budget for this problem
        max_budget = max(budget for _, budget in alg_budget_results.keys())

        # Prepare results in the format expected by generate_plots
        # Format: {(algorithm, budget): [list of run dicts]}
        generate_plots(
            suite_name=suite_name,
            problem_name=problem_name,
            n=n,
            f_star=f_star,
            results=alg_budget_results,
            max_budget=max_budget,
            budget_for_plots=config.get("plotting", {}).get("budget_for_plots"),
            output_dir=config["suite"]
            .get("output", {})
            .get("figures_root", "results/figures"),
            plotting_config=config.get("plotting"),
        )

    print("\n[analyze] Analysis completed")
