"""
Command-line interface for experiment pipeline.

Usage:
    ```
    run_experiment validate path/to/config.yaml [path/to/config2.yaml ...]
    ```
    ```
    run_experiment plan path/to/config.yaml [path/to/config2.yaml ...]
    ```
    ```
    run_experiment run path/to/config.yaml [path/to/config2.yaml ...] [--no-plot]
    ```
    ```
    run_experiment analyze path/to/config.yaml [path/to/config2.yaml ...]
    ```
"""

import sys
import traceback
from pathlib import Path
from typing import Annotated

import typer

from regret.experiments.orchestration import (
    analyze_results,
    execute_experiments,
    plan_execution,
)
from regret.experiments.validation import ValidationError, validate_config

app = typer.Typer(help="Regret experiment pipeline")


def _print_config_header(config: Path, index: int, total: int) -> None:
    """Print a per-config banner when multiple configs are provided."""
    if total > 1:
        print(f"\n[{index}/{total}] Config: {config}")


def _validate_single(config_path: Path) -> bool:
    """Validate a single config file."""
    try:
        config = validate_config(config_path)
        print(f"Config validation passed: {config_path}")
        print(f"Suite: {config['suite']['name']}")
        print(f"Mode: {config['suite']['mode']}")
        print(f"Problems: {len(config['problems'])}")
        print(f"Algorithms: {len(config['algorithms'])}")
        return True
    except ValidationError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return False


def _plan_single(config_path: Path) -> bool:
    """Display execution plan for a single config file."""
    try:
        config = validate_config(config_path)
        plan = plan_execution(config)

        print("=" * 60)
        print("EXECUTION PLAN")
        print("=" * 60)
        print(f"Config:        {config_path}")
        print(f"Suite:         {plan['suite_name']}")
        print(f"Mode:          {plan['mode']}")
        print(f"Parallel:      {plan['parallel']}")
        print(f"Profiling:     {plan['profile']}")
        print(f"Plotting:      {plan['plotting_enabled']}")
        print(f"Raw output:    {plan['output_raw_root']}")
        print(f"Figures out:   {plan['output_figures_root']}")
        print(f"Runs/combo:    {plan['runs_per_combo']}")
        print(f"Budgets:       {plan['budgets']}")
        print(f"Budget span:   min={plan['budget_min']}, max={plan['budget_max']}, count={plan['budget_count']}")

        print(f"\nProblems ({len(plan['problem_details'])}):")
        for problem in plan["problem_details"]:
            n_value = problem["n"] if problem["n"] is not None else "n/a"
            params = problem["params"] or {}
            print(
                "  - "
                f"{problem['name']} "
                f"[class={problem['class_name']}, n={n_value}, budget_for_plots={problem['budget_for_plots']}]"
            )
            if params:
                print(f"      params={params}")

        print(f"\nAlgorithms ({len(plan['algorithm_details'])}):")
        for algorithm in plan["algorithm_details"]:
            print(
                "  - "
                f"{algorithm['name']} "
                f"[class={algorithm['class_name']}, problem_overrides={algorithm['override_problem_count']}]"
            )
            defaults = algorithm["defaults"] or {}
            if defaults:
                print(f"      defaults={defaults}")
            by_problem = algorithm["by_problem"] or {}
            if by_problem:
                print(f"      by_problem={list(by_problem.keys())}")

        print(f"\nCombinations:  {plan['combination_count']}")
        print(f"Combos/budget: {plan['combos_per_budget']}")
        print(f"Runs/budget:   {plan['runs_per_budget']}")
        print(f"Total runs:    {plan['total_runs']}")
        print(f"Total evals:   {plan['total_evaluations']}")
        print("=" * 60)
        return True
    except ValidationError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return False


def _run_single(config_path: Path, *, no_plot: bool) -> bool:
    """Execute experiments for a single config file."""
    try:
        print(f"Loading config: {config_path}")
        config = validate_config(config_path)
        print("Config validated\n")

        plan = plan_execution(config)
        print(f"Executing {plan['total_runs']} runs across {plan['combination_count']} configurations...")
        print(f"Mode: {plan['mode']}, Parallel: {plan['parallel']}\n")

        execute_experiments(config, plot=not no_plot)
        print("\nExperiments completed")
        return True
    except ValidationError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return False


def _analyze_single(config_path: Path) -> bool:
    """Regenerate plots for a single config file."""
    try:
        config = validate_config(config_path)
        analyze_results(config)
        print("Analysis completed")
        return True
    except ValidationError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return False
    except NotImplementedError as e:
        print(f"{e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        return False


def _exit_on_failures(failures: int) -> None:
    """Return non-zero exit code when any config command fails."""
    if failures > 0:
        raise typer.Exit(code=1)


ConfigPaths = Annotated[
    list[Path],
    typer.Argument(
        ..., help="One or more YAML config file paths to process.", exists=True, readable=True, dir_okay=False
    ),
]


@app.command()
def validate(configs: ConfigPaths) -> None:
    """Validate config files without executing experiments."""
    failures = 0
    total = len(configs)
    for index, config_path in enumerate(configs, start=1):
        _print_config_header(config_path, index, total)
        if not _validate_single(config_path):
            failures += 1
    _exit_on_failures(failures)


@app.command()
def plan(configs: ConfigPaths) -> None:
    """Display execution plans without running experiments."""
    failures = 0
    total = len(configs)
    for index, config_path in enumerate(configs, start=1):
        _print_config_header(config_path, index, total)
        if not _plan_single(config_path):
            failures += 1
    _exit_on_failures(failures)


@app.command()
def run(
    configs: ConfigPaths,
    no_plot: Annotated[bool, typer.Option("--no-plot", help="Skip plot generation")] = False,
) -> None:
    """Execute experiments for one or more config files."""
    failures = 0
    total = len(configs)
    for index, config_path in enumerate(configs, start=1):
        _print_config_header(config_path, index, total)
        if not _run_single(config_path, no_plot=no_plot):
            failures += 1
    _exit_on_failures(failures)


@app.command()
def analyze(configs: ConfigPaths) -> None:
    """Regenerate plots from existing results for one or more config files."""
    failures = 0
    total = len(configs)
    for index, config_path in enumerate(configs, start=1):
        _print_config_header(config_path, index, total)
        if not _analyze_single(config_path):
            failures += 1
    _exit_on_failures(failures)


def main() -> None:
    """CLI entry point for script and module execution."""
    app()


if __name__ == "__main__":
    main()
