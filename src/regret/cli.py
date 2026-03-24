"""
Command-line interface for experiment pipeline.

Usage:
    python -m regret.experiments path/to/config.yaml validate
    python -m regret.experiments path/to/config.yaml plan
    python -m regret.experiments path/to/config.yaml run [--no-plot]
    python -m regret.experiments path/to/config.yaml analyze [--results-dir DIR]
"""

import argparse
import sys
from pathlib import Path

from .experiments.orchestration import (
    analyze_results,
    execute_experiments,
    plan_execution,
)
from .experiments.validation import ValidationError, validate_config


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate config file.

    Returns:
        0 on success, 1 on validation failure.
    """
    try:
        config = validate_config(args.config)
        print(f"Config validation passed: {args.config}")
        print(f"Suite: {config['suite']['name']}")
        print(f"Mode: {config['suite']['mode']}")
        print(f"Problems: {len(config['problems'])}")
        print(f"Algorithms: {len(config['algorithms'])}")
        return 0
    except ValidationError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return 1


def cmd_plan(args: argparse.Namespace) -> int:
    """Display execution plan without running.

    Returns:
        0 on success, 1 on validation failure.
    """
    try:
        config = validate_config(args.config)
        plan = plan_execution(config)

        print("=" * 60)
        print("EXECUTION PLAN")
        print("=" * 60)
        print(f"Suite:         {plan['suite_name']}")
        print(f"Mode:          {plan['mode']}")
        print(f"Parallel:      {plan['parallel']}")
        print(f"Runs/combo:    {plan['runs_per_combo']}")
        print(f"Budgets:       {plan['budgets']}")
        print(f"\nProblems ({len(plan['problems'])}):")
        for p in plan["problems"]:
            print(f"  - {p}")
        print(f"\nAlgorithms ({len(plan['algorithms'])}):")
        for a in plan["algorithms"]:
            print(f"  - {a}")
        print(f"\nCombinations:  {plan['combination_count']}")
        print(f"Total runs:    {plan['total_runs']}")
        print("=" * 60)
        return 0
    except ValidationError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Execute experiments.

    Returns:
        0 on success, 1 on failure.
    """
    try:
        print(f"Loading config: {args.config}")
        config = validate_config(args.config)
        print("Config validated\n")

        plan = plan_execution(config)
        print(f"Executing {plan['total_runs']} runs across {plan['combination_count']} configurations...")
        print(f"Mode: {plan['mode']}, Parallel: {plan['parallel']}\n")

        execute_experiments(config, plot=not args.no_plot)
        print("\nExperiments completed")
        return 0
    except ValidationError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """Regenerate plots from existing results.

    Returns:
        0 on success, 1 on failure.
    """
    try:
        config = validate_config(args.config)
        analyze_results(config)
        print("Analysis completed")
        return 0
    except ValidationError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return 1
    except NotImplementedError as e:
        print(f"{e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        return 1


def app() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Regret experiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML config file",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Validate subcommand
    subparsers.add_parser("validate", help="Validate config file without executing")

    # Plan subcommand
    subparsers.add_parser("plan", help="Display execution plan without running")

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Execute experiments")
    run_parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation",
    )

    # Analyze subcommand
    subparsers.add_parser(
        "analyze",
        help="Regenerate plots from existing results obtained using the provided config",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handlers
    commands = {
        "validate": cmd_validate,
        "plan": cmd_plan,
        "run": cmd_run,
        "analyze": cmd_analyze,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(app())
