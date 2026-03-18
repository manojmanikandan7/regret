"""
Config validation using JSON Schema and semantic checks.

Validates YAML experiment configs against schema and registry constraints.
"""

from pathlib import Path
from typing import Any

import jsonschema
import yaml


from .schema import CONFIG_SCHEMA

from .utils import ALGORITHM_REGISTRY, PROBLEM_REGISTRY, COOLING_REGISTRY


class ValidationError(Exception):
    """Raised when config validation fails."""

    pass


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Parsed config dictionary.

    Raises:
        ValidationError: If file cannot be loaded or parsed.
    """
    if not config_path.exists():
        raise ValidationError(f"Config file not found: {config_path}")

    try:
        with config_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValidationError(f"Failed to parse YAML: {e}") from e

    if not cfg or not isinstance(cfg, dict):
        raise ValidationError("Config must be a non-empty mapping")

    return cfg


def validate_schema(config: dict[str, Any]) -> None:
    """Validate config against JSON schema.

    Args:
        config: Parsed config dictionary.

    Raises:
        ValidationError: If schema validation fails.
    """
    try:
        jsonschema.validate(instance=config, schema=CONFIG_SCHEMA)
    except jsonschema.ValidationError as e:
        raise ValidationError(f"Schema validation failed: {e.message}") from e
    except jsonschema.SchemaError as e:
        raise ValidationError(f"Invalid schema definition: {e.message}") from e


def validate_semantic(config: dict[str, Any]) -> None:
    """Perform semantic validation beyond schema structure.

    Validates:
    - Problem classes exist in PROBLEM_REGISTRY
    - Algorithm classes exist in ALGORITHM_REGISTRY
    - Cooling schedule types exist in COOLING_REGISTRY

    Args:
        config: Parsed config dictionary.

    Raises:
        ValidationError: If semantic validation fails.
    """
    # Validate problem classes
    for idx, problem in enumerate(config.get("problems", [])):
        class_name = problem.get("class")
        if class_name not in PROBLEM_REGISTRY:
            raise ValidationError(
                f"problems[{idx}].class: Unknown problem class '{class_name}'. "
                f"Available: {sorted(PROBLEM_REGISTRY.keys())}"
            )

    # Validate algorithm classes and args structure
    for idx, algorithm in enumerate(config.get("algorithms", [])):
        class_name = algorithm.get("class")
        if class_name not in ALGORITHM_REGISTRY:
            raise ValidationError(
                f"algorithms[{idx}].class: Unknown algorithm class '{class_name}'. "
                f"Available: {sorted(ALGORITHM_REGISTRY.keys())}"
            )

        # Validate cooling schedules if present
        args = algorithm.get("args", {})
        defaults = args.get("defaults", {})
        by_problem = args.get("by_problem", {})

        # Check defaults
        _validate_cooling_schedule(defaults, f"algorithms[{idx}].args.defaults")

        # Check by_problem overrides
        for prob_name, overrides in by_problem.items():
            _validate_cooling_schedule(
                overrides, f"algorithms[{idx}].args.by_problem[{prob_name}]"
            )

    # Validate optional plotting budget selector
    plotting_cfg = config.get("plotting", {})
    selected_budget = plotting_cfg.get("budget_for_plots")
    if selected_budget is not None:
        suite_budgets = {int(b) for b in config.get("suite", {}).get("budgets", [])}
        if int(selected_budget) not in suite_budgets:
            raise ValidationError(
                "plotting.budget_for_plots: must be one of suite.budgets "
                f"{sorted(suite_budgets)}"
            )


def _validate_cooling_schedule(args_dict: dict[str, Any], path: str) -> None:
    """Validate cooling schedule specification in algorithm args.

    Args:
        args_dict: Dictionary containing potential T_func or cooling specs.
        path: Path string for error messages.

    Raises:
        ValidationError: If cooling schedule is invalid.
    """
    for key in ["T_func", "cooling"]:
        if key not in args_dict:
            continue

        spec = args_dict[key]
        if isinstance(spec, dict):
            sched_type = spec.get("type", "").strip().lower()
            if sched_type not in COOLING_REGISTRY:
                raise ValidationError(
                    f"{path}.{key}.type: Unknown cooling schedule '{spec.get('type')}'. "
                    f"Available: {sorted(COOLING_REGISTRY.keys())}"
                )
        elif isinstance(spec, str):
            sched_type = spec.strip().lower()
            if sched_type not in COOLING_REGISTRY:
                raise ValidationError(
                    f"{path}.{key}: Unknown cooling schedule '{spec}'. "
                    f"Available: {sorted(COOLING_REGISTRY.keys())}"
                )


def validate_config(config_path: Path) -> dict[str, Any]:
    """Complete config validation pipeline.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Validated config dictionary.

    Raises:
        ValidationError: If any validation step fails.
    """
    config = load_config(config_path)
    validate_schema(config)
    validate_semantic(config)
    return config
