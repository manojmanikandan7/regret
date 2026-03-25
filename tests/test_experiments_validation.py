"""Tests for experiment config validation pipeline."""

from pathlib import Path

import pytest

from regret.experiments.validation import (
    ValidationError,
    _validate_cooling_schedule,
    load_config,
    validate_config,
    validate_schema,
    validate_semantic,
)


def _valid_config() -> dict:
    """Return a minimal valid experiment config."""
    return {
        "suite": {
            "name": "test_suite",
            "runs": 2,
            "mode": "lite",
            "budgets": [10, 20],
            "output": {
                "raw_root": "results/raw",
                "figures_root": "results/figures",
            },
        },
        "problems": [{"name": "OneMax", "class": "OneMax", "params": {"n": 8}}],
        "algorithms": [
            {
                "name": "RLS",
                "class": "RLS",
                "args": {
                    "defaults": {},
                    "by_problem": {},
                },
            }
        ],
        "plotting": {
            "enabled": False,
        },
    }


def test_load_config_raises_for_missing_file(tmp_path: Path) -> None:
    """load_config should fail for non-existent file paths."""
    missing = tmp_path / "does_not_exist.yaml"

    with pytest.raises(ValidationError, match="Config file not found"):
        load_config(missing)


def test_load_config_raises_for_malformed_yaml(tmp_path: Path) -> None:
    """Malformed YAML should be wrapped as ValidationError."""
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("suite: [", encoding="utf-8")

    with pytest.raises(ValidationError, match="Failed to parse YAML"):
        load_config(cfg_path)


def test_load_config_raises_for_non_mapping_yaml(tmp_path: Path) -> None:
    """YAML payload must parse to a non-empty mapping."""
    cfg_path = tmp_path / "list.yaml"
    cfg_path.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="non-empty mapping"):
        load_config(cfg_path)


def test_validate_schema_accepts_valid_config() -> None:
    """A known-good config should pass JSON schema validation."""
    validate_schema(_valid_config())


def test_validate_schema_accepts_suite_profile_flag() -> None:
    """suite.profile should be accepted as an optional boolean toggle."""
    config = _valid_config()
    config["suite"]["profile"] = True

    validate_schema(config)


def test_validate_schema_requires_figures_root_when_suite_profile_enabled() -> None:
    """figures_root is required when suite.profile is true."""
    config = _valid_config()
    config["suite"]["profile"] = True
    del config["suite"]["output"]["figures_root"]

    with pytest.raises(ValidationError, match="Schema validation failed"):
        validate_schema(config)


def test_validate_schema_rejects_profile_plots_when_suite_profile_disabled() -> None:
    """Runtime profile plot toggles must be disabled when suite.profile is false."""
    config = _valid_config()
    config["suite"]["profile"] = False
    config["plotting"]["plots"] = {
        "inverse_runtime_profile_surface": {"enabled": True},
        "inverse_runtime_profile_curves": {"enabled": False},
        "cr_profile_verification": {"enabled": False},
    }

    with pytest.raises(ValidationError, match="Schema validation failed"):
        validate_schema(config)


def test_validate_schema_accepts_disabled_profile_plots_when_suite_profile_disabled() -> None:
    """Runtime profile plot toggles may be present only when explicitly disabled."""
    config = _valid_config()
    config["suite"]["profile"] = False
    config["plotting"]["plots"] = {
        "inverse_runtime_profile_surface": {"enabled": False},
        "inverse_runtime_profile_curves": {"enabled": False},
        "cr_profile_verification": {"enabled": False},
    }

    validate_schema(config)


def test_validate_schema_allows_missing_figures_root_when_plotting_disabled() -> None:
    """figures_root is optional when plotting is disabled."""
    config = _valid_config()
    del config["suite"]["output"]["figures_root"]
    config["plotting"]["enabled"] = False

    validate_schema(config)


def test_validate_schema_requires_figures_root_when_plotting_enabled() -> None:
    """figures_root is required when plotting is enabled."""
    config = _valid_config()
    del config["suite"]["output"]["figures_root"]
    config["plotting"]["enabled"] = True

    with pytest.raises(ValidationError, match="Schema validation failed"):
        validate_schema(config)


def test_validate_schema_rejects_missing_required_key() -> None:
    """Missing required top-level keys should fail schema validation."""
    config = _valid_config()
    del config["suite"]

    with pytest.raises(ValidationError, match="Schema validation failed"):
        validate_schema(config)


def test_validate_schema_rejects_additional_top_level_properties() -> None:
    """Unknown top-level keys should fail because schema forbids extras."""
    config = _valid_config()
    config["unexpected"] = True

    with pytest.raises(ValidationError, match="Schema validation failed"):
        validate_schema(config)


def test_validate_schema_rejects_problem_without_name() -> None:
    """Problem entries must provide an explicit name."""
    config = _valid_config()
    del config["problems"][0]["name"]

    with pytest.raises(ValidationError, match="Schema validation failed"):
        validate_schema(config)


def test_validate_schema_rejects_algorithm_without_name() -> None:
    """Algorithm entries must provide an explicit name."""
    config = _valid_config()
    del config["algorithms"][0]["name"]

    with pytest.raises(ValidationError, match="Schema validation failed"):
        validate_schema(config)


def test_validate_semantic_rejects_unknown_problem_class() -> None:
    """Semantic validation should reject unknown problem classes."""
    config = _valid_config()
    config["problems"][0]["class"] = "NotAProblem"

    with pytest.raises(ValidationError, match="Unknown problem class"):
        validate_semantic(config)


def test_validate_semantic_rejects_unknown_algorithm_class() -> None:
    """Semantic validation should reject unknown algorithm classes."""
    config = _valid_config()
    config["algorithms"][0]["class"] = "NotAnAlgorithm"

    with pytest.raises(ValidationError, match="Unknown algorithm class"):
        validate_semantic(config)


def test_validate_semantic_rejects_unknown_cooling_string() -> None:
    """String cooling schedules must exist in COOLING_REGISTRY."""
    config = _valid_config()
    config["algorithms"][0]["args"]["defaults"] = {"cooling": "unknown-schedule"}

    with pytest.raises(ValidationError, match="Unknown cooling schedule"):
        validate_semantic(config)


def test_validate_semantic_rejects_unknown_cooling_type_in_mapping() -> None:
    """Dict cooling schedules must have a valid type."""
    config = _valid_config()
    config["algorithms"][0]["args"]["defaults"] = {"T_func": {"type": "unknown", "params": {"T0": 1.0}}}

    with pytest.raises(ValidationError, match="Unknown cooling schedule"):
        validate_semantic(config)


def test_validate_semantic_rejects_unknown_cooling_in_by_problem() -> None:
    """Problem-specific cooling overrides are semantically validated too."""
    config = _valid_config()
    config["algorithms"][0]["args"]["by_problem"] = {"OneMax": {"cooling": "not-valid"}}

    with pytest.raises(ValidationError, match="Unknown cooling schedule"):
        validate_semantic(config)


def test_validate_semantic_rejects_duplicate_problem_names() -> None:
    """Problem names should be unique for unambiguous by_problem mapping."""
    config = _valid_config()
    config["problems"] = [
        {"name": "P", "class": "OneMax", "params": {"n": 8}},
        {"name": "P", "class": "LeadingOnes", "params": {"n": 8}},
    ]

    with pytest.raises(ValidationError, match="Duplicate problem name"):
        validate_semantic(config)


def test_validate_semantic_rejects_duplicate_algorithm_names() -> None:
    """Algorithm names should be unique for stable keys/legends."""
    config = _valid_config()
    config["algorithms"] = [
        {"name": "A", "class": "RLS", "args": {"defaults": {}, "by_problem": {}}},
        {
            "name": "A",
            "class": "RLSExploration",
            "args": {"defaults": {}, "by_problem": {}},
        },
    ]

    with pytest.raises(ValidationError, match="Duplicate algorithm name"):
        validate_semantic(config)


def test_validate_semantic_rejects_unknown_by_problem_target() -> None:
    """by_problem overrides must target configured problem names."""
    config = _valid_config()
    config["algorithms"][0]["args"]["by_problem"] = {"MissingProblem": {"cooling": "linear"}}

    with pytest.raises(ValidationError, match="unknown problem name"):
        validate_semantic(config)


def test_validate_semantic_rejects_redundant_cooling_keys() -> None:
    """Config should use only one cooling key to avoid ambiguity."""
    config = _valid_config()
    config["algorithms"][0]["args"]["defaults"] = {
        "cooling": "linear",
        "T_func": "linear",
    }

    with pytest.raises(ValidationError, match="use only one of 'cooling' or 'T_func'"):
        validate_semantic(config)


def test_validate_cooling_schedule_ignores_absent_keys() -> None:
    """Helper should be a no-op when no cooling key is present."""
    _validate_cooling_schedule({"alpha": 0.1}, "algorithms[0].args.defaults")


def test_validate_config_no_plotting(tmp_path: Path) -> None:
    """validate_config should fail schema validation when plotting is missing."""
    cfg_path = tmp_path / "bad_no_plot.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "suite:",
                "  name: test_suite",
                "  runs: 2",
                "  mode: lite",
                "  budgets: [10, 20]",
                "  output:",
                "    raw_root: results/raw",
                "    figures_root: results/figures",
                "problems:",
                "  - name: OneMax",
                "    class: OneMax",
                "    params:",
                "      n: 8",
                "algorithms:",
                "  - name: RLS",
                "    class: RLS",
                "    args:",
                "      defaults: {}",
                "      by_problem: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Schema validation failed"):
        validate_config(cfg_path)


def test_validate_config_end_to_end_success(tmp_path: Path) -> None:
    """validate_config should load and validate a good YAML config."""
    cfg_path = tmp_path / "ok.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "suite:",
                "  name: test_suite",
                "  runs: 2",
                "  mode: lite",
                "  budgets: [10, 20]",
                "  output:",
                "    raw_root: results/raw",
                "    figures_root: results/figures",
                "problems:",
                "  - name: OneMax",
                "    class: OneMax",
                "    params:",
                "      n: 8",
                "algorithms:",
                "  - name: RLS",
                "    class: RLS",
                "    args:",
                "      defaults: {}",
                "      by_problem: {}",
                "plotting:",
                "   enabled: False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = validate_config(cfg_path)
    assert parsed["suite"]["name"] == "test_suite"
    assert parsed["problems"][0]["class"] == "OneMax"
    assert parsed["plotting"]["enabled"] is False


def test_validate_config_end_to_end_semantic_failure(tmp_path: Path) -> None:
    """validate_config should surface semantic validation failures."""
    cfg_path = tmp_path / "bad_semantic.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "suite:",
                "  name: test_suite",
                "  runs: 1",
                "  mode: lite",
                "  budgets: [10]",
                "  output:",
                "    raw_root: results/raw",
                "    figures_root: results/figures",
                "problems:",
                "  - name: OneMax",
                "    class: OneMax",
                "    params:",
                "      n: 8",
                "algorithms:",
                "  - name: RLS",
                "    class: RLS",
                "    args:",
                "      defaults:",
                "        cooling: not-valid",
                "      by_problem: {}",
                "plotting:",
                "  enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Unknown cooling schedule"):
        validate_config(cfg_path)
