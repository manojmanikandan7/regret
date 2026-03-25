"""Tests for configurable budget-specific plotting behavior."""

import csv
from pathlib import Path

import pytest

from regret.experiments.utils import export_runtime_profile_data, generate_plots
from regret.experiments.validation import (
    ValidationError,
    validate_schema,
    validate_semantic,
)


def _base_config() -> dict:
    return {
        "suite": {
            "name": "test_suite",
            "runs": 2,
            "mode": "lite",
            "budgets": [10, 20, 50],
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
                "args": {"defaults": {}, "by_problem": {}},
            }
        ],
        "plotting": {"enabled": False},
    }


def test_schema_accepts_plotting_budget_for_plots() -> None:
    config = _base_config()
    config["plotting"] = {"budget_for_plots": 20}

    # Should not raise schema validation errors.
    validate_schema(config)


def test_schema_accepts_problem_level_budget_for_plots() -> None:
    config = _base_config()
    config["problems"][0]["budget_for_plots"] = 10

    # Should not raise schema validation errors.
    validate_schema(config)


def test_schema_accepts_optional_plotting_stats_keys() -> None:
    config = _base_config()
    config["plotting"] = {
        "enabled": True,
        "layout": {
            "structure": {
                "aggregate": "aggregate",
                "history": "history",
                "distribution": "distribution",
            }
        },
        "plots": {
            "regret_curves": {
                "enabled": True,
                "spread": "bootstrap_ci",
                "confidence": 0.95,
                "n_bootstrap": 256,
                "annotate_pairwise": True,
                "comparison_budget": 20,
                "paired_runs": False,
            },
            "convergence_probability": {
                "enabled": True,
                "show_confidence_band": True,
                "confidence": 0.9,
            },
            "regret_boxplots": {
                "enabled": True,
                "show_points": True,
                "annotate_pairwise": True,
                "reference_algorithm": "RLS",
                "paired_runs": False,
            },
            "performance_profile": {
                "enabled": True,
                "annotate_pairwise": True,
                "reference_algorithm": "RLS",
                "paired_runs": False,
            },
            "history_current": {
                "enabled": False,
                "spread": "sd",
                "confidence": 0.9,
                "n_bootstrap": 128,
            },
            "regret_instantaneous": {
                "enabled": False,
                "spread": "iqr",
                "confidence": 0.9,
                "n_bootstrap": 128,
            },
            "ttfo_distribution": {
                "enabled": False,
                "tolerance": 1e-8,
                "show_median": True,
                "annotate_pairwise": True,
                "reference_algorithm": "RLS",
                "paired_runs": False,
            },
        },
    }

    validate_schema(config)


def test_semantic_rejects_plotting_budget_not_in_suite_budgets() -> None:
    config = _base_config()
    config["plotting"] = {"budget_for_plots": 999}

    with pytest.raises(ValidationError, match="plotting.budget_for_plots"):
        validate_semantic(config)


def test_semantic_rejects_problem_budget_not_in_suite_budgets() -> None:
    config = _base_config()
    config["problems"][0]["budget_for_plots"] = 999

    with pytest.raises(ValidationError, match="problems\\[0\\]\\.budget_for_plots"):
        validate_semantic(config)


def test_generate_plots_uses_selected_budget_for_budget_specific_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, dict[str, object]] = {}

    def _record_boxplot(results, budget, save_path, title, show):
        calls["boxplot"] = {
            "budget": budget,
            "save_path": save_path,
            "title": title,
        }

    def _record_profile(results, budget, save_path, title, show):
        calls["profile"] = {
            "budget": budget,
            "save_path": save_path,
            "title": title,
        }

    def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr("regret.experiments.utils.plot_simple_regret_boxplots", _record_boxplot)
    monkeypatch.setattr("regret.experiments.utils.plot_performance_profile", _record_profile)
    monkeypatch.setattr("regret.experiments.utils.plot_simple_regret_curves", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_convergence_probability", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_comparison_heatmap", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_history", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_regret_curves", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_ttfo_distribution", _noop)

    results = {
        ("RLS", 10): [{"regret": 0.5}],
        ("RLS", 20): [{"regret": 0.1}],
    }

    generate_plots(
        suite_name="suite",
        problem_name="OneMax",
        n=8,
        f_star=None,
        results=results,
        budget_for_plots=10,
        output_dir=tmp_path,
        plotting_config={
            "layout": {
                "per_problem_dir": True,
                "include_n_subdir": True,
                "structure": {
                    "aggregate": "aggregate",
                    "history": "history",
                    "distribution": "distribution",
                },
            },
            "plots": {
                "regret_boxplots": {"enabled": True},
                "performance_profile": {"enabled": True},
                "history_current": {"enabled": False},
                "history_best": {"enabled": False},
                "regret_instantaneous": {"enabled": False},
                "regret_cumulative": {"enabled": False},
                "regret_instantaneous_best": {"enabled": False},
                "regret_cumulative_best": {"enabled": False},
                "ttfo_distribution": {"enabled": False},
                "regret_curves": {"enabled": False},
                "convergence_probability": {"enabled": False},
                "comparison_heatmap": {"enabled": False},
            },
        },
    )

    assert calls["boxplot"]["budget"] == 10
    assert calls["profile"]["budget"] == 10
    assert "distribution" in str(calls["boxplot"]["save_path"])
    assert "distribution" in str(calls["profile"]["save_path"])


def test_generate_plots_problem_budget_overrides_global_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, dict[str, object]] = {}

    def _record_boxplot(results, budget, save_path, title, show):
        calls["boxplot"] = {
            "budget": budget,
            "save_path": save_path,
            "title": title,
        }

    def _record_profile(results, budget, save_path, title, show):
        calls["profile"] = {
            "budget": budget,
            "save_path": save_path,
            "title": title,
        }

    def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr("regret.experiments.utils.plot_simple_regret_boxplots", _record_boxplot)
    monkeypatch.setattr("regret.experiments.utils.plot_performance_profile", _record_profile)
    monkeypatch.setattr("regret.experiments.utils.plot_simple_regret_curves", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_convergence_probability", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_comparison_heatmap", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_history", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_regret_curves", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_ttfo_distribution", _noop)

    results = {
        ("RLS", 10): [{"regret": 0.5}],
        ("RLS", 20): [{"regret": 0.1}],
    }

    # Simulate orchestration precedence: per-problem value selected over global.
    selected_budget = 10

    generate_plots(
        suite_name="suite",
        problem_name="OneMax",
        n=8,
        f_star=None,
        results=results,
        budget_for_plots=selected_budget,
        output_dir=tmp_path,
        plotting_config={
            "budget_for_plots": 20,
            "layout": {
                "per_problem_dir": True,
                "include_n_subdir": True,
                "structure": {
                    "aggregate": "aggregate",
                    "history": "history",
                    "distribution": "distribution",
                },
            },
            "plots": {
                "regret_boxplots": {"enabled": True},
                "performance_profile": {"enabled": True},
                "history_current": {"enabled": False},
                "history_best": {"enabled": False},
                "regret_instantaneous": {"enabled": False},
                "regret_cumulative": {"enabled": False},
                "regret_instantaneous_best": {"enabled": False},
                "regret_cumulative_best": {"enabled": False},
                "ttfo_distribution": {"enabled": False},
                "regret_curves": {"enabled": False},
                "convergence_probability": {"enabled": False},
                "comparison_heatmap": {"enabled": False},
            },
        },
    )

    assert calls["boxplot"]["budget"] == 10
    assert calls["profile"]["budget"] == 10
    assert "distribution" in str(calls["boxplot"]["save_path"])
    assert "distribution" in str(calls["profile"]["save_path"])


def test_generate_plots_does_not_write_runtime_profile_csv_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _noop(*args, **kwargs):
        return None

    def _fake_profile_analysis(results, f_star, budget):
        _ = (results, f_star, budget)
        time_grid = [1.0, 2.0, 3.0]
        fitness_levels = [1.0, 2.0]
        profiles = {"RLS": [[0.5, 0.7, 1.0], [0.2, 0.6, 1.0]]}
        empirical_ecr = {"RLS": [2.0, 1.0, 0.0]}
        profile_ecr = {"RLS": [2.1, 1.1, 0.1]}
        return time_grid, fitness_levels, profiles, empirical_ecr, profile_ecr

    monkeypatch.setattr("regret.experiments.utils.plot_simple_regret_curves", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_convergence_probability", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_comparison_heatmap", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_simple_regret_boxplots", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_performance_profile", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_history", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_regret_curves", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_ttfo_distribution", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_inverse_runtime_profile_surface", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_inverse_runtime_profile_curves", _noop)
    monkeypatch.setattr("regret.experiments.utils.plot_cr_profile_verification", _noop)
    monkeypatch.setattr("regret.analysis.profiles.run_profile_analysis", _fake_profile_analysis)

    results = {
        ("RLS", 10): [
            {
                "regret": 0.5,
                "trajectory": [
                    (1, 1.0, 1.0),
                    (10, 2.0, 2.0),
                ],
            }
        ]
    }

    generate_plots(
        suite_name="suite",
        problem_name="OneMax",
        n=8,
        f_star=2.0,
        results=results,
        budget_for_plots=10,
        output_dir=tmp_path,
        plotting_config={
            "layout": {
                "per_problem_dir": True,
                "include_n_subdir": True,
                "structure": {
                    "aggregate": "aggregate",
                    "history": "history",
                    "distribution": "distribution",
                    "profile": "profiles",
                },
            },
            "plots": {
                "regret_boxplots": {"enabled": False},
                "performance_profile": {"enabled": False},
                "history_current": {"enabled": False},
                "history_best": {"enabled": False},
                "regret_instantaneous": {"enabled": False},
                "regret_cumulative": {"enabled": False},
                "regret_instantaneous_best": {"enabled": False},
                "regret_cumulative_best": {"enabled": False},
                "ttfo_distribution": {"enabled": False},
                "regret_curves": {"enabled": False},
                "convergence_probability": {"enabled": False},
                "comparison_heatmap": {"enabled": False},
                "inverse_runtime_profile_surface": {"enabled": False},
                "inverse_runtime_profile_curves": {"enabled": False},
                "cr_profile_verification": {"enabled": False},
            },
        },
    )

    csv_path = tmp_path / "suite" / "onemax" / "profiles" / "n8" / "cr_profile_data_rls.csv"
    assert not csv_path.exists()


def test_export_runtime_profile_csv_writes_to_raw_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_profile_analysis(results, f_star, budget):
        _ = (results, f_star, budget)
        time_grid = [1.0, 2.0, 3.0]
        fitness_levels = [1.0, 2.0]
        profiles = {"RLS": [[0.5, 0.7, 1.0], [0.2, 0.6, 1.0]]}
        empirical_ecr = {"RLS": [2.0, 1.0, 0.0]}
        profile_ecr = {"RLS": [2.1, 1.1, 0.1]}
        return time_grid, fitness_levels, profiles, empirical_ecr, profile_ecr

    monkeypatch.setattr("regret.analysis.profiles.run_profile_analysis", _fake_profile_analysis)

    results = {
        ("RLS", 10): [
            {
                "regret": 0.5,
                "trajectory": [
                    (1, 1.0, 1.0),
                    (10, 2.0, 2.0),
                ],
            }
        ]
    }

    export_runtime_profile_data(
        suite_name="suite",
        problem_name="OneMax",
        n=8,
        f_star=2.0,
        results=results,
        budget_for_plots=10,
        plotting_config={
            "layout": {
                "per_problem_dir": True,
                "include_n_subdir": True,
                "structure": {
                    "profile": "profiles",
                },
            }
        },
        raw_output_dir=tmp_path,
    )

    csv_path = tmp_path / "suite" / "onemax" / "profiles" / "n8" / "cr_profile_data_rls.csv"
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == [
        "evaluation",
        "expected_cumulative_regret",
        "mean_cumulative_regret",
    ]
    assert rows[1] == ["1.0", "2.1", "2.0"]
    assert rows[-1] == ["3.0", "0.1", "0.0"]


def test_export_runtime_profile_csv_plots_when_figures_dir_is_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, str] = {}

    def _fake_profile_analysis(results, f_star, budget):
        _ = (results, f_star, budget)
        time_grid = [1.0, 2.0, 3.0]
        fitness_levels = [1.0, 2.0]
        profiles = {"RLS": [[0.5, 0.7, 1.0], [0.2, 0.6, 1.0]]}
        empirical_ecr = {"RLS": [2.0, 1.0, 0.0]}
        profile_ecr = {"RLS": [2.1, 1.1, 0.1]}
        return time_grid, fitness_levels, profiles, empirical_ecr, profile_ecr

    def _record_surface(*, save_path, **kwargs):
        _ = kwargs
        calls["surface"] = save_path

    def _record_curves(*, save_path, **kwargs):
        _ = kwargs
        calls["curves"] = save_path

    def _record_verification(*, save_path, **kwargs):
        _ = kwargs
        calls["verification"] = save_path

    monkeypatch.setattr("regret.analysis.profiles.run_profile_analysis", _fake_profile_analysis)
    monkeypatch.setattr("regret.experiments.utils.plot_inverse_runtime_profile_surface", _record_surface)
    monkeypatch.setattr("regret.experiments.utils.plot_inverse_runtime_profile_curves", _record_curves)
    monkeypatch.setattr("regret.experiments.utils.plot_cr_profile_verification", _record_verification)

    results = {
        ("RLS", 10): [
            {
                "regret": 0.5,
                "trajectory": [
                    (1, 1.0, 1.0),
                    (10, 2.0, 2.0),
                ],
            }
        ]
    }

    export_runtime_profile_data(
        suite_name="suite",
        problem_name="OneMax",
        n=8,
        f_star=2.0,
        results=results,
        budget_for_plots=10,
        plotting_config={
            "layout": {
                "per_problem_dir": True,
                "include_n_subdir": True,
                "structure": {
                    "profile": "profiles",
                },
            },
            "plots": {
                "inverse_runtime_profile_surface": {"enabled": True},
                "inverse_runtime_profile_curves": {"enabled": True},
                "cr_profile_verification": {"enabled": True},
            },
        },
        raw_output_dir=tmp_path / "raw",
        figures_output_dir=tmp_path / "figures",
    )

    assert "figures/suite/onemax/n8/profiles" in calls["surface"]
    assert "figures/suite/onemax/n8/profiles" in calls["curves"]
    assert "figures/suite/onemax/n8/profiles" in calls["verification"]

    csv_path = tmp_path / "raw" / "suite" / "onemax" / "profiles" / "n8" / "cr_profile_data_rls.csv"
    assert csv_path.exists()
