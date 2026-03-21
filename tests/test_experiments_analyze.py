"""Regression tests for analyze plot regeneration behavior."""

from __future__ import annotations

import json
from pathlib import Path

from regret.experiments.orchestration import analyze_results


def _write_result(
    root: Path,
    suite_slug: str,
    problem_slug: str,
    algorithm_slug: str,
    n: int,
    budget: int,
) -> None:
    out_path = (
        root / suite_slug / problem_slug / algorithm_slug / f"n{n}" / f"b{budget}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "name": f"{suite_slug}/{problem_slug}/{algorithm_slug}/n{n}/b{budget}",
            "algorithm": "SimulatedAnnealing",
            "problem": "BinVal",
            "problem_size": n,
            "budget": budget,
            "runs": 1,
            "mode": "full",
            "timestamp": "2026-03-19T00:00:00",
        },
        "statistics": {
            "mean": 0.1,
            "median": 0.1,
            "std": 0.0,
            "min": 0.1,
            "max": 0.1,
            "prob_optimal": 0.0,
            "global_optimum": 10.0,
        },
        "results": [
            {
                "regret": 0.1,
                "best_value": 9.9,
                "optimal": False,
                "evaluations": budget,
                "seed": 0,
                "trajectory": [
                    [1, 1.0, 1.0],
                    [budget, 9.9, 9.9],
                ],
            }
        ],
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def test_analyze_preserves_problem_slug_and_algorithm_uniqueness(
    tmp_path: Path, monkeypatch
) -> None:
    raw_root = tmp_path / "raw"

    # Intentionally use slugs that do not match current config-derived slugs.
    # This simulates analyzing historical runs after config naming changes.
    _write_result(raw_root, "01_baseline", "bin_val", "sa_log_old", n=65, budget=200)
    _write_result(raw_root, "01_baseline", "bin_val", "sa_lin_old", n=65, budget=200)

    captured: list[dict] = []

    def _capture_generate_plots(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(
        "regret.experiments.orchestration.generate_plots", _capture_generate_plots
    )

    config = {
        "suite": {
            "name": "01_baseline",
            "runs": 1,
            "mode": "full",
            "budgets": [200],
            "output": {
                "raw_root": str(raw_root),
                "figures_root": str(tmp_path / "figures"),
            },
        },
        # Current config names do not match old raw slug `bin_val`.
        "problems": [
            {
                "name": "Bin-Val",
                "class": "BinVal",
                "params": {"n": 65},
            }
        ],
        "algorithms": [
            {
                "name": "SA-Log-Old",
                "class": "SimulatedAnnealing",
                "args": {"defaults": {}},
            },
            {
                "name": "SA-Lin-Old",
                "class": "SimulatedAnnealing",
                "args": {"defaults": {}},
            },
        ],
        "plotting": {
            "enabled": True,
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
    }

    analyze_results(config)

    assert len(captured) == 1
    call = captured[0]

    # Algorithm keys must not collapse to metadata class name.
    alg_names = sorted({alg for alg, _ in call["results"].keys()})
    assert alg_names == ["SA-Lin-Old", "SA-Log-Old"]
