"""Tests for regret.core.metrics module."""

import numpy as np
import pytest

from regret.analysis.profiles import run_profile_analysis
from regret.analysis.tables import (
    TableFormat,
    compute_table_statistics,
    create_aggregate_table,
    create_results_table,
    export_latex_table,
    export_table,
)
from regret.core.metrics import (
    compute_inv_runtime_profile,
    compute_statistics,
    cumulative_regret,
    expected_simple_regret,
    history_best_series,
    history_current_series,
    instantaneous_regret,
    inv_profile_to_expected_cumulative_regret,
    inv_runtime_profile_single_run,
    normalized_regret,
    probability_optimal,
    simple_regret,
    time_to_target,
    ttfo,
)


class TestSimpleRegret:
    """Test suite for simple regret calculation."""

    @pytest.mark.parametrize(
        "solution_value,f_star,expected",
        [
            (10.0, 10.0, 0.0),
            (8.0, 10.0, 2.0),
            (11.0, 10.0, -1.0),
        ],
    )
    def test_simple_regret_values(self, solution_value, f_star, expected):
        """simple_regret should match the exact f_star - solution_value formula."""
        regret = simple_regret(solution_value, f_star)
        assert regret == expected

    def test_simple_regret_trajectory(self):
        """Test simple regret for a trajectory of solution values."""
        solution_values = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        f_star = 10.0
        regrets = [simple_regret(v, f_star) for v in solution_values]
        # Regrets should be decreasing
        assert all(regrets[i] >= regrets[i + 1] for i in range(len(regrets) - 1))


class TestSummaryStatistics:
    """Test suite for aggregate summary helpers."""

    def test_compute_statistics_returns_expected_quantities(self):
        """compute_statistics should return correct moments and quartiles."""
        regrets = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        stats = compute_statistics(regrets)

        assert stats["mean"] == 2.0
        assert stats["median"] == 2.0
        assert stats["min"] == 0.0
        assert stats["max"] == 4.0
        assert stats["q25"] == 1.0
        assert stats["q75"] == 3.0

    def test_probability_optimal_matches_tolerance_rule(self):
        """probability_optimal should count regrets <= tolerance."""
        regrets = np.array([0.0, 1e-10, 1e-4, 1.0])

        assert probability_optimal(regrets, tolerance=1e-9) == 0.5
        assert probability_optimal(regrets, tolerance=1e-3) == 0.75

    def test_compute_statistics_rejects_empty_input(self):
        """Empty arrays should fail fast with a clear error."""
        with pytest.raises(ValueError, match="non-empty"):
            compute_statistics(np.array([]))

    def test_probability_optimal_rejects_empty_input(self):
        """Empty arrays should fail fast with a clear error."""
        with pytest.raises(ValueError, match="non-empty"):
            probability_optimal(np.array([[]]))

    def test_probability_optimal_rejects_negative_tolerance(self):
        """Tolerance is a threshold and must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            probability_optimal(np.array([0.0, 1.0]), tolerance=-1e-9)


class TestInstantaneousRegret:
    """Test suite for instantaneous regret."""

    def test_instantaneous_regret_single_evaluation(self):
        """Test instantaneous regret for a single evaluation."""
        trajectory = [(0, 8.0, 8.0)]
        f_star = 10.0
        regrets = instantaneous_regret(trajectory, f_star, track_incumbent=True)
        assert len(regrets) == 1
        assert regrets[0][1] == 2.0  # f_star - best_value

    def test_instantaneous_regret_trajectory(self):
        """Test instantaneous regret for a trajectory."""
        trajectory = [
            (1, 5.0, 5.0),
            (2, 6.0, 6.0),
            (3, 7.0, 7.0),
            (4, 8.0, 8.0),
            (5, 9.0, 9.0),
        ]
        f_star = 10.0
        regrets = instantaneous_regret(trajectory, f_star, track_incumbent=True)
        expected_values = [5.0, 4.0, 3.0, 2.0, 1.0]
        actual_values = [r[1] for r in regrets]
        assert actual_values == expected_values


class TestCumulativeRegret:
    """Test suite for cumulative regret."""

    def test_cumulative_regret_basic(self):
        """Test cumulative regret calculation."""
        # Trajectory with evaluations at 1, 2, 3, 4, 5, 6
        trajectory = [
            (1, 5.0, 5.0),
            (2, 6.0, 6.0),
            (3, 7.0, 7.0),
            (4, 8.0, 8.0),
            (5, 9.0, 9.0),
            (6, 10.0, 10.0),
        ]
        f_star = 10.0
        cum_regrets = cumulative_regret(trajectory, f_star, track_incumbent=True)
        # First entry should account for regret from time 0 to first evaluation
        # Regret at eval 1 is 10.0 - 5.0 = 5.0, so at t=1 it's 1 * 5.0 = 5.0
        assert cum_regrets[0][1] == 5.0
        # Cumulative regrets should be non-decreasing
        values = [r[1] for r in cum_regrets]
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))

    def test_cumulative_regret_length(self):
        """Test that cumulative regret has same length as trajectory."""
        trajectory = [
            (1, 1.0, 1.0),
            (2, 2.0, 2.0),
            (3, 3.0, 3.0),
            (4, 4.0, 4.0),
            (5, 5.0, 5.0),
        ]
        f_star = 10.0
        cum_regrets = cumulative_regret(trajectory, f_star, track_incumbent=True)
        assert len(cum_regrets) == len(trajectory)

    def test_cumulative_regret_monotonic(self):
        """Test that cumulative regret is monotonically non-decreasing."""
        trajectory = [
            (1, 1.0, 1.0),
            (2, 3.0, 3.0),
            (3, 5.0, 5.0),
            (4, 7.0, 7.0),
            (5, 9.0, 9.0),
            (6, 10.0, 10.0),
        ]
        f_star = 10.0
        cum_regrets = cumulative_regret(trajectory, f_star, track_incumbent=True)
        values = [r[1] for r in cum_regrets]
        # Should be monotonically non-decreasing
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


class TestTTFO:
    """Test suite for time to first optimum (TTFO)."""

    def test_ttfo_found(self):
        """Test TTFO when optimum is found."""
        trajectory = [
            (1, 5.0, 5.0),
            (2, 6.0, 6.0),
            (3, 7.0, 7.0),
            (4, 8.0, 8.0),
            (5, 10.0, 10.0),
        ]
        f_star = 10.0
        result = ttfo(trajectory, f_star, tolerance=1e-9)
        assert result == 5

    def test_ttfo_not_found(self):
        """Test TTFO when optimum is not found."""
        trajectory = [
            (1, 5.0, 5.0),
            (2, 6.0, 6.0),
            (3, 7.0, 7.0),
        ]
        f_star = 10.0
        result = ttfo(trajectory, f_star, tolerance=1e-9)
        assert result is None

    def test_ttfo_with_tolerance(self):
        """Test TTFO with tolerance parameter."""
        trajectory = [
            (1, 5.0, 5.0),
            (2, 6.0, 6.0),
            (3, 9.99, 9.99),
        ]
        f_star = 10.0
        result = ttfo(trajectory, f_star, tolerance=0.1)
        assert result == 3


class TestMetricsIntegration:
    """Integration tests for metrics calculations."""

    def test_metrics_consistency(self):
        """Test that metrics are consistent with each other."""
        trajectory = [
            (1, 2.0, 2.0),
            (2, 4.0, 4.0),
            (3, 6.0, 6.0),
            (4, 8.0, 8.0),
            (5, 10.0, 10.0),
        ]
        f_star = 10.0

        # Simple regret at final step should be 0 when optimum is found
        final_simple_regret = simple_regret(trajectory[-1][2], f_star)
        assert final_simple_regret == 0.0

        # Instantaneous regrets should be decreasing
        inst_regrets = instantaneous_regret(trajectory, f_star, track_incumbent=True)
        regret_values = [r[1] for r in inst_regrets]
        assert all(regret_values[i] >= regret_values[i + 1] for i in range(len(regret_values) - 1))

    def test_metrics_with_realistic_optimization_run(self):
        """Test metrics on a realistic optimization trajectory."""
        # Simulate optimization trajectory: starts badly, improves, plateaus
        trajectory = [
            (1, 1.0, 1.0),
            (2, 2.0, 2.0),
            (3, 2.0, 2.0),
            (4, 3.0, 3.0),
            (5, 5.0, 5.0),
            (6, 8.0, 8.0),
            (7, 13.0, 13.0),
            (8, 13.0, 13.0),
            (9, 13.0, 13.0),
        ]
        f_star = 13.0

        # Calculate metrics
        inst_regrets = instantaneous_regret(trajectory, f_star, track_incumbent=True)
        cum_regrets = cumulative_regret(trajectory, f_star, track_incumbent=True)
        found_time = ttfo(trajectory, f_star, tolerance=1e-9)

        # Instantaneous regrets should be non-negative
        inst_values = [r[1] for r in inst_regrets]
        assert all(r >= 0 for r in inst_values)

        # Cumulative regrets should be non-decreasing
        cum_values = [r[1] for r in cum_regrets]
        assert all(cum_values[i] <= cum_values[i + 1] for i in range(len(cum_values) - 1))

        # Should find optimum
        assert found_time == 7


class TestMetricsEdgeCases:
    """Test edge cases for metrics calculations."""

    def test_single_evaluation(self):
        """Test metrics with single evaluation."""
        trajectory = [(1, 5.0, 5.0)]
        f_star = 10.0

        inst_regrets = instantaneous_regret(trajectory, f_star, track_incumbent=True)
        assert len(inst_regrets) == 1
        assert inst_regrets[0][1] == 5.0

        cum_regrets = cumulative_regret(trajectory, f_star, track_incumbent=True)
        assert len(cum_regrets) == 1
        assert cum_regrets[0][1] == 0.0  # First cumulative regret is 0

    def test_all_optimal_evaluations(self):
        """Test metrics when all evaluations are optimal."""
        trajectory = [
            (1, 10.0, 10.0),
            (2, 10.0, 10.0),
            (3, 10.0, 10.0),
            (4, 10.0, 10.0),
        ]
        f_star = 10.0

        inst_regrets = instantaneous_regret(trajectory, f_star, track_incumbent=True)
        inst_values = [r[1] for r in inst_regrets]
        assert inst_values == [0.0, 0.0, 0.0, 0.0]

    def test_largest_value_first(self):
        """Test metrics when best value appears first."""
        trajectory = [
            (1, 10.0, 10.0),
            (2, 9.0, 10.0),
            (3, 8.0, 10.0),
            (4, 7.0, 10.0),
            (5, 6.0, 10.0),
        ]
        f_star = 10.0

        inst_regrets = instantaneous_regret(trajectory, f_star, track_incumbent=True)
        inst_values = [r[1] for r in inst_regrets]
        # All should be 0 because best_value stays at 10
        assert inst_values == [0.0, 0.0, 0.0, 0.0, 0.0]


class TestHistorySeries:
    """Test history extraction helper functions."""

    def test_history_current_series_extraction(self):
        """history_current_series should extract current values from trajectory."""
        trajectory = [
            (1, 2.0, 2.0),
            (2, 3.0, 3.0),
            (3, 5.0, 5.0),
            (4, 4.0, 5.0),
            (5, 6.0, 6.0),
        ]
        current_series = history_current_series(trajectory)

        assert len(current_series) == len(trajectory)
        assert current_series == [(1, 2.0), (2, 3.0), (3, 5.0), (4, 4.0), (5, 6.0)]

    def test_history_best_series_extraction(self):
        """history_best_series should extract best values from trajectory."""
        trajectory = [
            (1, 2.0, 2.0),
            (2, 3.0, 3.0),
            (3, 5.0, 5.0),
            (4, 4.0, 5.0),
            (5, 6.0, 6.0),
        ]
        best_series = history_best_series(trajectory)

        assert len(best_series) == len(trajectory)
        assert best_series == [(1, 2.0), (2, 3.0), (3, 5.0), (4, 5.0), (5, 6.0)]

    def test_history_current_series_single_evaluation(self):
        """history_current_series should handle single-point trajectories."""
        trajectory = [(1, 7.5, 7.5)]
        current_series = history_current_series(trajectory)

        assert len(current_series) == 1
        assert current_series[0] == (1, 7.5)

    def test_history_best_series_monotonic(self):
        """Best series from history should always be non-decreasing."""
        trajectory = [
            (1, 3.0, 3.0),
            (2, 1.0, 3.0),
            (3, 5.0, 5.0),
            (4, 2.0, 5.0),
            (5, 2.0, 5.0),
        ]
        best_series = history_best_series(trajectory)
        best_values = [v for _, v in best_series]

        # Best values should be non-decreasing
        assert all(best_values[i] <= best_values[i + 1] for i in range(len(best_values) - 1))

    def test_history_series_alignment(self):
        """Current and best series should have evaluation times aligned."""
        trajectory = [
            (1, 2.0, 2.0),
            (3, 4.0, 4.0),
            (5, 3.0, 4.0),
            (8, 5.0, 5.0),
        ]
        current_series = history_current_series(trajectory)
        best_series = history_best_series(trajectory)

        # Same evaluation times
        current_times = [t for t, _ in current_series]
        best_times = [t for t, _ in best_series]
        assert current_times == best_times


class TestRuntimeProfileHelpers:
    """Test runtime profile helpers and tail-sum conversion."""

    def test_runtime_profile_single_run(self):
        trajectory = [(1, 1.0, 1.0), (3, 3.0, 3.0), (5, 4.0, 4.0)]
        levels = np.array([1.0, 2.0, 4.0, 5.0])

        np.testing.assert_array_equal(
            inv_runtime_profile_single_run(trajectory, levels),
            np.array([1.0, 3.0, 5.0, np.inf]),
        )

    def test_runtime_profile_single_run_first_hitting_found(self):
        trajectory = [
            (1, 1.0, 1.0),
            (2, 2.0, 2.0),
            (3, 1.5, 2.5),
            (4, 3.0, 3.0),
        ]
        levels = np.array([2.5])

        hitting_times = inv_runtime_profile_single_run(trajectory, levels)
        np.testing.assert_array_equal(hitting_times, np.array([3.0]))

    def test_runtime_profile_single_run_first_hitting_not_found(self):
        trajectory = [
            (1, 0.5, 0.5),
            (2, 1.0, 1.0),
            (3, 1.5, 1.5),
        ]
        levels = np.array([5.0])

        hitting_times = inv_runtime_profile_single_run(trajectory, levels)
        np.testing.assert_array_equal(hitting_times, np.array([np.inf]))

    def test_runtime_profile_single_run_first_hitting_at_start(self):
        trajectory = [
            (1, 10.0, 10.0),
            (2, 9.0, 10.0),
            (3, 8.0, 10.0),
        ]
        levels = np.array([10.0])

        hitting_times = inv_runtime_profile_single_run(trajectory, levels)
        np.testing.assert_array_equal(hitting_times, np.array([1.0]))

    def test_runtime_profile_single_run_uses_best_value(self):
        trajectory = [
            (1, 5.0, 5.0),
            (2, 3.0, 5.0),
            (3, 4.0, 6.0),
            (4, 2.0, 6.0),
        ]
        levels = np.array([5.5])

        hitting_times = inv_runtime_profile_single_run(trajectory, levels)
        np.testing.assert_array_equal(hitting_times, np.array([3.0]))

    def test_runtime_profile_single_run_multiple_levels_sequence(self):
        trajectory = [
            (1, 1.0, 1.0),
            (2, 2.0, 2.0),
            (3, 3.0, 3.0),
            (4, 4.0, 4.0),
            (5, 5.0, 5.0),
        ]
        levels = np.array([1.5, 2.5, 3.5, 4.5])

        hitting_times = inv_runtime_profile_single_run(trajectory, levels)
        np.testing.assert_array_equal(hitting_times, np.array([2.0, 3.0, 4.0, 5.0]))

    def test_runtime_profile_single_run_exact_boundary(self):
        trajectory = [
            (1, 1.0, 1.0),
            (2, 3.0, 3.0),
            (3, 2.0, 3.0),
        ]
        levels = np.array([3.0])

        hitting_times = inv_runtime_profile_single_run(trajectory, levels)
        np.testing.assert_array_equal(hitting_times, np.array([2.0]))

    def test_compute_runtime_profile(self):
        trajectories = [
            [(1, 1.0, 1.0), (2, 2.0, 2.0), (3, 3.0, 3.0)],
            [(1, 0.5, 0.5), (3, 2.5, 2.5)],
        ]
        levels = np.array([1.0, 2.0, 3.0])
        time_grid = np.array([1.0, 2.0, 3.0])

        profile = compute_inv_runtime_profile(trajectories, levels, time_grid)
        expected = np.array(
            [
                [0.5, 0.5, 1.0],
                [0.0, 0.5, 1.0],
                [0.0, 0.0, 0.5],
            ]
        )
        np.testing.assert_allclose(profile, expected)

    def test_compute_runtime_profile_single_level(self):
        trajectories = [
            [(1, 0.4, 0.4), (3, 1.2, 1.2)],
            [(1, 0.2, 0.2), (2, 0.8, 0.8)],
            [(1, 1.5, 1.5)],
        ]
        levels = np.array([1.0])
        time_grid = np.array([1.0, 2.0, 3.0])

        profile = compute_inv_runtime_profile(trajectories, levels, time_grid)
        expected = np.array([[1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0]])
        np.testing.assert_allclose(profile, expected)

    def test_profile_to_expected_cumulative_regret_unit_spacing(self):
        profile = np.array([[0.0, 1.0], [0.0, 0.0]])
        fitness_levels = np.array([1.0, 2.0])
        time_grid = np.array([1.0, 2.0])

        # At T=1: (1-0) + (1-0) = 2
        # At T=2: (1-1) + (1-0) added over time => 3 total
        ecr = inv_profile_to_expected_cumulative_regret(profile, fitness_levels, time_grid)
        np.testing.assert_allclose(ecr, np.array([2.0, 3.0]))

    def test_runtime_profile_rejects_unsorted_time_grid(self):
        trajectories = [[(1, 1.0, 1.0)]]
        levels = np.array([1.0])
        time_grid = np.array([2.0, 1.0])

        with pytest.raises(ValueError, match="time_grid"):
            compute_inv_runtime_profile(trajectories, levels, time_grid)

    def test_profile_to_expected_cumulative_regret_rejects_shape_mismatch(self):
        profile = np.zeros((2, 3))
        fitness_levels = np.array([1.0])
        time_grid = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="profile shape"):
            inv_profile_to_expected_cumulative_regret(profile, fitness_levels, time_grid)

    def test_compute_runtime_profile_rejects_empty_trajectories(self):
        levels = np.array([1.0])
        time_grid = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="at least one trajectory"):
            compute_inv_runtime_profile([], levels, time_grid)

    def test_runtime_profile_single_run_rejects_unsorted_levels(self):
        trajectory = [(1, 1.0, 1.0)]
        levels = np.array([2.0, 1.0])

        with pytest.raises(ValueError, match="sorted"):
            inv_runtime_profile_single_run(trajectory, levels)


class TestAnalysisTables:
    """Test suite for analysis.tables module."""

    def test_create_results_table_basic(self):
        """Test create_results_table produces valid DataFrame."""
        results = {
            ("RLS", 100): [{"regret": 0.0}, {"regret": 1.0}, {"regret": 2.0}],
            ("EA", 100): [{"regret": 0.5}, {"regret": 1.5}, {"regret": 2.5}],
        }
        df = create_results_table(results, budget=100)
        assert len(df) == 2
        assert "Algorithm" in df.columns
        assert "Mean SR" in df.columns  # SR = Simple Regret
        assert "Median SR" in df.columns
        assert "Std" in df.columns
        assert "IQR" in df.columns
        assert "CI (95%)" in df.columns
        assert "P(opt)" in df.columns
        assert "Runs" in df.columns

    def test_create_results_table_filters_by_budget(self):
        """Test create_results_table filters results by budget."""
        results = {
            ("RLS", 100): [{"regret": 0.0}, {"regret": 1.0}],
            ("RLS", 200): [{"regret": 0.5}, {"regret": 1.5}],
            ("EA", 100): [{"regret": 0.2}, {"regret": 0.8}],
        }
        df = create_results_table(results, budget=100)
        # Should only include results for budget=100
        assert len(df) == 2
        algorithms = df["Algorithm"].tolist()
        assert "RLS" in algorithms
        assert "EA" in algorithms

    def test_create_results_table_computes_p_opt(self):
        """Test create_results_table computes P(opt) correctly."""
        results = {
            ("RLS", 100): [
                {"regret": 0.0},
                {"regret": 0.0},
                {"regret": 1.0},
                {"regret": 2.0},
            ],
        }
        df = create_results_table(results, budget=100)
        # 2 out of 4 runs have regret < 1e-9 (i.e., regret == 0)
        p_opt = float(df.loc[df["Algorithm"] == "RLS", "P(opt)"].values[0])
        assert p_opt == 0.5

    def test_create_results_table_sorted_by_mean(self):
        """Test create_results_table sorts by mean regret ascending."""
        results = {
            ("WorstAlg", 100): [{"regret": 10.0}, {"regret": 10.0}],
            ("BestAlg", 100): [{"regret": 0.1}, {"regret": 0.1}],
            ("MidAlg", 100): [{"regret": 5.0}, {"regret": 5.0}],
        }
        df = create_results_table(results, budget=100)
        algorithms = df["Algorithm"].tolist()
        assert algorithms == ["BestAlg", "MidAlg", "WorstAlg"]

    def test_compute_table_statistics(self):
        """Test compute_table_statistics returns all expected fields."""
        regrets = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        stats = compute_table_statistics(regrets)

        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "iqr" in stats
        assert "ci_lower" in stats
        assert "ci_upper" in stats
        assert "p_opt" in stats
        assert "n_runs" in stats

        assert stats["mean"] == 2.0
        assert stats["median"] == 2.0
        assert stats["n_runs"] == 5
        assert stats["ci_lower"] <= stats["mean"] <= stats["ci_upper"]

    def test_create_aggregate_table(self):
        """Test create_aggregate_table produces cross-problem comparison."""
        results_by_problem = {
            "OneMax": {
                ("RLS", 100): [{"regret": 0.0}, {"regret": 0.5}],
                ("EA", 100): [{"regret": 1.0}, {"regret": 1.5}],
            },
            "LeadingOnes": {
                ("RLS", 100): [{"regret": 0.5}, {"regret": 1.0}],
                ("EA", 100): [{"regret": 0.1}, {"regret": 0.2}],
            },
        }
        df = create_aggregate_table(results_by_problem, budget=100)

        assert len(df) == 2
        assert "Algorithm" in df.columns
        assert "Avg Mean SR" in df.columns  # SR = Simple Regret
        assert "Avg P(opt)" in df.columns
        assert "Best Count" in df.columns
        assert "Problems" in df.columns

        # RLS is best on OneMax, EA is best on LeadingOnes
        rls_row = df[df["Algorithm"] == "RLS"].iloc[0]
        ea_row = df[df["Algorithm"] == "EA"].iloc[0]
        assert rls_row["Best Count"] == 1
        assert ea_row["Best Count"] == 1
        assert rls_row["Problems"] == 2
        assert ea_row["Problems"] == 2

    def test_export_latex_table(self, tmp_path):
        """Test export_latex_table writes valid LaTeX file."""
        import pandas as pd

        df = pd.DataFrame({"Algorithm": ["RLS", "EA"], "Mean": ["1.0", "2.0"]})
        save_path = str(tmp_path / "table.tex")
        export_latex_table(df, save_path)

        # Verify file was created and contains LaTeX table markers
        with open(save_path) as f:
            content = f.read()
        assert "\\begin{tabular}" in content
        assert "\\end{tabular}" in content
        assert "RLS" in content
        assert "EA" in content

    def test_export_table_csv(self, tmp_path):
        """Test export_table with CSV format."""
        import pandas as pd

        df = pd.DataFrame({"Algorithm": ["RLS", "EA"], "Mean": ["1.0", "2.0"]})
        save_path = tmp_path / "table.csv"
        export_table(df, save_path, TableFormat.CSV)

        with open(save_path) as f:
            content = f.read()
        assert "Algorithm,Mean" in content
        assert "RLS,1.0" in content
        assert "EA,2.0" in content

    def test_export_table_markdown(self, tmp_path):
        """Test export_table with Markdown format."""
        import pandas as pd

        df = pd.DataFrame({"Algorithm": ["RLS", "EA"], "Mean": ["1.0", "2.0"]})
        save_path = tmp_path / "table.md"
        export_table(df, save_path, TableFormat.MARKDOWN)

        with open(save_path) as f:
            content = f.read()
        assert "Algorithm" in content
        assert "RLS" in content
        assert "|" in content  # Markdown table uses pipes


class TestAnalysisProfiles:
    """Test suite for analysis.profiles module."""

    def test_run_profile_analysis_basic(self):
        """Test run_profile_analysis returns expected structure."""
        results = {
            "RLS": [
                {
                    "trajectory": [
                        (1, 1.0, 1.0),
                        (2, 2.0, 2.0),
                        (3, 3.0, 3.0),
                        (4, 4.0, 4.0),
                        (5, 5.0, 5.0),
                    ]
                },
                {
                    "trajectory": [
                        (1, 1.0, 1.0),
                        (2, 1.5, 1.5),
                        (3, 2.5, 2.5),
                        (4, 3.5, 3.5),
                        (5, 4.5, 4.5),
                    ]
                },
            ],
        }
        time_grid, fitness_levels, inv_profiles, empirical_ecr, profile_ecr = run_profile_analysis(
            results, f_star=5.0, budget=5
        )

        # Check return types
        assert isinstance(time_grid, np.ndarray)
        assert isinstance(fitness_levels, np.ndarray)
        assert isinstance(inv_profiles, dict)
        assert isinstance(empirical_ecr, dict)
        assert isinstance(profile_ecr, dict)

        # Check RLS is in results
        assert "RLS" in inv_profiles
        assert "RLS" in empirical_ecr
        assert "RLS" in profile_ecr

    def test_run_profile_analysis_multiple_algorithms(self):
        """Test run_profile_analysis handles multiple algorithms."""
        results = {
            "RLS": [
                {
                    "trajectory": [
                        (1, 1.0, 1.0),
                        (2, 2.0, 2.0),
                        (3, 3.0, 3.0),
                    ]
                },
            ],
            "EA": [
                {
                    "trajectory": [
                        (1, 1.5, 1.5),
                        (2, 2.5, 2.5),
                        (3, 3.0, 3.0),
                    ]
                },
            ],
        }
        time_grid, fitness_levels, inv_profiles, empirical_ecr, profile_ecr = run_profile_analysis(
            results, f_star=3.0, budget=3
        )

        assert "RLS" in inv_profiles
        assert "EA" in inv_profiles
        assert "RLS" in empirical_ecr
        assert "EA" in empirical_ecr


class TestMetricsEdgeCasesExtended:
    """Extended edge case tests for metrics."""

    def test_cumulative_regret_empty_trajectory(self):
        """Test cumulative_regret with empty trajectory."""
        trajectory = []
        f_star = 10.0
        cum_regrets = cumulative_regret(trajectory, f_star, track_incumbent=True)
        assert len(cum_regrets) == 0

    def test_cumulative_regret_single_point(self):
        """Test cumulative_regret with single point trajectory."""
        trajectory = [(1, 5.0, 5.0)]
        f_star = 10.0
        cum_regrets = cumulative_regret(trajectory, f_star, track_incumbent=True)
        assert len(cum_regrets) == 1
        # First cumulative regret starts from 0
        assert cum_regrets[0][1] == 0.0

    def test_instantaneous_regret_empty_trajectory(self):
        """Test instantaneous_regret with empty trajectory."""
        trajectory = []
        f_star = 10.0
        inst_regrets = instantaneous_regret(trajectory, f_star, track_incumbent=True)
        assert len(inst_regrets) == 0

    def test_instantaneous_regret_use_current(self):
        """Test instantaneous_regret with track_incumbent=False (use current value)."""
        trajectory = [
            (1, 5.0, 5.0),
            (2, 3.0, 5.0),  # current=3, best=5
            (3, 6.0, 6.0),
        ]
        f_star = 10.0
        inst_regrets = instantaneous_regret(trajectory, f_star, track_incumbent=False)
        # With track_incumbent=False, regret is based on current value
        regret_values = [r[1] for r in inst_regrets]
        # At t=2, current=3, so regret = 10 - 3 = 7
        assert regret_values[1] == 7.0

    def test_ttfo_empty_trajectory(self):
        """Test ttfo with empty trajectory."""
        trajectory = []
        f_star = 10.0
        result = ttfo(trajectory, f_star, tolerance=1e-9)
        assert result is None

    def test_ttfo_immediate_optimum(self):
        """Test ttfo when optimum is found immediately."""
        trajectory = [(1, 10.0, 10.0)]
        f_star = 10.0
        result = ttfo(trajectory, f_star, tolerance=1e-9)
        assert result == 1


class TestTimeToTarget:
    """Test suite for time-to-target regret metric."""

    def test_time_to_target_reached(self):
        """Test time_to_target when target is reached."""
        trajectory = [
            (1, 5.0, 5.0),
            (2, 7.0, 7.0),
            (3, 9.0, 9.0),
        ]
        f_star = 10.0
        # Target regret of 1.0 means fitness >= 9.0
        result = time_to_target(trajectory, f_star, target_regret=1.0)
        assert result == 3

    def test_time_to_target_not_reached(self):
        """Test time_to_target when target is never reached."""
        trajectory = [
            (1, 5.0, 5.0),
            (2, 6.0, 6.0),
        ]
        f_star = 10.0
        # Target regret of 0.5 means fitness >= 9.5 (never reached)
        result = time_to_target(trajectory, f_star, target_regret=0.5)
        assert result is None

    def test_time_to_target_immediate(self):
        """Test time_to_target when target is reached immediately."""
        trajectory = [(1, 9.5, 9.5)]
        f_star = 10.0
        result = time_to_target(trajectory, f_star, target_regret=1.0)
        assert result == 1

    def test_time_to_target_equivalence_to_ttfo(self):
        """Test that time_to_target with 0.0 regret is equivalent to ttfo."""
        trajectory = [
            (1, 5.0, 5.0),
            (2, 8.0, 8.0),
            (3, 10.0, 10.0),
        ]
        f_star = 10.0
        ttfo_result = ttfo(trajectory, f_star, tolerance=1e-9)
        ttt_result = time_to_target(trajectory, f_star, target_regret=0.0, tolerance=1e-9)
        assert ttfo_result == ttt_result == 3

    def test_time_to_target_with_tolerance(self):
        """Test time_to_target respects tolerance parameter."""
        trajectory = [
            (1, 8.0, 8.0),
            (2, 9.99, 9.99),  # Within tolerance of 10.0
        ]
        f_star = 10.0
        result = time_to_target(trajectory, f_star, target_regret=0.0, tolerance=0.1)
        assert result == 2


class TestNormalizedRegret:
    """Test suite for normalized regret metric."""

    def test_normalized_regret_optimal(self):
        """Test normalized regret at optimum."""
        nr = normalized_regret(solution_value=10.0, f_star=10.0, f_worst=0.0)
        assert nr == 0.0

    def test_normalized_regret_worst(self):
        """Test normalized regret at worst case."""
        nr = normalized_regret(solution_value=0.0, f_star=10.0, f_worst=0.0)
        assert nr == 1.0

    def test_normalized_regret_middle(self):
        """Test normalized regret at midpoint."""
        nr = normalized_regret(solution_value=5.0, f_star=10.0, f_worst=0.0)
        assert nr == 0.5

    def test_normalized_regret_scale_invariance(self):
        """Test that normalized regret is scale-invariant."""
        # Problem A: optimum=10, worst=0, solution=7 -> regret=3, NR=3/10=0.3
        nr_a = normalized_regret(solution_value=7.0, f_star=10.0, f_worst=0.0)
        # Problem B: optimum=100, worst=0, solution=70 -> regret=30, NR=30/100=0.3
        nr_b = normalized_regret(solution_value=70.0, f_star=100.0, f_worst=0.0)
        assert abs(nr_a - nr_b) < 1e-9

    def test_normalized_regret_nonzero_worst(self):
        """Test normalized regret with non-zero worst value."""
        # Jump function example: f_star=13, f_worst=1, solution=10
        nr = normalized_regret(solution_value=10.0, f_star=13.0, f_worst=1.0)
        expected = (13.0 - 10.0) / (13.0 - 1.0)  # 3/12 = 0.25
        assert abs(nr - expected) < 1e-9

    def test_normalized_regret_degenerate_optimal(self):
        """Test normalized regret when f_star == f_worst and solution is optimal."""
        nr = normalized_regret(solution_value=5.0, f_star=5.0, f_worst=5.0)
        assert nr == 0.0

    def test_normalized_regret_degenerate_suboptimal(self):
        """Test normalized regret when f_star == f_worst and solution is suboptimal."""
        nr = normalized_regret(solution_value=4.0, f_star=5.0, f_worst=5.0)
        assert nr == 1.0


class TestExpectedSimpleRegret:
    """Test suite for expected simple regret metric."""

    def test_expected_simple_regret_basic(self):
        """Test expected simple regret with basic values."""
        regrets = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        esr = expected_simple_regret(regrets)
        assert esr == 2.0

    def test_expected_simple_regret_all_optimal(self):
        """Test expected simple regret when all runs are optimal."""
        regrets = np.array([0.0, 0.0, 0.0])
        esr = expected_simple_regret(regrets)
        assert esr == 0.0

    def test_expected_simple_regret_single_run(self):
        """Test expected simple regret with single run."""
        regrets = np.array([5.0])
        esr = expected_simple_regret(regrets)
        assert esr == 5.0

    def test_expected_simple_regret_equivalence_to_compute_statistics(self):
        """Test that expected_simple_regret matches compute_statistics['mean']."""
        regrets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        esr = expected_simple_regret(regrets)
        stats_mean = compute_statistics(regrets)["mean"]
        assert esr == stats_mean

    def test_expected_simple_regret_rejects_empty(self):
        """Test that expected_simple_regret rejects empty array."""
        with pytest.raises(ValueError, match="non-empty"):
            expected_simple_regret(np.array([]))

    def test_expected_simple_regret_rejects_2d(self):
        """Test that expected_simple_regret rejects 2D array."""
        with pytest.raises(ValueError, match="non-empty 1D"):
            expected_simple_regret(np.array([[1.0, 2.0], [3.0, 4.0]]))


class TestProblemWorstValues:
    """Test suite for f_worst property in Problem classes."""

    def test_onemax_f_worst(self):
        """Test that OneMax has correct f_worst."""
        from regret.problems.pseudo_boolean import OneMax

        problem = OneMax(n=10)
        assert problem.f_worst == 0.0
        assert problem.f_star == 10.0

    def test_jump_f_worst(self):
        """Test that Jump has correct f_worst."""
        from regret.problems.pseudo_boolean import Jump

        problem = Jump(n=10, k=3)
        assert problem.f_worst == 1.0
        assert problem.f_star == 13.0

    def test_twomax_f_worst(self):
        """Test that TwoMax has correct f_worst."""
        from regret.problems.pseudo_boolean import TwoMax

        problem = TwoMax(n=10)
        assert problem.f_worst == 5.0  # (10+1)//2 = 5
        assert problem.f_star == 10.0

    def test_twomax_f_worst_odd_n(self):
        """Test TwoMax f_worst with odd dimension."""
        from regret.problems.pseudo_boolean import TwoMax

        problem = TwoMax(n=11)
        assert problem.f_worst == 6.0  # (11+1)//2 = 6
        assert problem.f_star == 11.0

    def test_hiff_f_worst(self):
        """Test that HIFF has correct f_worst."""
        from regret.problems.pseudo_boolean import HIFF

        problem = HIFF(n=8)
        assert problem.f_worst == 8.0
        # HIFF optimum: n * (log2(n) + 1) = 8 * (3 + 1) = 32
        assert problem.f_star == 32.0

    def test_normalized_regret_with_jump(self):
        """Integration test: normalized regret with Jump problem."""
        from regret.problems.pseudo_boolean import Jump

        problem = Jump(n=10, k=3)
        # Test with a solution value of 10
        nr = normalized_regret(
            solution_value=10.0,
            f_star=problem.f_star,
            f_worst=problem.f_worst,
        )
        # (13 - 10) / (13 - 1) = 3 / 12 = 0.25
        assert abs(nr - 0.25) < 1e-9
