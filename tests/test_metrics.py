"""Tests for regret.core.metrics module."""

import numpy as np
import pytest

from regret.core.metrics import (
    compute_inv_runtime_profile,
    compute_statistics,
    cumulative_regret,
    history_best_series,
    history_current_series,
    instantaneous_regret,
    inv_profile_to_expected_cumulative_regret,
    inv_runtime_profile_single_run,
    probability_optimal,
    simple_regret,
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
