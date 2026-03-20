"""Tests for regret.core.metrics module."""

import pytest
import numpy as np
from regret.core.metrics import (
    simple_regret,
    instantaneous_regret,
    cumulative_regret,
    ttfo,
    compute_statistics,
    probability_optimal,
)


class TestSimpleRegret:
    """Test suite for simple regret calculation."""

    def test_simple_regret_zero(self):
        """Test simple regret when optimum is found."""
        solution_value = 10.0
        f_star = 10.0
        regret = simple_regret(solution_value, f_star)
        assert regret == 0.0

    def test_simple_regret_positive(self):
        """Test simple regret is positive when away from optimum."""
        solution_value = 8.0
        f_star = 10.0
        regret = simple_regret(solution_value, f_star)
        assert regret == 2.0
        assert regret > 0

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


class TestInstantaneousRegret:
    """Test suite for instantaneous regret."""

    def test_instantaneous_regret_single_evaluation(self):
        """Test instantaneous regret for a single evaluation."""
        trajectory = [(0, 8.0, 8.0)]
        f_star = 10.0
        regrets = instantaneous_regret(trajectory, f_star, use_best=True)
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
        regrets = instantaneous_regret(trajectory, f_star, use_best=True)
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
        cum_regrets = cumulative_regret(trajectory, f_star, use_best=True)
        # First entry should be 0
        assert cum_regrets[0][1] == 0.0
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
        cum_regrets = cumulative_regret(trajectory, f_star, use_best=True)
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
        cum_regrets = cumulative_regret(trajectory, f_star, use_best=True)
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
        inst_regrets = instantaneous_regret(trajectory, f_star, use_best=True)
        regret_values = [r[1] for r in inst_regrets]
        assert all(
            regret_values[i] >= regret_values[i + 1]
            for i in range(len(regret_values) - 1)
        )

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
        inst_regrets = instantaneous_regret(trajectory, f_star, use_best=True)
        cum_regrets = cumulative_regret(trajectory, f_star, use_best=True)
        found_time = ttfo(trajectory, f_star, tolerance=1e-9)

        # Instantaneous regrets should be non-negative
        inst_values = [r[1] for r in inst_regrets]
        assert all(r >= 0 for r in inst_values)

        # Cumulative regrets should be non-decreasing
        cum_values = [r[1] for r in cum_regrets]
        assert all(
            cum_values[i] <= cum_values[i + 1] for i in range(len(cum_values) - 1)
        )

        # Should find optimum
        assert found_time == 7


class TestMetricsEdgeCases:
    """Test edge cases for metrics calculations."""

    def test_single_evaluation(self):
        """Test metrics with single evaluation."""
        trajectory = [(1, 5.0, 5.0)]
        f_star = 10.0

        inst_regrets = instantaneous_regret(trajectory, f_star, use_best=True)
        assert len(inst_regrets) == 1
        assert inst_regrets[0][1] == 5.0

        cum_regrets = cumulative_regret(trajectory, f_star, use_best=True)
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

        inst_regrets = instantaneous_regret(trajectory, f_star, use_best=True)
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

        inst_regrets = instantaneous_regret(trajectory, f_star, use_best=True)
        inst_values = [r[1] for r in inst_regrets]
        # All should be 0 because best_value stays at 10
        assert inst_values == [0.0, 0.0, 0.0, 0.0, 0.0]
