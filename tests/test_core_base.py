"""Tests for regret.core.base module (Problem and Algorithm ABCs)."""

import pytest
import numpy as np
from regret.core.base import Problem, Algorithm


class TestProblemABC:
    """Test suite for Problem abstract base class."""

    def test_problem_is_abstract(self):
        """Test that Problem cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Problem(5)

    def test_problem_requires_evaluate_method(self, simple_problem):
        """Test that Problem subclass must implement evaluate()."""
        assert callable(simple_problem.evaluate)
        # Ensure implementation is provided by subclass, not abstract base.
        assert type(simple_problem).evaluate is not Problem.evaluate

    def test_problem_requires_get_optimum_value_method(self, simple_problem):
        """Test that Problem subclass must implement get_optimum_value()."""
        assert callable(simple_problem.get_optimum_value)
        assert type(simple_problem).get_optimum_value is not Problem.get_optimum_value

    def test_evaluate_returns_float(self, simple_problem, zero_bitstring):
        """Test that evaluate() returns a float."""
        result = simple_problem.evaluate(zero_bitstring)
        assert isinstance(result, (int, float, np.number))

    def test_evaluate_all_zeros(self, simple_problem, zero_bitstring):
        """Test evaluation on all-zeros bitstring."""
        result = simple_problem.evaluate(zero_bitstring)
        assert result == 0

    def test_evaluate_all_ones(self, simple_problem, ones_bitstring):
        """Test evaluation on all-ones bitstring."""
        result = simple_problem.evaluate(ones_bitstring)
        assert result == 10

    def test_get_optimum_value(self, simple_problem):
        """Test that get_optimum_value() returns the optimal value."""
        optimum = simple_problem.get_optimum_value()
        assert optimum == 10
        assert isinstance(optimum, (int, float, np.number))

    def test_problem_consistency(self, simple_problem, ones_bitstring):
        """Test that best solution achieves optimum value."""
        optimum = simple_problem.get_optimum_value()
        best_solution_value = simple_problem.evaluate(ones_bitstring)
        assert best_solution_value == optimum


class TestAlgorithmABC:
    """Test suite for Algorithm abstract base class."""

    def test_algorithm_is_abstract(self):
        """Test that Algorithm cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Algorithm(None)


class TestProblemEvaluation:
    """Test Problem evaluation with different bitstring patterns."""

    @pytest.mark.parametrize(
        "n_ones,expected",
        [
            (0, 0),
            (1, 1),
            (5, 5),
            (10, 10),
        ],
    )
    def test_evaluate_n_ones(self, simple_problem, n_ones, expected):
        """Test evaluation with different numbers of ones."""
        x = np.zeros(10, dtype=int)
        x[:n_ones] = 1
        result = simple_problem.evaluate(x)
        assert result == expected

    def test_evaluate_deterministic(self, simple_problem, random_bitstring):
        """Test that evaluation is deterministic."""
        result1 = simple_problem.evaluate(random_bitstring)
        result2 = simple_problem.evaluate(random_bitstring)
        assert result1 == result2
