"""Pytest configuration and shared fixtures for regret tests."""

import numpy as np
import pytest

from regret.core.base import Problem
from regret.problems.pseudo_boolean import LeadingOnes, OneMax


class SimpleProblem(Problem):
    """Simple test problem for algorithm testing."""

    def evaluate(self, x: np.ndarray) -> float:
        """Count number of ones in the bitstring."""
        return float(np.sum(x))

    def get_optimum_value(self) -> float:
        """Optimum is all ones."""
        return float(self.n)


@pytest.fixture
def simple_problem():
    """Fixture providing a simple OneMax problem."""
    return SimpleProblem(n=10)


@pytest.fixture
def onemax_problem():
    """Fixture providing a OneMax problem."""
    return OneMax(n=10)


@pytest.fixture
def leadingones_problem():
    """Fixture providing a LeadingOnes problem."""
    return LeadingOnes(n=10)


@pytest.fixture
def random_bitstring():
    """Fixture providing a random bitstring."""
    rng = np.random.default_rng(123)
    return rng.integers(0, 2, size=10)


@pytest.fixture
def zero_bitstring():
    """Fixture providing an all-zeros bitstring."""
    return np.zeros(10, dtype=int)


@pytest.fixture
def ones_bitstring():
    """Fixture providing an all-ones bitstring."""
    return np.ones(10, dtype=int)
