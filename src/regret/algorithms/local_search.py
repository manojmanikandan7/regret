"""Local search algorithms for pseudo-boolean optimization.

This module implements local search algorithms that explore the search space
by iteratively modifying candidate solutions. These algorithms are simple yet
effective baselines for benchmarking on binary optimization problems.

Algorithms:
    RLS: Randomized Local Search (single bit-flip hill climber).
    RLSExploration: RLS with configurable exploration probability.
"""

from collections.abc import Callable

import numpy as np

from regret.core.base import Algorithm, Problem


class RLS(Algorithm):
    """Randomized Local Search (Stochastic Hill Climber).

    A simple local search that flips one random bit per iteration and accepts
    the neighbor if it is at least as good as the current solution. This is
    one of the simplest randomized optimization algorithms.

    Attributes:
        current: Binary array representing the current candidate solution.
        current_value: Fitness value of the current solution.
    """

    def __init__(
        self,
        problem: Problem,
        seed: int | None = None,
        callback: Callable[[int, float, float, np.ndarray], None] | None = None,
    ):
        """Initialize RLS with optional RNG seed.

        Args:
            problem: Problem to optimize.
            seed: Optional seed for reproducibility.
            callback: Optional callback invoked at each step.
        """
        super().__init__(problem, seed, callback)

    def reset(self):
        """Initialize state, solution, and tracking for a fresh run."""
        super().reset()
        self.current = self.rng.integers(0, 2, size=self.problem.n)
        self.current_value = self.problem.evaluate(self.current)
        self.evaluations = 1
        self.best_value = self.current_value
        self.best_solution = self.current.copy()
        self._record_history(self.current_value, self.current)

    def step(self):
        """Perform one bit-flip move with hill-climbing acceptance."""
        neighbor = self.current.copy()
        i = self.rng.integers(0, self.problem.n)
        neighbor[i] = 1 - neighbor[i]

        neighbor_value = self.problem.evaluate(neighbor)
        self.evaluations += 1

        if neighbor_value >= self.current_value:
            self.current = neighbor
            self.current_value = neighbor_value

        if self.current_value > self.best_value:
            self.best_value = self.current_value
            self.best_solution = self.current.copy()

        self._record_history(self.current_value, self.current)


class RLSExploration(Algorithm):
    """RLS with exploration probability.

    Extends RLS by occasionally accepting non-improving moves with probability
    epsilon, allowing escape from local optima. Based on Hoos & Stützle (2004),
    "Stochastic Local Search: Foundations and Applications", Chapter 2.

    Attributes:
        base_epsilon: Initial exploration probability (defaults to 1/n).
        decay: Whether to decay epsilon over evaluations.
        current: Binary array representing the current candidate solution.
        current_value: Fitness value of the current solution.
    """

    def __init__(
        self,
        problem: Problem,
        epsilon: float | None = None,
        decay: bool = True,
        seed: int | None = None,
        callback: Callable[[int, float, float, np.ndarray], None] | None = None,
    ):
        """Initialize exploratory RLS with exploration rate settings.

        Args:
            problem: Problem to optimize.
            epsilon: Base exploration probability; defaults to 1/n.
            decay: Whether to decay epsilon over evaluations.
            seed: Optional seed for reproducibility.
            callback: Optional callback invoked at each step.
        """
        self.base_epsilon = epsilon or (1.0 / problem.n)
        self.decay = decay
        super().__init__(problem, seed, callback)

    def reset(self):
        """Initialize state, solution, and history for a fresh exploratory run."""
        super().reset()
        self.current = self.rng.integers(0, 2, size=self.problem.n)
        self.current_value = self.problem.evaluate(self.current)
        self.evaluations = 1
        self.best_value = self.current_value
        self.best_solution = self.current.copy()
        self._record_history(self.current_value, self.current)

    def step(self):
        """Perform one iteration combining exploration and local search."""
        epsilon = self.base_epsilon / self.evaluations if self.decay else self.base_epsilon

        neighbor = self.current.copy()
        i = self.rng.integers(0, self.problem.n)
        neighbor[i] = 1 - neighbor[i]

        neighbor_value = self.problem.evaluate(neighbor)
        self.evaluations += 1

        if self.rng.random() < epsilon:
            # Uninformed Random Walk step: move to a random neighbor unconditionally
            # Accept without fitness gate
            self.current = neighbor
            self.current_value = neighbor_value
        else:
            if neighbor_value >= self.current_value:
                self.current = neighbor
                self.current_value = neighbor_value

        if self.current_value > self.best_value:
            self.best_value = self.current_value
            self.best_solution = self.current.copy()

        self._record_history(self.current_value, self.current)
