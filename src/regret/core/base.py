"""Core abstractions for problems and algorithms."""

from abc import ABC, abstractmethod

import numpy as np


class Problem(ABC):
    """Abstract pseudo-boolean optimization problem."""

    def __init__(self, n: int):
        """Initialize problem with dimension and cache optimum.

        Args:
            n: Dimension of the bitstring search space.
        """
        self.n = n
        self.f_star = self.get_optimum_value()

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the fitness of a candidate solution.

        Args:
            x: Binary vector representing a candidate solution.

        Returns:
            Fitness value of the candidate.
        """
        pass

    @abstractmethod
    def get_optimum_value(self) -> float:
        """Return the known global optimum value for the problem.

        Returns:
            Objective value of an optimal solution.
        """
        pass


class Algorithm(ABC):
    """Abstract optimization algorithm operating on a `Problem`."""

    def __init__(self, problem: Problem, seed: int | None = None):
        """Set up algorithm state and RNG.

        Args:
            problem: Problem instance to optimize.
            seed: Optional seed for reproducible randomness.
        """
        self.problem = problem
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Reset counters and tracking structures for a fresh run."""
        self.evaluations = 0
        self.best_value = -np.inf
        self.best_solution = None
        self.history = []

    def _record_history(self, current_value):
        """Append a trajectory record of evaluations, current, and best value."""
        self.history.append((self.evaluations, current_value, self.best_value))

    @abstractmethod
    def step(self):
        """Perform one iteration of the algorithm (in-place state update)."""
        pass

    def run(self, budget: int) -> tuple[float, np.ndarray | None]:
        """Execute the algorithm for a fixed evaluation budget.

        Args:
            budget: Maximum number of evaluations to perform.

        Returns:
            Tuple of (best_value, best_solution) observed during the run.
        """
        self.reset()
        while self.evaluations < budget:
            self.step()
        return self.best_value, self.best_solution
