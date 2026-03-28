"""Core abstractions for problems and algorithms."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np


class Problem(ABC):
    """Abstract pseudo-boolean optimization problem.

    Provides a common interface for defining fitness landscapes over binary
    search spaces. Subclasses implement specific objective functions.

    Attributes:
        n: Dimension of the bitstring search space.
        f_star: Cached global optimum value (computed at initialization).
        f_worst: Cached worst possible fitness value (computed at initialization).
    """

    def __init__(self, n: int):
        """Initialize problem with dimension and cache optimum.

        Args:
            n: Dimension of the bitstring search space.
        """
        self.n = n
        self.f_star = self.get_optimum_value()
        self.f_worst = self.get_worst_value()

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

    def get_worst_value(self) -> float:
        """Return the worst possible fitness value for the problem.

        The default implementation returns 0.0, which is suitable for
        most non-negative pseudo-Boolean functions where the worst case
        is the all-zeros bitstring.

        Override this method if your problem has a different worst-case value.

        Returns:
            Objective value of a worst possible solution.
        """
        return 0.0


class Algorithm(ABC):
    """Abstract optimization algorithm operating on a `Problem`.

    Provides a common interface for iterative optimization algorithms that
    operate on binary search spaces. Subclasses implement specific search
    strategies by overriding the `step` method.

    Attributes:
        problem: The problem instance being optimized.
        rng: NumPy random number generator for reproducible randomness.
        callback: Optional callback invoked at each step with signature
            (evaluations, current_value, best_value, current_solution).
        evaluations: Total number of fitness evaluations performed.
        best_value: Best fitness value observed so far.
        best_solution: Binary array representing the best solution found.
        history: List of (evaluations, current_value, best_value) tuples
            recording the search trajectory.
    """

    def __init__(
        self,
        problem: Problem,
        seed: int | None = None,
        callback: Callable[[int, float, float, np.ndarray], None] | None = None,
    ):
        """Set up algorithm state and RNG.

        Args:
            problem: Problem instance to optimize.
            seed: Optional seed for reproducible randomness.
            callback: Optional callback invoked at each step with signature
                     (evaluations, current_value, best_value, current_solution).
        """
        self.problem = problem
        self.rng = np.random.default_rng(seed)
        self.callback = callback
        self.reset()

    def reset(self):
        """Reset counters and tracking structures for a fresh run."""
        self.evaluations = 0
        self.best_value = -np.inf
        self.best_solution = None
        self.history = []

    def _record_history(self, current_value: float, current_solution: np.ndarray | None = None):
        """Append a trajectory record and invoke callback if configured.

        Args:
            current_value: Fitness value of the current candidate.
            current_solution: Binary array representing the current candidate (optional).
        """
        self.history.append((self.evaluations, current_value, self.best_value))
        if self.callback is not None and current_solution is not None:
            self.callback(self.evaluations, current_value, self.best_value, current_solution)

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
