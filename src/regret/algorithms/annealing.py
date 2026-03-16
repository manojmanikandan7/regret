from dataclasses import dataclass
from typing import Protocol

import numpy as np

from regret.core.base import Algorithm, Problem


class CoolingSchedule(Protocol):
    def __call__(self, t: int) -> int | float: ...


class SimulatedAnnealing(Algorithm):
    """Simulated Annealing with configurable cooling schedule."""

    def __init__(
        self,
        problem: Problem,
        T_func: CoolingSchedule | None = None,
        min_T: float | None = None,
        seed: int | None = None,
    ):
        """Initialize simulated annealing with a cooling schedule.

        Args:
            problem: Problem to optimize.
            T_func: Temperature schedule callable; defaults to logarithmic.
            min_T: Minimum temperature clamp to avoid zero temperature.
            seed: Optional RNG seed for reproducibility.
        """
        self.T_func = T_func or LogarithmicCooling()
        self.min_T = min_T or 1e-9
        super().__init__(problem, seed)

    def reset(self):
        """Reinitialize state, solution, and trajectory for a fresh run."""
        super().reset()
        self.current = self.rng.integers(0, 2, size=self.problem.n)
        self.current_value = self.problem.evaluate(self.current)
        self.evaluations = 1
        self.best_value = self.current_value
        self.best_solution = self.current.copy()
        self._record_history(self.current_value)

    def step(self):
        """Perform one Metropolis update using the configured cooling schedule."""
        # Generate neighbour
        neighbour = self.current.copy()
        i = self.rng.integers(0, self.problem.n)
        neighbour[i] = 1 - neighbour[i]

        neighbour_value = self.problem.evaluate(neighbour)

        # NOTE: Capping down at a minimum temperature
        # TODO: Explore if it is better to stop early
        # (since it essentially reached absolute zero)
        T = max(self.min_T, self.T_func(self.evaluations))
        # Since evals = 1 initially, due to the reset,
        # incremented after temperature calculation to avoid one-off deviation
        self.evaluations += 1
        delta = neighbour_value - self.current_value

        # Metropolis acceptance criterion
        # NOTE: Original Metropolis acceptance uses exp(-delta/T), for minimisation problems
        # Here, we use exp(delta/T), since we use maximisation problems
        if delta >= 0 or self.rng.random() < np.exp(delta / T):
            self.current = neighbour
            self.current_value = neighbour_value

        if self.current_value > self.best_value:
            self.best_value = self.current_value
            self.best_solution = self.current.copy()

        self._record_history(self.current_value)


@dataclass(frozen=True)
class LogarithmicCooling(CoolingSchedule):
    """Logarithmic cooling schedule initializer."""

    d: float = 1.0

    def __call__(self, t: int) -> int | float:
        """Compute temperature at evaluation t using logarithmic cooling."""
        return self.d / np.log(t + 1)


@dataclass(frozen=True)
class ExponentialCooling(CoolingSchedule):
    """Exponential cooling schedule initializer."""

    T0: float = 1.0
    alpha: float = 0.95

    def __call__(self, t: int) -> int | float:
        """Compute temperature at evaluation t using exponential decay."""
        return self.T0 * (self.alpha**t)


@dataclass(frozen=True)
class LinearCooling(CoolingSchedule):
    """Linear cooling schedule initializer."""

    T0: float = 1.0
    Tf: float = 0.01
    max_iter: int = 10000

    def __call__(self, t: int) -> int | float:
        """Compute temperature at evaluation t using linear decay."""
        return max(self.Tf, self.T0 - (self.T0 - self.Tf) * t / self.max_iter)
