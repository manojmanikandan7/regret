from typing import Protocol

import numpy as np

from regret.core.base import Algorithm, Problem


class CoolingSchedule(Protocol):
    def __call__(self, t: int, **kwargs: float) -> int | float: ...


class SimulatedAnnealing(Algorithm):
    """Simulated Annealing with configurable cooling schedule."""

    def __init__(
        self,
        problem: Problem,
        T_func: CoolingSchedule | None = None,
        seed: int | None = None,
    ):
        self.T_func = T_func or logarithmic_cooling
        super().__init__(problem, seed)

    def reset(self):
        super().reset()
        self.current = self.rng.integers(0, 2, size=self.problem.n)
        self.current_value = self.problem.evaluate(self.current)
        self.evaluations = 1
        self.best_value = self.current_value
        self.best_solution = self.current.copy()
        self._record_history(self.current_value)

    def step(self):
        # Generate neighbor
        neighbour = self.current.copy()
        i = self.rng.integers(0, self.problem.n)
        neighbour[i] = 1 - neighbour[i]

        neighbor_value = self.problem.evaluate(neighbour)
        self.evaluations += 1

        # Metropolis acceptance criterion
        T = self.T_func(self.evaluations)
        delta = neighbor_value - self.current_value

        if delta >= 0 or self.rng.random() < np.exp(delta / T):
            self.current = neighbour
            self.current_value = neighbor_value

        if self.current_value > self.best_value:
            self.best_value = self.current_value
            self.best_solution = self.current.copy()
        
        self._record_history(self.current_value)


def logarithmic_cooling(t: int) -> int | float:
    """Logarithmic cooling schedule."""
    return 1.0 / np.log(t + 2)


def exponential_cooling(t: int, T0: float = 1.0, alpha: float = 0.95) -> int | float:
    """Exponential cooling schedule."""
    return T0 * (alpha**t)


def linear_cooling(
    t: int, T0: float = 1.0, Tf: float = 0.01, max_iter: int = 10000
) -> int | float:
    """Linear cooling schedule."""
    return max(Tf, T0 - (T0 - Tf) * t / max_iter)
