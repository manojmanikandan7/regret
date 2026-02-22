from abc import ABC, abstractmethod

import numpy as np


class Problem(ABC):
    def __init__(self, n: int):
        self.n = n
        self.f_star = self.get_optimum_value()

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def get_optimum_value(self) -> float:
        pass


class Algorithm(ABC):
    def __init__(self, problem: Problem, seed: int | None = None):
        self.problem = problem
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Reset algorithm state"""
        self.evaluations = 0
        self.best_value = -np.inf
        self.best_solution = None
        self.history = []
        
        
    def _record_history(self, current_value):
        self.history.append((self.evaluations, current_value, self.best_value))

    @abstractmethod
    def step(self):
        pass

    def run(self, budget: int):
        self.reset()
        while self.evaluations < budget:
            self.step()
        return self.best_value, self.best_solution
