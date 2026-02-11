import numpy as np
from regret.core.base import Problem


class OneMax(Problem):
    """Count number of ones in binary string."""

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(x))

    def get_optimum_value(self) -> float:
        return float(self.n)


class LeadingOnes(Problem):
    """Count leading ones before first zero."""

    def evaluate(self, x: np.ndarray) -> float:
        for i in range(self.n):
            if x[i] == 0:
                return float(i)
        return float(self.n)

    def get_optimum_value(self) -> float:
        return float(self.n)


class Jump(Problem):
    """Jump function with gap of size k."""

    def __init__(self, n: int, k: int = 3):
        self.k = k
        super().__init__(n)

    def evaluate(self, x: np.ndarray) -> float:
        ones = np.sum(x)
        if ones == self.n or ones <= self.n - self.k:
            return float(ones)
        return float(self.n - ones)

    def get_optimum_value(self) -> float:
        return float(self.n)


class TwoMax(Problem):
    """Two global optima: all zeros or all ones."""

    def evaluate(self, x: np.ndarray) -> float:
        ones = np.sum(x)
        return float(max(ones, self.n - ones))

    def get_optimum_value(self) -> float:
        return float(self.n)
