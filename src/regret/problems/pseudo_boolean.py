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
    """
    Jump function with gap of size k.
    [n - k + 1, n] represents the "valley" to climb to reach maxima (local: n - k + 1, global: n)
    """

    def __init__(self, n: int, k: int = 3):
        self.k = k
        super().__init__(n)

    def evaluate(self, x: np.ndarray) -> float:
        ones = np.sum(x)
        # If there are n ones or if the number of ones is less than or equals to (n - k), return the ones
        # Else, return (n - number of ones)
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


class BinVal(Problem):
    """Binary value: weighted sum with exponential weights."""

    def evaluate(self, x: np.ndarray) -> float:
        # Use Python integers (arbitrary precision) to avoid overflow/precision loss.
        weights = [2**i for i in range(self.n)]
        return float(sum(int(x[i]) * weights[i] for i in range(self.n)))

    def get_optimum_value(self) -> float:
        return float(2**self.n - 1)
        # NOTE: For large n this float is rounded, but it equals evaluate(all-ones)
        # by the same rounding, so regret still reaches 0 correctly.


class Trap(Problem):
    """
    Trap function (Deb & Goldberg, FOGA 1992, Eq. 10).

    Deceptive attractor at all zeros (value = n - k), global optimum at all
    ones (value = n), trough at u (i.e., number of ones) = n - k (value = 0). k = 1 is the classical
    fully deceptive trap: f(u) = n-1-u for u < n, f(n) = n.
    """

    def __init__(self, n: int, k: int = 1):
        self.k = k
        super().__init__(n)

    def evaluate(self, x: np.ndarray) -> float:
        ones = int(np.sum(x))
        z = self.n - self.k
        return float(z - ones if ones <= z else self.n * (ones - z) / self.k)

    def get_optimum_value(self) -> float:
        return float(self.n)


class Plateau(Problem):
    """
    OneMax with a flat region.

    Returns the number of ones, but creates a plateau (flat fitness)
    when the number of ones is between (n - k) and (n - 1).
    """

    def __init__(self, n: int, k: int = 3):
        self.k = k
        super().__init__(n)

    def evaluate(self, x: np.ndarray) -> float:
        ones = np.sum(x)
        if ones == self.n:
            return float(self.n)
        elif ones >= self.n - self.k:
            # Plateau region
            return float(self.n - self.k)
        else:
            return float(ones)

    def get_optimum_value(self) -> float:
        return float(self.n)


class HIFF(Problem):
    """
    Hierarchical If-and-only-If function.

    Rewards blocks of identical bits at multiple hierarchical levels.
    The problem size n must be a power of 2.
    """

    def __init__(self, n: int):
        # Ensure n is a power of 2
        if n & (n - 1) != 0 or n < 2:
            raise ValueError("HIFF requires n to be a power of 2")
        super().__init__(n)

    def _hiff_value(self, block: np.ndarray) -> tuple[float, int | None]:
        """
        Compute HIFF value for a block.

        Returns (fitness contribution, consensus value or None).
        Consensus is 0 if all zeros, 1 if all ones, None otherwise.
        """
        if len(block) == 1:
            return 1.0, int(block[0])

        mid = len(block) // 2
        left_fit, left_val = self._hiff_value(block[:mid])
        right_fit, right_val = self._hiff_value(block[mid:])

        fitness = left_fit + right_fit

        # Check if both halves have same consensus
        if left_val is not None and right_val is not None and left_val == right_val:
            fitness += len(block)
            return fitness, left_val
        else:
            return fitness, None

    def evaluate(self, x: np.ndarray) -> float:
        fitness, _ = self._hiff_value(x)
        return fitness

    def get_optimum_value(self) -> float:
        # Optimum is when all bits are identical (all 0s or all 1s)
        # At each level k (0 to log2(n)), we get n contributions
        # Total = n * (log2(n) + 1)
        levels = int(np.log2(self.n)) + 1
        return float(self.n * levels)
