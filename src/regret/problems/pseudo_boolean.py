"""Classic pseudo-boolean optimization problems.

This module defines standard benchmark problems for evaluating optimization
algorithms on binary search spaces. These problems are widely used in the
theory of evolutionary computation and have well-understood properties.

Problems:
    OneMax: Maximize the count of ones (simple unimodal baseline).
    LeadingOnes: Maximize the leading-ones prefix length.
    Jump: OneMax with a fitness valley near the optimum.
    TwoMax: Two global optima at all-zeros and all-ones.
    BinVal: Binary value with exponential bit weights.
    Trap: Deceptive function with local optimum at all-zeros.
    Plateau: OneMax with a flat fitness region near optimum.
    HIFF: Hierarchical If-and-only-If function with multi-level structure.
"""

import numpy as np

from regret.core.base import Problem


class OneMax(Problem):
    """Count the number of ones in a binary string.

    The simplest pseudo-boolean benchmark: fitness equals the number of 1-bits.
    Unimodal with smooth gradient toward the all-ones optimum. Used as a
    baseline to verify algorithm correctness and basic performance.

    Reference:
        Droste, S., Jansen, T. and Wegener, I., 2002.
        On the analysis of the (1+1) evolutionary algorithm.
        Theoretical Computer Science, 276(1-2), pp.51-81.
    """

    def evaluate(self, x: np.ndarray) -> float:
        """Compute fitness as the count of ones.

        Args:
            x: Binary vector.

        Returns:
            Number of ones in the vector.
        """
        return float(np.sum(x))

    def get_optimum_value(self) -> float:
        """Return the global optimum value."""
        return float(self.n)


class LeadingOnes(Problem):
    """Count leading ones before the first zero.

    Fitness equals the length of the contiguous prefix of 1-bits. Requires
    bits to be set in order from left to right, making it harder than OneMax
    for many algorithms due to sequential dependencies.
    """

    def evaluate(self, x: np.ndarray) -> float:
        """Count leading ones until the first zero.

        Args:
            x: Binary vector.

        Returns:
            Length of the leading-ones prefix.
        """
        for i in range(self.n):
            if x[i] == 0:
                return float(i)
        return float(self.n)

    def get_optimum_value(self) -> float:
        """Return the global optimum value."""
        return float(self.n)


class Jump(Problem):
    """Jump function with a fitness gap of size k.

    Creates a valley in the fitness landscape near the optimum. Solutions with
    ones in the range [n-k+1, n-1] have reduced fitness, requiring algorithms
    to "jump" across k bits simultaneously to reach the global optimum.

    Attributes:
        k: Gap width defining the jump valley.

    Reference:
        Droste, S., Jansen, T. and Wegener, I., 2002.
        On the analysis of the (1+1) evolutionary algorithm.
        Theoretical Computer Science, 276(1-2), pp.51-81.
    """

    def __init__(self, n: int, k: int = 3):
        """Initialize Jump problem with dimension and gap.

        Args:
            n: Dimension of the bitstring.
            k: Gap width defining the jump valley.
        """
        self.k = k
        super().__init__(n)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate Jump fitness with the defined gap of size k.

        Args:
            x: Binary vector.

        Returns:
            Fitness value with valley penalization near the optimum.
        """
        ones = np.sum(x)
        # If there are n ones or if the number of ones is less than or equals to (n - k), return the ones
        # Else, return (n - number of ones)
        if ones == self.n or ones <= self.n - self.k:
            return float(ones) + self.k
        return float(self.n - ones)

    def get_optimum_value(self) -> float:
        """Return the global optimum value."""
        return float(self.n + self.k)

    def get_worst_value(self) -> float:
        """Return the worst possible fitness value.

        The worst case for Jump occurs at n-1 ones (just before the optimum),
        where the fitness is n - (n-1) = 1.
        """
        return 1.0


class TwoMax(Problem):
    """Two global optima: all zeros or all ones.

    Bimodal fitness landscape where both the all-zeros and all-ones bitstrings
    are global optima. Tests an algorithm's ability to commit to one basin of
    attraction rather than oscillating between them.
    """

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate fitness as distance to either all-ones or all-zeros optimum.

        Args:
            x: Binary vector.

        Returns:
            Fitness favoring the closer global optimum.
        """
        ones = np.sum(x)
        return float(max(ones, self.n - ones))

    def get_optimum_value(self) -> float:
        """Return the global optimum value."""
        return float(self.n)


class BinVal(Problem):
    """Binary value: weighted sum with exponential weights.

    Treats the bitstring as a binary number with position i having weight 2^i.
    Creates a highly non-uniform fitness landscape where later bits contribute
    exponentially more to fitness than earlier bits.
    """

    def evaluate(self, x: np.ndarray) -> float:
        """Compute weighted binary value of the bitstring given by x.

        Args:
            x: Binary vector.

        Returns:
            Numeric value of the bitstring.
        """
        # Use Python integers (arbitrary precision) to avoid overflow/precision loss.
        weights = [2**i for i in range(self.n)]
        return float(sum(int(x[i]) * weights[i] for i in range(self.n)))

    def get_optimum_value(self) -> float:
        """Return the maximum achievable binary value."""
        return float(2**self.n - 1)
        # NOTE: For large n this float is rounded, but it equals evaluate(all-ones)
        # by the same rounding, so regret still reaches 0 correctly.


class Trap(Problem):
    """Trap function with deceptive attractor.

    A deceptive function where the local gradient points toward the all-zeros
    bitstring (deceptive attractor), but the global optimum is at all-ones.
    The parameter k controls the width of the deceptive region.

    With k=1, this is the classical fully deceptive trap: f(u) = n-1-u for
    u < n, f(n) = n, where u is the number of ones.

    Attributes:
        k: Width of the deceptive region; slope change occurs at n-k ones.

    Reference:
        Deb, K. and Goldberg, D.E., 1993. Analyzing deception in trap functions.
        In Foundations of genetic algorithms (Vol. 2, pp. 93-108). Elsevier.
    """

    def __init__(self, n: int, k: int = 1):
        """Initialize Trap with dimension and deception width.

        Args:
            n: Dimension of the bitstring.
            k: Width of the deceptive region.
                (n - k) gives the unitation z, at which the slope change occurs.
        """
        self.k = k
        super().__init__(n)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate Trap fitness accounting for the deception width k.

        Args:
            x: Binary vector.

        Returns:
            Fitness penalizing solutions near the deceptive attractor.
        """
        ones = int(np.sum(x))
        z = self.n - self.k
        return float(z - ones if ones <= z else self.n * (ones - z) / self.k)

    def get_optimum_value(self) -> float:
        """Return the global optimum value."""
        return float(self.n)


class Plateau(Problem):
    """OneMax with a flat fitness region near the optimum.

    Returns the number of ones, but creates a plateau (constant fitness) when
    the count is between (n-k) and (n-1). Tests an algorithm's ability to
    navigate neutral fitness landscapes via random drift.

    Attributes:
        k: Width of the flat fitness region before the optimum.
    """

    def __init__(self, n: int, k: int = 3):
        """Initialize Plateau with dimension and plateau width.

        Args:
            n: Dimension of the bitstring.
            k: Width of the flat fitness region.
        """
        self.k = k
        super().__init__(n)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate OneMax with a plateau near the optimum.

        Args:
            x: Binary vector.

        Returns:
            Fitness value with flat region before the optimum.
        """
        ones = np.sum(x)
        if ones == self.n:
            return float(self.n)
        elif ones >= self.n - self.k:
            # Plateau region
            return float(self.n - self.k)
        else:
            return float(ones)

    def get_optimum_value(self) -> float:
        """Return the global optimum value."""
        return float(self.n)


class HIFF(Problem):
    """Hierarchical If-and-only-If function.

    Rewards blocks of identical bits at multiple hierarchical levels. At each
    level, adjacent blocks of matching bits contribute additional fitness. The
    global optimum is achieved when all bits are identical (all-zeros or
    all-ones), creating a hierarchical structure that tests building-block
    assembly.

    The problem size n must be a power of 2.
    """

    def __init__(self, n: int):
        """Initialize HIFF with a power-of-two dimension.

        Args:
            n: Dimension of the bitstring; must be a power of two.

        Raises:
            ValueError: If n is not a power of two or is less than 2.
        """
        # Ensure n is a power of 2
        if n & (n - 1) != 0 or n < 2:
            raise ValueError("HIFF requires n to be a power of 2")
        super().__init__(n)

    def _hiff_value(self, block: np.ndarray) -> tuple[float, int | None]:
        """Recursively compute HIFF contribution for a block.

        Args:
            block: Subsequence of the bitstring.

        Returns:
            Tuple of (fitness contribution, consensus bit or None).
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
        """Evaluate HIFF fitness for the full bitstring.

        Args:
            x: Binary vector.

        Returns:
            Fitness value normalized by hierarchy contributions.
        """
        fitness, _ = self._hiff_value(x)
        return fitness

    def get_optimum_value(self) -> float:
        """Return the analytical optimum value for HIFF."""
        # Optimum is when all bits are identical (all 0s or all 1s)
        # At each level k (0 to log2(n)), we get n contributions
        # Total = n * (log2(n) + 1)
        levels = int(np.log2(self.n)) + 1
        return float(self.n * levels)
