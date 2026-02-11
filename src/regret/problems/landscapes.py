import numpy as np
from regret.core.base import Problem


class NKLandscape(Problem):
    """NK-landscape with tunable ruggedness."""

    def __init__(self, n: int, k: int = 2, seed: int | None = None):
        self.k = min(k, n - 1)
        self.rng = np.random.default_rng(seed)
        self._initialize_landscape(n)
        super().__init__(n)

    def _initialize_landscape(self, n: int):
        """Generate random fitness contributions."""
        self.contributions = []
        for i in range(n):
            table_size = 2 ** (self.k + 1)
            self.contributions.append(self.rng.random(table_size))

    def evaluate(self, x: np.ndarray) -> float:
        fitness = 0.0
        for i in range(self.n):
            indices = [i] + [(i + j + 1) % self.n for j in range(self.k)]
            key = sum(x[idx] * (2**j) for j, idx in enumerate(indices))
            fitness += self.contributions[i][key]
        return fitness / self.n

    def get_optimum_value(self) -> float:
        # For NK landscapes, optimum is unknown a priori
        # Use exhaustive search for small n or estimate
        if self.n <= 20:
            best = -np.inf
            for i in range(2**self.n):
                x = np.array([int(b) for b in format(i, f"0{self.n}b")])
                best = max(best, self.evaluate(x))
            return best
        else:
            # Estimate: assume max contribution from each position
            return 1.0
