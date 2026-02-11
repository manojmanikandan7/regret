import numpy as np
from regret.core.base import Problem


class MaxSAT(Problem):
    """Random k-SAT problem (maximization version)."""

    def __init__(
        self, n: int, m: int | None = None, k: int = 3, seed: int | None = None
    ):
        self.k = k
        self.m = m or 4 * n  # Clause-to-variable ratio
        self.rng = np.random.default_rng(seed)
        self._generate_clauses(n)
        super().__init__(n)

    def _generate_clauses(self, n: int):
        """Generate random k-SAT clauses."""
        self.clauses = []
        for _ in range(self.m):
            variables = self.rng.choice(n, size=self.k, replace=False)
            negations = self.rng.integers(0, 2, size=self.k)
            self.clauses.append((variables, negations))

    def evaluate(self, x: np.ndarray) -> float:
        satisfied = 0
        for variables, negations in self.clauses:
            clause_sat = False
            for var, neg in zip(variables, negations):
                if (x[var] == 1) != (neg == 1):
                    clause_sat = True
                    break
            if clause_sat:
                satisfied += 1
        return float(satisfied)

    def get_optimum_value(self) -> float:
        # For random MaxSAT, optimum is typically all clauses satisfied
        return float(self.m)
