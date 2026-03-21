import numpy as np

from regret.core.base import Problem


class MaxkSAT(Problem):
    """Random k-SAT problem (maximization version)."""

    def __init__(
        self, n: int, m: int | None = None, k: int = 3, seed: int | None = None
    ):
        """Initialize random k-SAT instance and generate clauses.

        Args:
            n: Number of boolean variables.
            m: Number of clauses; defaults to 4n.
            k: Clause width.
            seed: Optional RNG seed for reproducibility.
        """
        self.k = k
        self.m = (
            m or 4 * n
        )  # Number of clauses; default is 4 times the number of variables
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
        """Count satisfied clauses for a candidate assignment.

        Args:
            x: Binary assignment vector of length n.

        Returns:
            Number of satisfied clauses as a float.
        """
        satisfied = 0
        for variables, negations in self.clauses:
            clause_sat = False
            for var, neg in zip(variables, negations):
                # If the variable in x is true and if the corresponding variable is not negated, the clause is satisfied
                if (x[var] == 1) != (neg == 1):
                    clause_sat = True
                    break
            if clause_sat:
                satisfied += 1
        return float(satisfied)

    def get_optimum_value(self) -> float:
        """
        Return theoretical upper bound of satisfied clauses.

        IMPORTANT NOTE: the theoretical upper bound is very rarely achievable in practice.
        This means instantaneous regret will almost never reach zero, and cumulative regret will appear artificially high.
        The choice for the optimum chosen is intentional, but not appropriate for practical analyis.
        """
        # For random MaxSAT, optimum is typically all clauses satisfied
        return float(self.m)
