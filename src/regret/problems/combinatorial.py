"""Combinatorial optimization problems encoded as pseudo-boolean functions.

This module provides optimization problems from combinatorial domains (e.g.,
constraint satisfaction) encoded as fitness functions over binary strings.
These problems often have known structure that enables analysis of algorithm
behavior on realistic instances.

Problems:
    MaxkSAT: Random k-SAT maximization (count satisfied clauses).
    PetersenColoringMaxSAT: Graph coloring of Petersen graph as MaxSAT.
"""

import numpy as np

from regret.core.base import Problem


class MaxkSAT(Problem):
    """Random k-SAT problem (maximization version).

    Generates a random k-SAT formula with m clauses over n variables, where
    each clause contains k literals. Fitness equals the number of satisfied
    clauses under a given variable assignment.

    Attributes:
        k: Number of literals per clause.
        m: Total number of clauses.
        rng: NumPy random generator for reproducible clause generation.
        clauses: List of (variables, negations) tuples defining each clause.
    """

    def __init__(self, n: int, m: int | None = None, k: int = 3, seed: int | None = None):
        """Initialize random k-SAT instance and generate clauses.

        Args:
            n: Number of boolean variables.
            m: Number of clauses; defaults to 4n.
            k: Clause width.
            seed: Optional RNG seed for reproducibility.
        """
        self.k = k
        self.m = m or 4 * n  # Number of clauses; default is 4 times the number of variables
        self.rng = np.random.default_rng(seed)
        self._generate_clauses(n)
        super().__init__(n)

    def _generate_clauses(self, n: int):
        """Generate random k-SAT clauses.

        Args:
            n: Number of boolean variables.
        """
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
            for var, neg in zip(variables, negations, strict=False):
                # If the variable in x is true and if the corresponding variable is not negated, the clause is satisfied
                if (x[var] == 1) != (neg == 1):
                    clause_sat = True
                    break
            if clause_sat:
                satisfied += 1
        return float(satisfied)

    def get_optimum_value(self) -> float:
        """Return theoretical upper bound of satisfied clauses.

        Returns:
            Total number of clauses m (all satisfied).

        Note:
            The theoretical upper bound is very rarely achievable in practice
            for random instances. This means instantaneous regret will almost
            never reach zero, and cumulative regret will appear artificially
            high. This choice is intentional for theoretical analysis but may
            not be appropriate for practical comparisons.
        """
        # For random MaxSAT, optimum is typically all clauses satisfied
        return float(self.m)


class PetersenColoringMaxSAT(MaxkSAT):
    """MaxSAT encoding of 3-coloring the Petersen graph.

    Encodes the graph coloring problem as a MaxSAT instance where each vertex
    must be assigned exactly one of 3 colors, and adjacent vertices cannot
    share a color. The Petersen graph has chromatic number 3, so a valid
    3-coloring exists and all 85 clauses can be satisfied simultaneously.

    This makes it superior to random MaxkSAT for regret analysis, as simple
    regret genuinely reaches zero for strong algorithms.

    Info:
        # Petersen Coloring
        ## Variable layout

        The bitstring has length n = N_VERTICES * K_COLORS = 30.
        Variable at index (v * K_COLORS + c) is 1 if vertex v is assigned
        colour c, and 0 otherwise.

        ## Clause types (all hard constraints encoded as soft clauses)

        1. At-least-one per vertex (length 3, 10 clauses):

            x[v][0] OR x[v][1] OR x[v][2]

        Satisfied when the vertex receives at least one colour.

        2. At-most-one per vertex (length 2, 30 clauses):

            ¬x[v][c1] OR ¬x[v][c2]   for each pair c1 < c2

        Satisfied when the vertex does not hold two colours simultaneously.

        3. Edge conflict per colour (length 2, 45 clauses):

            ¬x[u][c] OR ¬x[v][c]   for each edge (u,v) and colour c

        Satisfied when adjacent vertices do not share the same colour.

        Total: 85 clauses.  Clause lengths are mixed (2 and 3), so this is
        general MaxSAT, not Max-3-SAT.  The existing evaluate() loop in
        MaxkSAT is clause-length agnostic and handles this without change.

        ## Why the optimum is exactly $m$

        The Petersen graph has chromatic number 3, so a valid 3-colouring
        exists.  Any valid colouring satisfies all 85 clauses simultaneously,
        giving get_optimum_value() == m == 85.  This means:

        - Simple regret genuinely reaches zero for strong algorithms.

        - TTFO markers fire correctly.

        - Convergence probability rises to 1 with sufficient budget.

        This is the key advantage over random MaxkSAT, whose declared optimum
        is almost never achievable (as noted in that class's docstring).

        ## Petersen graph topology

        Vertices 0-4: outer pentagon   (0-1-2-3-4-0)

        Vertices 5-9: inner pentagram  (5-7-9-6-8-5)

        Spokes:                        (0-5, 1-6, 2-7, 3-8, 4-9)

    Attributes:
        N_VERTICES: Number of vertices in the Petersen graph (10).
        K_COLORS: Number of colors available (3).
        EDGES: List of edges defining the Petersen graph topology.

    """

    N_VERTICES: int = 10
    K_COLORS: int = 3
    EDGES: list[tuple[int, int]] = [
        # Outer pentagon
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        # Inner pentagram
        (5, 7),
        (7, 9),
        (9, 6),
        (6, 8),
        (8, 5),
        # Spokes
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
    ]

    def __init__(self):
        """Initialize the Petersen graph coloring MaxSAT instance.

        Creates a 30-variable MaxSAT instance with 85 clauses encoding the
        3-coloring constraints for the Petersen graph. No randomness is
        involved; the instance is fully deterministic.
        """
        # n = number of binary variables (one per vertex-colour pair)
        n = self.N_VERTICES * self.K_COLORS  # 30
        # self.k is inherited from MaxkSAT but meaningless here since
        # clause lengths are mixed.  Set to None for clarity.
        self.k = None
        self._generate_clauses(n)  # sets self.clauses, self.m
        Problem.__init__(self, n)  # sets self.n, self.f_star

    def _generate_clauses(self, n: int) -> None:
        """Build all 85 clauses deterministically from the graph structure.

        Args:
            n: Number of binary variables (must be N_VERTICES * K_COLORS).
        """
        k = self.K_COLORS
        self.clauses = []

        for v in range(self.N_VERTICES):
            base = v * k  # index of x[v][0] in the bitstring

            # --- At-least-one (length-3 clause) ---
            # x[v][0] OR x[v][1] OR x[v][2]
            self.clauses.append(
                (
                    np.array([base, base + 1, base + 2]),
                    np.array([0, 0, 0]),  # no negation: literal is true when bit=1
                )
            )

            # --- At-most-one (length-2 clauses, one per colour pair) ---
            # ¬x[v][c1] OR ¬x[v][c2]
            for c1 in range(k):
                for c2 in range(c1 + 1, k):
                    self.clauses.append(
                        (
                            np.array([base + c1, base + c2]),
                            np.array([1, 1]),  # both negated: true when bit=0
                        )
                    )

        # --- Edge conflict (length-2 clause per edge per colour) ---
        # ¬x[u][c] OR ¬x[v][c]
        for u, v in self.EDGES:
            for c in range(k):
                self.clauses.append(
                    (
                        np.array([u * k + c, v * k + c]),
                        np.array([1, 1]),
                    )
                )

        self.m = len(self.clauses)  # always 85 for the Petersen graph

    def get_optimum_value(self) -> float:
        """Return m: all clauses can be satisfied for the Petersen graph.

        Returns:
            Total number of clauses (85), since the Petersen graph is
            3-colorable and a valid coloring satisfies all constraints.
        """
        return float(self.m)
