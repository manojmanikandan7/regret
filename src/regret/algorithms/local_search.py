from regret.core.base import Algorithm, Problem


class RLS(Algorithm):
    """Randomized Local Search / Stochastic Hill Climber."""

    def __init__(self, problem: Problem, seed: int | None = None):
        """Initialize RLS with optional RNG seed.

        Args:
            problem: Problem to optimize.
            seed: Optional seed for reproducibility.
        """
        super().__init__(problem, seed)

    def reset(self):
        """Initialize state, solution, and tracking for a fresh run."""
        super().reset()
        self.current = self.rng.integers(0, 2, size=self.problem.n)
        self.current_value = self.problem.evaluate(self.current)
        self.evaluations = 1
        self.best_value = self.current_value
        self.best_solution = self.current.copy()
        self._record_history(self.current_value)

    def step(self):
        """Perform one bit-flip move with hill-climbing acceptance."""
        neighbor = self.current.copy()
        i = self.rng.integers(0, self.problem.n)
        neighbor[i] = 1 - neighbor[i]

        neighbor_value = self.problem.evaluate(neighbor)
        self.evaluations += 1

        if neighbor_value >= self.current_value:
            self.current = neighbor
            self.current_value = neighbor_value

        if self.current_value > self.best_value:
            self.best_value = self.current_value
            self.best_solution = self.current.copy()

        self._record_history(self.current_value)


class RLSExploration(Algorithm):
    """RLS with exploration probability."""

    def __init__(
        self,
        problem: Problem,
        epsilon: float | None = None,
        decay: bool = True,
        seed: int | None = None,
    ):
        """Initialize exploratory RLS with exploration rate settings.

        Args:
            problem: Problem to optimize.
            epsilon: Base exploration probability; defaults to 1/n.
            decay: Whether to decay epsilon over evaluations.
            seed: Optional seed for reproducibility.
        """
        self.base_epsilon = epsilon or (1.0 / problem.n)
        self.decay = decay
        super().__init__(problem, seed)

    def reset(self):
        """Initialize state, solution, and history for a fresh exploratory run."""
        super().reset()
        self.current = self.rng.integers(0, 2, size=self.problem.n)
        self.current_value = self.problem.evaluate(self.current)
        self.evaluations = 1
        self.best_value = self.current_value
        self.best_solution = self.current.copy()
        self._record_history(self.current_value)

    def step(self):
        """Perform one iteration combining exploration and local search."""
        epsilon = (
            self.base_epsilon / self.evaluations if self.decay else self.base_epsilon
        )

        if self.rng.random() < epsilon:
            # Exploration: random move
            neighbor = self.rng.integers(0, 2, size=self.problem.n)
        else:
            # Exploitation: local search
            neighbor = self.current.copy()
            i = self.rng.integers(0, self.problem.n)
            neighbor[i] = 1 - neighbor[i]

        neighbor_value = self.problem.evaluate(neighbor)
        self.evaluations += 1

        if neighbor_value >= self.current_value:
            self.current = neighbor
            self.current_value = neighbor_value

        if self.current_value > self.best_value:
            self.best_value = self.current_value
            self.best_solution = self.current.copy()

        self._record_history(self.current_value)
