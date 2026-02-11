from regret.core.base import Algorithm, Problem


class RLS(Algorithm):
    """Randomized Local Search / Stochastic Hill Climber."""

    def __init__(self, problem: Problem, seed: int | None = None):
        super().__init__(problem, seed)
        self.current = self.rng.integers(0, 2, size=problem.n)
        self.current_value = problem.evaluate(self.current)
        self.evaluations = 1
        self.best_value = self.current_value
        self.best_solution = self.current.copy()

    def step(self):
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


class RLSExploration(Algorithm):
    """RLS with exploration probability."""

    def __init__(
        self,
        problem: Problem,
        epsilon: float | None = None,
        decay: bool = True,
        seed: int | None = None,
    ):
        super().__init__(problem, seed)
        self.base_epsilon = epsilon or (1.0 / problem.n)
        self.decay = decay
        self.current = self.rng.integers(0, 2, size=problem.n)
        self.current_value = problem.evaluate(self.current)
        self.evaluations = 1
        self.best_value = self.current_value
        self.best_solution = self.current.copy()

    def step(self):
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
