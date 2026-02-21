import numpy as np

from regret.core.base import Algorithm, Problem


class OnePlusOneEA(Algorithm):
    """(1+1)-Evolutionary Algorithm with standard bit mutation."""

    def __init__(
        self,
        problem: Problem,
        mutation_rate: float | None = None,
        seed: int | None = None,
    ):
        self.mutation_rate = mutation_rate or (1.0 / problem.n)
        super().__init__(problem, seed)

    def reset(self):
        super().reset()
        self.current = self.rng.integers(0, 2, size=self.problem.n)
        self.current_value = self.problem.evaluate(self.current)
        self.evaluations = 1
        self.best_value = self.current_value
        self.best_solution = self.current.copy()

    def step(self):
        offspring = self.current.copy()
        for i in range(self.problem.n):
            # Flips a bit with the prob of mutation_rate
            if self.rng.random() < self.mutation_rate:
                offspring[i] = 1 - offspring[i]

        offspring_value = self.problem.evaluate(offspring)
        self.evaluations += 1

        if offspring_value >= self.current_value:
            self.current = offspring
            self.current_value = offspring_value

        if self.current_value > self.best_value:
            self.best_value = self.current_value
            self.best_solution = self.current.copy()


class MuPlusLambdaEA(Algorithm):
    """(μ+λ)-Evolutionary Algorithm."""

    def __init__(
        self,
        problem: Problem,
        mu: int = 10,
        lmbda: int = 10,
        mutation_rate: float | None = None,
        seed: int | None = None,
    ):
        self.mu = mu
        self.lmbda = lmbda
        self.mutation_rate = mutation_rate or (1.0 / problem.n)
        super().__init__(problem, seed)

    def reset(self):
        super().reset()
        # Initialize population
        self.population = [
            self.rng.integers(0, 2, size=self.problem.n) for _ in range(self.mu)
        ]
        self.fitness = [self.problem.evaluate(ind) for ind in self.population]
        self.evaluations = self.mu

        best_idx = np.argmax(self.fitness)
        self.best_value = self.fitness[best_idx]
        self.best_solution = self.population[best_idx].copy()

    def step(self):
        # Generate offspring
        offspring = []
        offspring_fitness = []

        for _ in range(self.lmbda):
            parent = self.population[self.rng.integers(0, self.mu)]
            child = parent.copy()
            for i in range(self.problem.n):
                if self.rng.random() < self.mutation_rate:
                    child[i] = 1 - child[i]

            fitness = self.problem.evaluate(child)
            offspring.append(child)
            offspring_fitness.append(fitness)
            self.evaluations += 1

        # Combine and select
        combined = self.population + offspring
        combined_fitness = self.fitness + offspring_fitness

        indices = np.argsort(combined_fitness)[-self.mu :]
        self.population = [combined[i] for i in indices]
        self.fitness = [combined_fitness[i] for i in indices]

        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.best_value:
            self.best_value = self.fitness[best_idx]
            self.best_solution = self.population[best_idx].copy()
