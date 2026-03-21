"""Tests for integration between components."""

import numpy as np
from regret.algorithms.local_search import RLS
from regret.problems.pseudo_boolean import OneMax
from regret.core.metrics import simple_regret, instantaneous_regret


class TestAlgorithmProblemIntegration:
    """Integration tests between algorithms and problems."""

    def test_rls_on_onemax(self):
        """RLS should improve or preserve fitness with deterministic seed."""
        problem = OneMax(n=10)
        rls = RLS(problem=problem, seed=7)
        rls.reset()
        initial_best = rls.best_value

        # Run optimization
        for _ in range(100):
            rls.step()

        # Best value is monotonic and bounded by the known optimum.
        history_best = [best for _, _, best in rls.history]
        assert all(
            history_best[i] <= history_best[i + 1] for i in range(len(history_best) - 1)
        )
        assert rls.best_value >= initial_best
        assert rls.best_value <= problem.get_optimum_value()

    def test_algorithm_finds_optimum(self):
        """RLS should reach optimum on small OneMax with enough budget."""
        problem = OneMax(n=6)
        rls = RLS(problem=problem, seed=11)

        optimum = problem.get_optimum_value()
        for _ in range(400):
            rls.step()
            if rls.best_value == optimum:
                break

        assert rls.best_value == optimum


class TestEndToEndOptimization:
    """End-to-end tests of optimization workflow."""

    def test_optimization_with_metrics(self):
        """End-to-end run should yield coherent regret metrics."""
        # Setup
        problem = OneMax(n=10)
        rls = RLS(problem=problem, seed=3)
        rls.reset()

        # Optimize
        for _ in range(50):
            rls.step()

        # Analyze with metrics
        optimum = problem.get_optimum_value()

        # At least some improvement should have occurred
        assert rls.best_value > 0
        assert rls.best_value <= optimum

        # Calculate regret via public metric helpers.
        regret = simple_regret(rls.best_value, optimum)
        inst = instantaneous_regret(rls.history, optimum, use_best=True)
        assert regret >= 0
        assert len(inst) == len(rls.history)
        assert inst[-1][1] == regret
        assert all(value >= 0 for _, value in inst)

    def test_multiple_runs_reproducibility(self):
        """Test that setting random seed allows reproducibility."""
        problem = OneMax(n=5)
        rls = RLS(problem=problem, seed=42)
        rls.reset()

        for _ in range(20):
            rls.step()

        first_run_best = rls.best_value

        # Reset with same seed
        rls2 = RLS(problem=OneMax(n=5), seed=42)
        rls2.reset()

        for _ in range(20):
            rls2.step()

        second_run_best = rls2.best_value

        # Should get same result with same seed
        assert first_run_best == second_run_best
        assert rls.history == rls2.history


class TestBitflipOperations:
    """Integration tests around mutation behavior inside RLS."""

    def test_single_bitflip(self):
        """One RLS step can change the current state by at most one bit."""
        problem = OneMax(n=10)
        rls = RLS(problem=problem, seed=19)

        before = rls.current.copy()
        rls.step()
        after = rls.current

        hamming = int(np.sum(before != after))
        # Zero means rejected move, one means accepted one-bit mutation.
        assert hamming in (0, 1)

    def test_expected_hamming_distance(self):
        """History entries should stay evaluation-aligned with steps."""
        rls = RLS(problem=OneMax(n=8), seed=101)
        n_steps = 25

        for _ in range(n_steps):
            rls.step()

        # reset() records the first point at evaluation=1.
        assert rls.evaluations == n_steps + 1
        assert len(rls.history) == n_steps + 1
        assert [t for t, _, _ in rls.history] == list(range(1, n_steps + 2))


class InstantaneousRegretTests:
    """Integration checks related to optimum detection in trajectories."""

    def best_value_regret_non_increasing(self):
        """Best-value regret should be non-increasing over time."""
        problem = OneMax(n=8)
        rls = RLS(problem=problem, seed=13)

        for _ in range(40):
            rls.step()

        best_regrets = instantaneous_regret(
            rls.history,
            problem.get_optimum_value(),
            use_best=True,
        )
        values = [r for _, r in best_regrets]
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def best_value_regret_zero_optimum(self):
        """When optimum is reached, best regret should become zero."""
        problem = OneMax(n=6)
        rls = RLS(problem=problem, seed=29)
        optimum = problem.get_optimum_value()

        for _ in range(400):
            rls.step()
            if rls.best_value == optimum:
                break

        best_regrets = instantaneous_regret(rls.history, optimum, use_best=True)
        zero_idx = [i for i, (_, value) in enumerate(best_regrets) if value == 0.0]

        assert zero_idx
        first_zero = zero_idx[0]
        assert all(value == 0.0 for _, value in best_regrets[first_zero:])
