"""Tests for regret.algorithms module."""

import pytest
import numpy as np
from regret.algorithms.local_search import RLS, RLSExploration
from regret.algorithms.evolutionary import OnePlusOneEA, MuPlusLambdaEA
from regret.algorithms.annealing import (
    SimulatedAnnealing,
    LogarithmicCooling,
    ExponentialCooling,
)
from regret.problems.pseudo_boolean import OneMax


class TestRLS:
    """Test suite for RLS (Random Local Search) algorithm."""

    def test_rls_initialization(self, simple_problem):
        """Test RLS can be initialized with a problem."""
        rls = RLS(problem=simple_problem)
        assert rls.problem == simple_problem

    def test_rls_reset(self, simple_problem):
        """Test RLS reset functionality."""
        rls = RLS(problem=simple_problem, seed=1)
        rls.reset()
        assert rls.evaluations == 1
        assert len(rls.history) == 1
        assert rls.best_solution is not None

    def test_rls_step(self, simple_problem):
        """Test RLS performs a single step on the problem."""
        rls = RLS(problem=simple_problem, seed=2)
        rls.reset()
        evals_before = rls.evaluations
        history_before = len(rls.history)
        rls.step()

        assert rls.evaluations == evals_before + 1
        assert len(rls.history) == history_before + 1
        assert rls.best_value >= rls.history[0][2]

    def test_rls_convergence_onemax(self, onemax_problem):
        """Test that RLS can improve solution on OneMax."""
        rls = RLS(problem=onemax_problem)
        rls.reset()
        initial_fitness = rls.best_value

        # Run multiple steps
        for _ in range(100):
            rls.step()

        # After many steps, should find better solutions
        final_fitness = rls.best_value
        assert final_fitness >= initial_fitness

    def test_rls_trajectory_length(self, simple_problem):
        """Test that RLS tracks evaluation history."""
        rls = RLS(problem=simple_problem, seed=3)
        rls.reset()

        # Run a few steps
        n_steps = 5
        for _ in range(n_steps):
            rls.step()

        # Includes one initial point from reset plus one per step.
        assert hasattr(rls, "history")
        assert len(rls.history) == n_steps + 1
        assert rls.history[-1][0] == rls.evaluations


class TestRLSExploration:
    """Test suite for RLS with exploration (epsilon-decay)."""

    def test_rls_exploration_initialization(self, simple_problem):
        """Test RLSExploration can be initialized."""
        rls_exp = RLSExploration(problem=simple_problem, epsilon=0.5)
        assert rls_exp.problem == simple_problem

    def test_rls_exploration_reset(self, simple_problem):
        """Test RLSExploration reset functionality."""
        rls_exp = RLSExploration(problem=simple_problem, epsilon=0.5, seed=4)
        rls_exp.reset()
        assert rls_exp.best_solution is not None
        assert rls_exp.evaluations == 1
        assert len(rls_exp.history) == 1


class TestOnePlusOneEA:
    """Test suite for (1+1)-EA algorithm."""

    def test_one_plus_one_ea_initialization(self, simple_problem):
        """Test (1+1)-EA can be initialized."""
        ea = OnePlusOneEA(problem=simple_problem, mutation_rate=0.1)
        assert ea.problem == simple_problem

    def test_one_plus_one_ea_reset(self, simple_problem):
        """Test (1+1)-EA reset functionality."""
        ea = OnePlusOneEA(problem=simple_problem, mutation_rate=0.1, seed=5)
        ea.reset()
        assert ea.best_solution is not None
        assert ea.evaluations == 1
        assert len(ea.history) == 1

    def test_one_plus_one_ea_step(self, simple_problem):
        """Test (1+1)-EA performs a single step."""
        ea = OnePlusOneEA(problem=simple_problem, mutation_rate=0.1, seed=6)
        ea.reset()
        evals_before = ea.evaluations
        ea.step()
        assert ea.evaluations == evals_before + 1
        assert len(ea.history) == 2

    def test_one_plus_one_ea_mutation_rate_bounds(self, simple_problem):
        """Test (1+1)-EA with different mutation rates."""
        for mutation_rate in [0.01, 0.1, 0.5, 1.0]:
            ea = OnePlusOneEA(
                problem=simple_problem, mutation_rate=mutation_rate, seed=7
            )
            ea.reset()
            before = ea.best_value
            ea.step()
            assert ea.best_value >= before


class TestMuPlusLambdaEA:
    """Test suite for (μ+λ)-EA algorithm."""

    def test_mu_plus_lambda_ea_initialization(self, simple_problem):
        """Test (μ+λ)-EA can be initialized."""
        ea = MuPlusLambdaEA(problem=simple_problem, mu=5, lmbda=10, mutation_rate=0.1)
        assert ea.problem == simple_problem

    def test_mu_plus_lambda_ea_reset(self, simple_problem):
        """Test (μ+λ)-EA reset functionality."""
        ea = MuPlusLambdaEA(
            problem=simple_problem, mu=5, lmbda=10, mutation_rate=0.1, seed=8
        )
        ea.reset()
        assert ea.best_solution is not None
        assert ea.evaluations == 5
        assert len(ea.history) == 1

    def test_mu_plus_lambda_ea_step(self, simple_problem):
        """Test (μ+λ)-EA performs a single step."""
        ea = MuPlusLambdaEA(
            problem=simple_problem, mu=5, lmbda=10, mutation_rate=0.1, seed=9
        )
        ea.reset()
        evals_before = ea.evaluations
        ea.step()
        assert ea.evaluations == evals_before + ea.lmbda
        assert len(ea.history) == 2


class TestSimulatedAnnealing:
    """Test suite for Simulated Annealing algorithm."""

    def test_sa_initialization(self, simple_problem):
        """Test Simulated Annealing can be initialized."""
        cooling = LogarithmicCooling(d=10.0)
        sa = SimulatedAnnealing(problem=simple_problem, T_func=cooling)
        assert sa.problem == simple_problem

    def test_sa_reset(self, simple_problem):
        """Test SA reset functionality."""
        cooling = LogarithmicCooling(d=10.0)
        sa = SimulatedAnnealing(problem=simple_problem, T_func=cooling, seed=10)
        sa.reset()
        assert sa.best_solution is not None
        assert sa.evaluations == 1
        assert len(sa.history) == 1

    def test_sa_step(self, simple_problem):
        """Test SA performs a single step."""
        cooling = LogarithmicCooling(d=10.0)
        sa = SimulatedAnnealing(problem=simple_problem, T_func=cooling, seed=11)
        sa.reset()
        evals_before = sa.evaluations
        sa.step()
        assert sa.evaluations == evals_before + 1
        assert len(sa.history) == 2


class TestCoolingSchedules:
    """Test suite for cooling schedules."""

    def test_logarithmic_cooling_initialization(self):
        """Test LogarithmicCooling can be initialized."""
        cooling = LogarithmicCooling(d=10.0)
        assert cooling.d == 10.0

    def test_logarithmic_cooling_decreases(self):
        """Test that temperature decreases over time."""
        cooling = LogarithmicCooling(d=10.0)
        t1 = cooling(1)
        t10 = cooling(10)
        t50 = cooling(50)
        assert t1 >= t10 >= t50

    def test_exponential_cooling_initialization(self):
        """Test ExponentialCooling can be initialized."""
        cooling = ExponentialCooling(T0=10.0, alpha=0.99)
        assert cooling.T0 == 10.0

    def test_exponential_cooling_decreases(self):
        """Test that temperature decreases exponentially."""
        cooling = ExponentialCooling(T0=10.0, alpha=0.99)
        t1 = cooling(1)
        t10 = cooling(10)
        assert t1 >= t10
        # Verify it's actually exponential decay
        assert t1 > t10


class TestAlgorithmIntegration:
    """Integration tests for algorithms."""

    @pytest.mark.parametrize(
        "AlgorithmClass,kwargs",
        [
            (RLS, {"problem": None}),
            (RLSExploration, {"problem": None, "epsilon": 0.5}),
            (OnePlusOneEA, {"problem": None, "mutation_rate": 0.1}),
        ],
    )
    def test_algorithm_run_steps(self, simple_problem, AlgorithmClass, kwargs):
        """Test that all algorithms can run multiple steps."""
        params = dict(kwargs)
        params["problem"] = simple_problem
        params["seed"] = 12
        algo = AlgorithmClass(**params)
        algo.reset()

        # Run multiple steps without errors
        for _ in range(10):
            algo.step()

        assert len(algo.history) >= 2
        assert algo.best_value == max(best for _, _, best in algo.history)

    def test_algorithms_improve_solution(self):
        """Test that algorithms improve solutions on OneMax."""
        onemax = OneMax(n=10)

        algorithms = [
            RLS(problem=onemax, seed=13),
            OnePlusOneEA(problem=onemax, mutation_rate=0.1, seed=14),
        ]

        for algo in algorithms:
            algo.reset()
            initial_best = algo.best_value

            # Run algorithm
            for _ in range(50):
                algo.step()

            final_best = algo.best_value
            history_best = [best for _, _, best in algo.history]
            assert all(
                history_best[i] <= history_best[i + 1]
                for i in range(len(history_best) - 1)
            )
            assert final_best >= initial_best
            assert final_best == history_best[-1]
            assert final_best <= onemax.get_optimum_value()
