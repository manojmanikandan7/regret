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
        history_before = len(ea.history)
        ea.step()

        assert ea.evaluations == evals_before + ea.lmbda
        # (μ+λ)-EA records each offspring evaluation during a generation.
        assert len(ea.history) == history_before + ea.lmbda
        assert ea.history[-1][0] == ea.evaluations


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
        "AlgorithmClass,kwargs,steps",
        [
            (RLS, {}, 7),
            (OnePlusOneEA, {"mutation_rate": 0.1}, 7),
            (SimulatedAnnealing, {"T_func": LogarithmicCooling(d=2.0)}, 7),
        ],
    )
    def test_history_evaluations_alignment(
        self, simple_problem, AlgorithmClass, kwargs, steps
    ):
        """History timestamps should stay aligned with evaluation counts."""
        algo = AlgorithmClass(problem=simple_problem, seed=123, **kwargs)
        algo.reset()

        for _ in range(steps):
            algo.step()

        eval_points = [t for t, _, _ in algo.history]
        assert eval_points == sorted(eval_points)
        assert eval_points[-1] == algo.evaluations

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

    def test_algorithm_run_respects_budget(self, simple_problem):
        """Algorithm.run(budget) should respect evaluation budget exactly."""
        for budget in [1, 5, 10, 50, 100]:
            rls = RLS(problem=simple_problem, seed=42)
            best_value, best_solution = rls.run(budget)
            
            # Budget constraint: evaluations recorded includes the reset evaluation
            assert rls.evaluations == budget
            assert len(rls.history) == budget

    def test_algorithm_run_returns_best_solution(self, simple_problem):
        """Algorithm.run() should return best_value and best_solution found."""
        rls = RLS(problem=simple_problem, seed=7)
        best_value, best_solution = rls.run(budget=50)
        
        # Return values should be the tracked best
        assert best_value == rls.best_value
        assert best_solution is rls.best_solution
        
        # best_solution should evaluate to best_value
        if best_solution is not None:
            eval_value = simple_problem.evaluate(best_solution)
            assert abs(eval_value - best_value) < 1e-9

    def test_algorithm_run_resets_history(self, simple_problem):
        """Algorithm.run() should reset history and start fresh."""
        rls = RLS(problem=simple_problem, seed=99)
        
        # First run
        rls.run(budget=20)
        first_evaluations = rls.evaluations
        
        # Second run should reset
        rls.run(budget=30)
        second_evaluations = rls.evaluations
        
        # Should not accumulate - second run resets
        assert second_evaluations == 30
        assert rls.evaluations == second_evaluations

    def test_algorithm_run_reproducibility(self, simple_problem):
        """Same seed should produce identical results across multiple runs."""
        seed = 123
        budget = 40
        
        rls1 = RLS(problem=simple_problem, seed=seed)
        best_val1, best_sol1 = rls1.run(budget=budget)
        history1 = list(rls1.history)
        
        rls2 = RLS(problem=simple_problem, seed=seed)
        best_val2, best_sol2 = rls2.run(budget=budget)
        history2 = list(rls2.history)
        
        assert best_val1 == best_val2
        assert np.array_equal(best_sol1, best_sol2)
        assert history1 == history2

    @pytest.mark.parametrize(
        "AlgorithmClass,kwargs",
        [
            (RLS, {}),
            (OnePlusOneEA, {"mutation_rate": 0.1}),
            (SimulatedAnnealing, {"T_func": LogarithmicCooling(d=2.0)}),
        ],
    )
    def test_algorithm_run_budget_constraint(
        self, simple_problem, AlgorithmClass, kwargs
    ):
        """All algorithms should respect budget constraint."""
        budget = 25
        alg = AlgorithmClass(problem=simple_problem, seed=42, **kwargs)
        best_value, best_solution = alg.run(budget)
        
        assert alg.evaluations == budget
        assert len(alg.history) == budget
        assert best_value <= simple_problem.get_optimum_value()

    def test_algorithm_run_best_solution_matches_value(self, simple_problem):
        """best_solution should always achieve best_value on evaluation."""
        rls = RLS(problem=simple_problem, seed=55)
        best_value, best_solution = rls.run(budget=60)
        
        if best_solution is not None:
            # Evaluate the solution and verify it matches
            solution_eval = simple_problem.evaluate(best_solution)
            assert solution_eval == best_value
            
            # Also verify this value is in history
            history_best_values = [best for _, _, best in rls.history]
            assert best_value in history_best_values
