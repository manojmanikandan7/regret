"""Tests for regret.problems module."""

import numpy as np
import pytest

from regret.problems.combinatorial import MaxkSAT
from regret.problems.landscapes import NKLandscape
from regret.problems.pseudo_boolean import (
    HIFF,
    BinVal,
    Jump,
    LeadingOnes,
    OneMax,
    Plateau,
    Trap,
    TwoMax,
)


class TestOneMax:
    """Test suite for OneMax problem."""

    def test_onemax_all_zeros(self):
        """Test OneMax evaluation on all zeros."""
        problem = OneMax(n=10)
        x = np.zeros(10, dtype=int)
        assert problem.evaluate(x) == 0.0

    def test_onemax_all_ones(self):
        """Test OneMax evaluation on all ones."""
        problem = OneMax(n=10)
        x = np.ones(10, dtype=int)
        assert problem.evaluate(x) == 10.0

    def test_onemax_optimum(self):
        """Test OneMax optimum value."""
        problem = OneMax(n=20)
        assert problem.get_optimum_value() == 20.0

    def test_onemax_partial_ones(self):
        """Test OneMax with partial ones."""
        problem = OneMax(n=10)
        x = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        assert problem.evaluate(x) == 3.0

    @pytest.mark.parametrize("n_bits", [1, 5, 10, 20, 100])
    def test_onemax_different_sizes(self, n_bits):
        """Test OneMax with different problem sizes."""
        problem = OneMax(n=n_bits)
        x = np.ones(n_bits, dtype=int)
        assert problem.evaluate(x) == float(n_bits)
        assert problem.get_optimum_value() == float(n_bits)


class TestLeadingOnes:
    """Test suite for LeadingOnes problem."""

    def test_leadingones_all_zeros(self):
        """Test LeadingOnes evaluation on all zeros."""
        problem = LeadingOnes(n=10)
        x = np.zeros(10, dtype=int)
        assert problem.evaluate(x) == 0.0

    def test_leadingones_all_ones(self):
        """Test LeadingOnes evaluation on all ones."""
        problem = LeadingOnes(n=10)
        x = np.ones(10, dtype=int)
        assert problem.evaluate(x) == 10.0

    def test_leadingones_partial_prefix(self):
        """Test LeadingOnes with leading ones followed by zero."""
        problem = LeadingOnes(n=10)
        x = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        assert problem.evaluate(x) == 4.0

    def test_leadingones_single_one(self):
        """Test LeadingOnes with single leading one."""
        problem = LeadingOnes(n=10)
        x = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert problem.evaluate(x) == 1.0

    def test_leadingones_optimum(self):
        """Test LeadingOnes optimum value."""
        problem = LeadingOnes(n=15)
        assert problem.get_optimum_value() == 15.0

    def test_leadingones_trailing_ones_ignored(self):
        """Test that trailing ones are not counted in LeadingOnes."""
        problem = LeadingOnes(n=10)
        x = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1])
        assert problem.evaluate(x) == 2.0


class TestJump:
    """Test suite for Jump problem."""

    def test_jump_all_zeros(self):
        """Test Jump evaluation on all zeros."""
        problem = Jump(n=10, k=3)
        x = np.zeros(10, dtype=int)
        assert problem.evaluate(x) == 0.0 + 3.0

    def test_jump_all_ones(self):
        """Test Jump evaluation on all ones."""
        problem = Jump(n=10, k=3)
        x = np.ones(10, dtype=int)
        assert problem.evaluate(x) == 10.0 + 3.0
        assert problem.evaluate(x) == problem.get_optimum_value()

    def test_jump_at_critical_point(self):
        """Test Jump evaluation on all ones."""
        problem = Jump(n=10, k=3)
        # Here, the point of jump would be at unitation n - k + 1 = 10 - 3 + 1 = 8
        x = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 1])
        assert problem.evaluate(x) == 10.0 - 8.0

    def test_jump_beyond_critical_point(self):
        """Test Jump evaluation on all ones."""
        problem = Jump(n=10, k=4)
        # Here, the point of jump would be at unitation n - k + 1 = 10 - 4 + 1 = 7
        # So, simulating bitstring with unitation 9
        x = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1])
        assert problem.evaluate(x) == 10.0 - 9.0

    def test_jump_optimum(self):
        """Test Jump optimum value."""
        problem = Jump(n=10, k=3)
        assert problem.get_optimum_value() == 10.0 + 3.0

    def test_jump_different_k_values(self):
        """Test Jump with different k parameter values."""
        for k in [2, 3, 4, 5]:
            problem = Jump(n=20, k=k)
            assert problem.get_optimum_value() == float(20 + k)


class TestTwoMax:
    """Test suite for TwoMax problem (two local optima)."""

    def test_twomax_optimum(self):
        """Test TwoMax optimum value."""
        problem = TwoMax(n=10)
        optimum = problem.get_optimum_value()
        assert optimum == 10.0

    def test_twomax_all_zeros(self):
        """Test TwoMax evaluation on all zeros."""
        problem = TwoMax(n=10)
        x = np.zeros(10, dtype=int)
        assert problem.evaluate(x) == 10.0

    def test_twomax_all_ones(self):
        """Test TwoMax evaluation on all ones."""
        problem = TwoMax(n=10)
        x = np.ones(10, dtype=int)
        assert problem.evaluate(x) == 10.0

    def test_twomax_middle_point(self):
        """Balanced strings are lower-fitness than either global optimum."""
        problem = TwoMax(n=10)
        x = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        assert problem.evaluate(x) == 5.0


class TestBinVal:
    """Test suite for BinVal problem (binary value problem)."""

    def test_binval_optimum(self):
        """Test BinVal optimum value."""
        problem = BinVal(n=10)
        assert problem.get_optimum_value() == float(2**10 - 1)

    def test_binval_exact_binary_weights(self):
        """Bit i contributes 2^i to the objective."""
        problem = BinVal(n=5)
        x = np.array([1, 1, 0, 0, 1])
        assert problem.evaluate(x) == float(1 + 2 + 16)

    def test_binval_all_zeros(self):
        """All-zero string should evaluate to zero."""
        problem = BinVal(n=8)
        x = np.zeros(8, dtype=int)
        assert problem.evaluate(x) == 0.0


class TestTrap:
    """Test suite for Trap problem."""

    def test_trap_all_zeros(self):
        """Test Trap evaluation on all zeros."""
        problem = Trap(n=5, k=5)
        x = np.zeros(5, dtype=int)
        assert problem.evaluate(x) == 0.0

    def test_trap_all_ones(self):
        """Test Trap evaluation on all ones."""
        problem = Trap(n=5, k=5)
        x = np.ones(5, dtype=int)
        assert problem.evaluate(x) == 5.0

    def test_trap_partial_ones(self):
        """Test Trap with partial ones."""
        problem = Trap(n=5, k=5)
        x = np.array([1, 1, 0, 0, 0])
        assert problem.evaluate(x) == 2.0

    def test_trap_classic_deceptive_case(self):
        """For k=1, all zeros are deceptive local optimum with value n-1."""
        problem = Trap(n=6, k=1)
        x = np.zeros(6, dtype=int)
        assert problem.evaluate(x) == 5.0


class TestPlateau:
    """Test suite for Plateau problem."""

    def test_plateau_outside_flat_region(self):
        """Below n-k, Plateau behaves like OneMax."""
        problem = Plateau(n=10, k=3)
        x = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        assert problem.evaluate(x) == 5.0

    def test_plateau_flat_region(self):
        """n-k <= ones < n maps to constant value n-k."""
        problem = Plateau(n=10, k=3)
        x = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        assert problem.evaluate(x) == 7.0

    def test_plateau_global_optimum(self):
        """All ones should still be the unique optimum."""
        problem = Plateau(n=10, k=3)
        x = np.ones(10, dtype=int)
        assert problem.evaluate(x) == 10.0
        assert problem.get_optimum_value() == 10.0


class TestHIFF:
    """Test suite for HIFF problem."""

    def test_hiff_requires_power_of_two_n(self):
        """HIFF rejects non power-of-two sizes."""
        with pytest.raises(ValueError, match="power of 2"):
            HIFF(n=6)

    @pytest.mark.parametrize("bit", [0, 1])
    def test_hiff_uniform_strings_are_optimal(self, bit):
        """All-zero and all-one strings are global optima."""
        problem = HIFF(n=8)
        x = np.full(8, bit, dtype=int)
        assert problem.evaluate(x) == problem.get_optimum_value() == 32.0

    def test_hiff_mixed_string_has_lower_fitness(self):
        """A mixed string should score lower than the uniform optimum."""
        problem = HIFF(n=8)
        x = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        assert problem.evaluate(x) < problem.get_optimum_value()


class TestNKLandscape:
    """Test suite for NKLandscape."""

    def test_nk_caps_k_at_n_minus_one(self):
        """k should be capped internally to n-1."""
        problem = NKLandscape(n=5, k=99, seed=1)
        assert problem.k == 4

    def test_nk_seed_reproducibility(self):
        """Same seed should produce identical landscape evaluations."""
        x = np.array([1, 0, 1, 1, 0, 1], dtype=int)
        p1 = NKLandscape(n=6, k=2, seed=123)
        p2 = NKLandscape(n=6, k=2, seed=123)
        assert p1.evaluate(x) == p2.evaluate(x)

    def test_nk_fitness_range(self):
        """Fitness is normalized by n and should lie in [0, 1]."""
        problem = NKLandscape(n=10, k=2, seed=1)
        x = np.zeros(10, dtype=int)
        fitness = problem.evaluate(x)
        assert 0.0 <= fitness <= 1.0

    def test_nk_optimum_value_small_n_matches_search(self):
        """For n <= 20, get_optimum_value uses exhaustive search."""
        problem = NKLandscape(n=4, k=2, seed=5)
        brute_force_best = max(
            problem.evaluate(np.array([int(b) for b in format(i, "04b")], dtype=int)) for i in range(2**4)
        )
        assert problem.get_optimum_value() == brute_force_best

    def test_nk_optimum_value_large_n_is_heuristic_bound(self):
        """For n > 20, code returns a fixed heuristic value."""
        problem = NKLandscape(n=21, k=2, seed=5)
        assert problem.get_optimum_value() == 1.0


class TestMaxkSAT:
    """Test suite for MaxkSAT."""

    def test_maxksat_seed_reproducibility(self):
        """Same seed should generate identical clause sets."""
        p1 = MaxkSAT(n=10, m=20, k=3, seed=42)
        p2 = MaxkSAT(n=10, m=20, k=3, seed=42)

        assert len(p1.clauses) == len(p2.clauses) == 20
        for (v1, n1), (v2, n2) in zip(p1.clauses, p2.clauses, strict=False):
            assert np.array_equal(v1, v2)
            assert np.array_equal(n1, n2)

    def test_maxksat_evaluate_known_instance(self):
        """Evaluation should count satisfied clauses exactly."""
        problem = MaxkSAT(n=3, m=2, k=2, seed=0)
        problem.clauses = [
            (np.array([0, 1]), np.array([0, 1])),
            (np.array([1, 2]), np.array([1, 0])),
        ]

        x = np.array([1, 0, 0], dtype=int)
        assert problem.evaluate(x) == 2.0

        x_unsat = np.array([0, 1, 1], dtype=int)
        assert problem.evaluate(x_unsat) == 1.0

    def test_maxksat_optimum_upper_bound_is_m(self):
        """Reported optimum value is the theoretical upper bound m."""
        problem = MaxkSAT(n=7, m=11, k=3, seed=7)
        assert problem.get_optimum_value() == 11.0


class TestProblemIntegration:
    """Integration tests for Problem implementations."""

    @pytest.mark.parametrize(
        "ProblemClass,n_bits,kwargs",
        [
            (OneMax, 10, {}),
            (LeadingOnes, 10, {}),
            (Jump, 10, {"k": 3}),
            (TwoMax, 10, {}),
            (Trap, 5, {}),
            (Plateau, 10, {"k": 3}),
        ],
    )
    def test_optimum_achievable(self, ProblemClass, n_bits, kwargs):
        """Test that all-ones solution achieves optimum value."""
        problem = ProblemClass(n=n_bits, **kwargs)
        x_optimum = np.ones(n_bits, dtype=int)
        optimum_value = problem.get_optimum_value()
        achieved_value = problem.evaluate(x_optimum)
        assert achieved_value == optimum_value

    def test_evaluation_deterministic(self):
        """Test that evaluation is deterministic across problem types."""
        problems = [
            OneMax(n=10),
            LeadingOnes(n=10),
            TwoMax(n=10),
            Plateau(n=10, k=3),
            Trap(n=10, k=1),
        ]
        x = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

        for problem in problems:
            result1 = problem.evaluate(x)
            result2 = problem.evaluate(x)
            assert result1 == result2
