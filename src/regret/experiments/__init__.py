"""Experiment pipeline package.

This module re-exports shared experiment utilities for a stable import surface
while internals are split across cli/validation/orchestration/utils modules.
"""

from collections.abc import Callable

from regret.algorithms.annealing import (
    ExponentialCooling,
    LinearCooling,
    LogarithmicCooling,
    SimulatedAnnealing,
)
from regret.algorithms.evolutionary import MuPlusLambdaEA, OnePlusOneEA
from regret.algorithms.local_search import RLS, RLSExploration
from regret.core.base import Algorithm, Problem
from regret.problems.combinatorial import MaxkSAT, PetersenColoringMaxSAT
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

# Registries
PROBLEM_REGISTRY: dict[str, type[Problem]] = {
    "OneMax": OneMax,
    "LeadingOnes": LeadingOnes,
    "TwoMax": TwoMax,
    "Jump": Jump,
    "Trap": Trap,
    "BinVal": BinVal,
    "Plateau": Plateau,
    "HIFF": HIFF,
    "NKLandscape": NKLandscape,
    "MaxkSAT": MaxkSAT,
    "PetersenColoringMaxSAT": PetersenColoringMaxSAT,
}

ALGORITHM_REGISTRY: dict[str, type[Algorithm]] = {
    "RLS": RLS,
    "RLSExploration": RLSExploration,
    "OnePlusOneEA": OnePlusOneEA,
    "SimulatedAnnealing": SimulatedAnnealing,
    "SA-Log": SimulatedAnnealing,
    "SA-Lin": SimulatedAnnealing,
    "SA-Exp": SimulatedAnnealing,
    "MuPlusLambdaEA": MuPlusLambdaEA,
}

COOLING_REGISTRY: dict[str, Callable[..., Callable[[int], float]]] = {
    "logarithmic": LogarithmicCooling,
    "linear": LinearCooling,
    "exponential": ExponentialCooling,
}

__all__ = [
    "PROBLEM_REGISTRY",
    "ALGORITHM_REGISTRY",
    "COOLING_REGISTRY",
]
