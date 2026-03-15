"""Experiment pipeline package.

This module re-exports shared experiment utilities for a stable import surface
while internals are split across cli/validation/orchestration/utils modules.
"""

from typing import Callable
from regret.core.base import Problem, Algorithm

from regret.problems.pseudo_boolean import (
    OneMax,
    LeadingOnes,
    TwoMax,
    Jump,
    Trap,
    BinVal,
    Plateau,
    HIFF
)
from regret.problems.combinatorial import MaxkSAT
from regret.problems.landscapes import NKLandscape


from regret.algorithms.annealing import (
    SimulatedAnnealing,
    LogarithmicCooling,
    LinearCooling,
    ExponentialCooling,
)

from regret.algorithms.local_search import RLS, RLSExploration
from regret.algorithms.evolutionary import OnePlusOneEA, MuPlusLambdaEA

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
}

ALGORITHM_REGISTRY: dict[str, type[Algorithm]] = {
    "RLS": RLS,
    "RLSExploration": RLSExploration,
    "OnePlusOneEA": OnePlusOneEA,
    "SimulatedAnnealing": SimulatedAnnealing,
    "SA-Log": SimulatedAnnealing,
    "SA-Lin": SimulatedAnnealing,
    "SA-Exp": SimulatedAnnealing,
    "MuPlusLambdaEA": MuPlusLambdaEA
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
