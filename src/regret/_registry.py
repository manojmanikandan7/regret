"""Centralized registry module for algorithms, problems, and cooling schedules.

This module provides a single, accessible location for all registries used in the
regret package. It enables easy registration and lookup of:
- Optimization algorithms (e.g., RLS, EA, SimulatedAnnealing)
- Benchmark problems (e.g., OneMax, Jump, NKLandscape)
- Cooling schedules for simulated annealing (e.g., logarithmic, linear, exponential)

Registry Lookup:
    Registry keys are **case-sensitive**. Use the exact names as registered.

Example:
    >>> from regret._registry import (
    ...     ALGORITHM_REGISTRY,
    ...     PROBLEM_REGISTRY,
    ... )
    >>> # Lookup an algorithm class
    >>> rls_class = ALGORITHM_REGISTRY["RLS"]
"""

from regret.algorithms.annealing import (
    CoolingSchedule,
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

# =============================================================================
# Problem Registry
# =============================================================================

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
"""Registry mapping problem names to their implementing classes.

Keys are case-sensitive problem names. Values are subclasses of Problem.

Available problems:
    - OneMax: Count the number of 1-bits
    - LeadingOnes: Count leading 1-bits from the left
    - TwoMax: Two global optima at all-zeros and all-ones
    - Jump: OneMax with a fitness gap (jump region)
    - Trap: Deceptive function leading search away from optimum
    - BinVal: Binary value interpretation
    - Plateau: OneMax with flat regions
    - HIFF: Hierarchical If-and-only-If function
    - NKLandscape: Tunable ruggedness via epistatic interactions
    - MaxkSAT: Maximum satisfiability with k-literal clauses
    - PetersenColoringMaxSAT: Graph coloring as MaxSAT instance
"""

# =============================================================================
# Algorithm Registry
# =============================================================================

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
"""Registry mapping algorithm names to their implementing classes.

Keys are case-sensitive algorithm names. Values are subclasses of Algorithm.

Note: Multiple keys may map to the same class with different configurations.
For example, SA-Log, SA-Lin, and SA-Exp all map to SimulatedAnnealing but
are typically used with different cooling schedule configurations.

Available algorithms:
    - RLS: Randomized Local Search (flip one random bit per iteration)
    - RLSExploration: RLS with exploration phase
    - OnePlusOneEA: (1+1) Evolutionary Algorithm with mutation
    - SimulatedAnnealing: SA with configurable cooling schedule
    - SA-Log: SimulatedAnnealing (alias for logarithmic cooling)
    - SA-Lin: SimulatedAnnealing (alias for linear cooling)
    - SA-Exp: SimulatedAnnealing (alias for exponential cooling)
    - MuPlusLambdaEA: (mu+lambda) Evolutionary Algorithm
"""

# =============================================================================
# Cooling Schedule Registry
# =============================================================================

COOLING_REGISTRY: dict[str, type[CoolingSchedule]] = {
    "logarithmic": LogarithmicCooling,
    "linear": LinearCooling,
    "exponential": ExponentialCooling,
}
"""Registry mapping cooling schedule names to their implementing classes.

Keys are lowercase cooling schedule names. Values are subclasses of CoolingSchedule.

Available cooling schedules:
    - logarithmic: T(t) = T_0 / (1 + alpha * log(1 + t))
    - linear: T(t) = max(T_min, T_0 - alpha * t)
    - exponential: T(t) = T_0 * alpha^t
"""


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Registries
    "ALGORITHM_REGISTRY",
    "PROBLEM_REGISTRY",
    "COOLING_REGISTRY",
]
