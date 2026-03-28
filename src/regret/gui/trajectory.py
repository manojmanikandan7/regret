"""Trajectory buffer for accumulating algorithm search history via callbacks."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryStep:
    """A single step in the algorithm's search trajectory.

    Attributes:
        evaluations: Total function evaluations at this step.
        current_value: Fitness of the current candidate solution.
        best_value: Best fitness found so far.
        current_solution: Binary array representing the current candidate.
    """

    evaluations: int
    current_value: float
    best_value: float
    current_solution: np.ndarray


class TrajectoryBuffer:
    """Accumulates algorithm trajectory data from callbacks.

    This buffer is designed to integrate with Algorithm.callback, streaming
    search state updates as the algorithm runs.

    Attributes:
        steps: List of TrajectoryStep instances recorded during algorithm run.
    """

    def __init__(self):
        """Initialize an empty trajectory buffer."""
        self.steps: list[TrajectoryStep] = []

    def record(self, evaluations: int, current_value: float, best_value: float, solution: np.ndarray):
        """Record a step in the trajectory.

        Args:
            evaluations: Total number of function evaluations so far.
            current_value: Fitness of the current candidate.
            best_value: Best fitness found so far.
            solution: Current candidate solution (binary array).
        """
        step = TrajectoryStep(
            evaluations=evaluations,
            current_value=current_value,
            best_value=best_value,
            current_solution=solution.copy(),
        )
        self.steps.append(step)

    def __len__(self) -> int:
        """Return number of recorded steps."""
        return len(self.steps)

    def __getitem__(self, index: int) -> TrajectoryStep:
        """Access a trajectory step by index."""
        return self.steps[index]

    def get_evaluation_counts(self) -> np.ndarray:
        """Return array of evaluation counts across trajectory.

        Returns:
            Array of evaluation counts at each recorded step.
        """
        return np.array([step.evaluations for step in self.steps])

    def get_current_values(self) -> np.ndarray:
        """Return array of current fitness values across trajectory.

        Returns:
            Array of current fitness values at each recorded step.
        """
        return np.array([step.current_value for step in self.steps])

    def get_best_values(self) -> np.ndarray:
        """Return array of best fitness values across trajectory.

        Returns:
            Array of best fitness values at each recorded step.
        """
        return np.array([step.best_value for step in self.steps])

    def get_solutions(self) -> list[np.ndarray]:
        """Return list of solutions at each step.

        Returns:
            List of binary solution arrays at each recorded step.
        """
        return [step.current_solution for step in self.steps]

    def get_solution_at_step(self, step_index: int) -> np.ndarray:
        """Get the solution at a specific trajectory step.

        Args:
            step_index: Index of the step to retrieve.

        Returns:
            Binary solution array at the specified step.

        Raises:
            IndexError: If step_index is out of range.
        """
        if step_index < 0 or step_index >= len(self.steps):
            raise IndexError(f"Step index {step_index} out of range [0, {len(self.steps) - 1}]")
        return self.steps[step_index].current_solution
