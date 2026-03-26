"""Interactive visualization module for combinatorial optimization algorithms.

This module provides tools for visualizing algorithm search trajectories on concrete
combinatorial problems like Petersen graph coloring. It is designed to be independent
of the experiment suite and can evolve separately.
"""

from regret.gui.interactive_widget import InteractiveTrajectoryViewer
from regret.gui.petersen_viz import PetersenGraphVisualizer
from regret.gui.trajectory import TrajectoryBuffer

__all__ = ["PetersenGraphVisualizer", "TrajectoryBuffer", "InteractiveTrajectoryViewer"]
