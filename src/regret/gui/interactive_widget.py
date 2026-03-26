"""Interactive Jupyter widget for algorithm trajectory visualization."""

from typing import Any

import numpy as np

from regret.core.metrics import instantaneous_regret
from regret.gui.petersen_viz import PetersenGraphVisualizer
from regret.gui.trajectory import TrajectoryBuffer
from regret.problems.combinatorial import PetersenColoringMaxSAT


class InteractiveTrajectoryViewer:
    """Interactive Jupyter widget for visualizing algorithm search on Petersen graph coloring.

    Displays:
    - Left panel: Petersen graph with vertex colors from current solution
    - Right panel: Regret curve with a marker at the current step

    Provides a slider to scrub through the algorithm's trajectory.
    """

    def __init__(self, trajectory: TrajectoryBuffer, problem: PetersenColoringMaxSAT):
        """Initialize the interactive viewer.

        Args:
            trajectory: TrajectoryBuffer containing the algorithm's search history.
            problem: PetersenColoringMaxSAT problem instance.
        """
        self.trajectory = trajectory
        self.problem = problem
        self.visualizer = PetersenGraphVisualizer(problem)

        # Compute regret metrics
        best_values = trajectory.get_best_values()
        self.f_star = problem.f_star
        regret_list = instantaneous_regret(
            [(i, best_values[i], best_values[i]) for i in range(len(best_values))],
            self.f_star,
            use_best=True,
        )
        # Extract just the regret values (second element) from tuples and convert to numpy array
        # regret_list is a list of (evaluation, regret) tuples
        self.regret: np.ndarray = np.array([regret for _, regret in regret_list])

        self._current_step = 0
        self._fig: Any = None
        self._ax_graph: Any = None
        self._ax_regret: Any = None
        self._slider: Any = None

    @property
    def current_step(self) -> int:
        """Current step index in the trajectory."""
        return self._current_step

    @current_step.setter
    def current_step(self, value: int):
        """Set current step and update plots."""
        if value < 0 or value >= len(self.trajectory):
            raise ValueError(f"Step {value} out of range [0, {len(self.trajectory) - 1}]")
        self._current_step = value
        self._update_plots()

    def _create_figure(self):
        """Create the matplotlib figure with two subplots."""
        import matplotlib.pyplot as plt

        self._fig, (self._ax_graph, self._ax_regret) = plt.subplots(1, 2, figsize=(14, 5))
        self._fig.suptitle("Interactive Algorithm Trajectory Viewer", fontsize=14, fontweight="bold")

    def _update_plots(self):
        """Update both the graph visualization and regret plot."""
        if self._ax_graph is None or self._ax_regret is None:
            self._create_figure()

        self._ax_graph.clear()
        self._ax_regret.clear()

        # Get solution at current step
        step = self.trajectory[self._current_step]
        solution = step.current_solution

        # Render Petersen graph
        self.visualizer.render(solution, self._ax_graph)

        # Add step info to graph
        self._ax_graph.text(
            0.5,
            -0.15,
            f"Step {self._current_step} | Evals: {step.evaluations} |"
            f"Current: {step.current_value:.1f} | Best: {step.best_value:.1f}",
            ha="center",
            transform=self._ax_graph.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Plot regret curve
        evaluations = self.trajectory.get_evaluation_counts()
        self._ax_regret.plot(evaluations, self.regret, "b-", linewidth=2, label="Instantaneous Regret")
        self._ax_regret.axvline(
            x=evaluations[self._current_step], color="red", linestyle="--", linewidth=2, label="Current"
        )
        # Use plot with marker instead of scatter for single point (handles numpy scalars better)
        self._ax_regret.plot(
            [evaluations[self._current_step]],
            [self.regret[self._current_step]],
            marker="o",
            color="red",
            markersize=10,
            linestyle="none",
            zorder=5,
        )
        self._ax_regret.set_xlabel("Evaluations", fontsize=11)
        self._ax_regret.set_ylabel("Instantaneous Regret", fontsize=11)
        self._ax_regret.set_title("Regret Trajectory")
        self._ax_regret.grid(True, alpha=0.3)
        self._ax_regret.legend(loc="best")

        # Set y-axis limits with some padding
        if self.regret.max() > 0:
            self._ax_regret.set_ylim(0, self.regret.max() * 1.1)

        self._fig.tight_layout()

    def _on_slider_change(self, change):
        """Callback for slider value changes."""
        self.current_step = change["new"]

    def show(self):
        """Display the interactive widget with slider in Jupyter notebook.

        The widget includes:
        - Slider to scrub through trajectory (0 to len(trajectory)-1)
        - Left panel: Petersen graph with current solution
        - Right panel: Regret curve with current step marker
        """
        import ipywidgets as widgets
        from IPython.display import display

        # Create initial plots
        self._create_figure()
        self._update_plots()

        # Create slider
        max_step = len(self.trajectory) - 1
        self._slider = widgets.IntSlider(
            value=0,
            min=0,
            max=max_step,
            step=1,
            description="Step:",
            continuous_update=False,
            readout=True,
            readout_format="d",
        )

        # Create output widget for the figure to enable updates
        figure_output = widgets.Output()
        with figure_output:
            display(self._fig)

        def update_figure_display(_):
            """Update the figure display when slider changes."""
            figure_output.clear_output(wait=True)
            with figure_output:
                display(self._fig)

        self._slider.observe(self._on_slider_change, names="value")
        self._slider.observe(update_figure_display, names="value")

        # Create info text
        info_output = widgets.Output()

        def update_info(_):
            info_output.clear_output()
            with info_output:
                step = self.trajectory[self._current_step]
                print("Trajectory Statistics:")
                print(f"  Total steps: {len(self.trajectory)}")
                print(f"  Current step: {self._current_step}")
                print(f"  Evaluations: {step.evaluations}")
                print(f"  Current value: {step.current_value:.2f}")
                print(f"  Best value: {step.best_value:.2f}")
                print(f"  Regret: {self.regret[self._current_step]:.4f}")

                # Validation info
                validation = self.visualizer.validate_coloring(step.current_solution)
                print("\n  Coloring validation:")
                print(f"    Valid: {validation['valid']}")
                print(f"    Unassigned vertices: {len(validation['unassigned_vertices'])}")
                print(f"    Conflicts: {len(validation['conflicts'])}")

        update_info(None)
        self._slider.observe(lambda change: update_info(change), names="value")

        # Display components
        display(self._slider)
        display(info_output)
        display(figure_output)
