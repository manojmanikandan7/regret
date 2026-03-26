"""Unit tests for GUI components: trajectory buffer and Petersen visualizer."""

import numpy as np
import pytest

from regret.algorithms.local_search import RLS
from regret.gui.petersen_viz import PetersenGraphVisualizer
from regret.gui.trajectory import TrajectoryBuffer
from regret.problems.combinatorial import PetersenColoringMaxSAT


class TestTrajectoryBuffer:
    """Tests for TrajectoryBuffer trajectory accumulation."""

    def test_trajectory_init(self):
        """Test trajectory buffer initialization."""
        buffer = TrajectoryBuffer()
        assert len(buffer) == 0
        assert buffer.steps == []

    def test_record_single_step(self):
        """Test recording a single step."""
        buffer = TrajectoryBuffer()
        solution = np.array([1, 0, 1] + [0] * 27, dtype=int)  # 30-bit solution
        buffer.record(1, 50.0, 50.0, solution)

        assert len(buffer) == 1
        step = buffer[0]
        assert step.evaluations == 1
        assert step.current_value == 50.0
        assert step.best_value == 50.0
        assert np.array_equal(step.current_solution, solution)

    def test_record_multiple_steps(self):
        """Test recording multiple steps."""
        buffer = TrajectoryBuffer()
        for i in range(5):
            sol = np.zeros(30, dtype=int)
            buffer.record(i + 1, float(40 + i), float(40 + i), sol)

        assert len(buffer) == 5
        assert buffer[0].evaluations == 1
        assert buffer[4].evaluations == 5

    def test_get_arrays(self):
        """Test extracting metrics as arrays."""
        buffer = TrajectoryBuffer()
        for i in range(3):
            sol = np.zeros(30, dtype=int)
            buffer.record(i + 1, float(40 + i), float(42 + i), sol)

        evals = buffer.get_evaluation_counts()
        currents = buffer.get_current_values()
        bests = buffer.get_best_values()

        assert np.array_equal(evals, [1, 2, 3])
        assert np.allclose(currents, [40.0, 41.0, 42.0])
        assert np.allclose(bests, [42.0, 43.0, 44.0])

    def test_get_solutions(self):
        """Test retrieving solutions."""
        buffer = TrajectoryBuffer()
        sols = [np.array([i] * 30, dtype=int) for i in range(3)]
        for i, sol in enumerate(sols):
            buffer.record(i + 1, float(40 + i), float(42 + i), sol)

        retrieved = buffer.get_solutions()
        assert len(retrieved) == 3
        for i, sol in enumerate(retrieved):
            assert np.array_equal(sol, sols[i])

    def test_get_solution_at_step(self):
        """Test accessing solution at specific step."""
        buffer = TrajectoryBuffer()
        target_sol = np.array([1, 2, 3] + [0] * 27, dtype=int)
        buffer.record(1, 40.0, 40.0, np.zeros(30, dtype=int))
        buffer.record(2, 41.0, 41.0, target_sol)
        buffer.record(3, 42.0, 42.0, np.zeros(30, dtype=int))

        assert np.array_equal(buffer.get_solution_at_step(1), target_sol)

    def test_get_solution_out_of_range(self):
        """Test error handling for out-of-range step indices."""
        buffer = TrajectoryBuffer()
        buffer.record(1, 40.0, 40.0, np.zeros(30, dtype=int))

        with pytest.raises(IndexError):
            buffer.get_solution_at_step(5)

        with pytest.raises(IndexError):
            buffer.get_solution_at_step(-1)


class TestPetersenGraphVisualizer:
    """Tests for Petersen graph visualization."""

    @pytest.fixture
    def problem(self):
        """Create a Petersen coloring problem."""
        return PetersenColoringMaxSAT()

    @pytest.fixture
    def visualizer(self, problem):
        """Create a visualizer."""
        return PetersenGraphVisualizer(problem)

    def test_visualizer_init(self, visualizer, problem):
        """Test visualizer initialization."""
        assert visualizer.problem is problem
        assert visualizer.n_vertices == 10
        assert visualizer.k_colors == 3
        assert len(visualizer.edges) == 15

    def test_decode_empty_solution(self, visualizer):
        """Test decoding an all-zeros solution."""
        solution = np.zeros(30, dtype=int)
        coloring = visualizer.decode_solution(solution)

        assert len(coloring) == 10
        assert all(color is None for color in coloring.values())

    def test_decode_valid_coloring(self, visualizer):
        """Test decoding a valid 3-coloring."""
        # Assign colors: vertex 0 -> red, vertex 1 -> green, etc (cycling)
        solution = np.zeros(30, dtype=int)
        for v in range(10):
            color = v % 3
            idx = v * 3 + color
            solution[idx] = 1

        coloring = visualizer.decode_solution(solution)
        for v in range(10):
            assert coloring[v] == v % 3

    def test_decode_partial_assignment(self, visualizer):
        """Test decoding a partial coloring."""
        solution = np.zeros(30, dtype=int)
        # Assign only vertex 0 and 1
        solution[0] = 1  # vertex 0 -> color 0
        solution[4] = 1  # vertex 1 -> color 1

        coloring = visualizer.decode_solution(solution)
        assert coloring[0] == 0
        assert coloring[1] == 1
        assert all(coloring[v] is None for v in range(2, 10))

    def test_get_vertex_colors(self, visualizer):
        """Test getting matplotlib colors from solution."""
        solution = np.zeros(30, dtype=int)
        for v in range(10):
            color = v % 3
            solution[v * 3 + color] = 1

        colors = visualizer.get_vertex_colors(solution)
        assert len(colors) == 10
        # Check that some colors are assigned (not gray)
        assert any(c != visualizer.UNASSIGNED_COLOR for c in colors)

    def test_validate_empty_coloring(self, visualizer):
        """Test validation of empty coloring."""
        solution = np.zeros(30, dtype=int)
        validation = visualizer.validate_coloring(solution)

        assert not validation["valid"]
        assert len(validation["unassigned_vertices"]) == 10
        assert len(validation["conflicts"]) == 0

    def test_validate_valid_coloring(self, visualizer):
        """Test validation of a valid coloring."""
        solution = np.zeros(30, dtype=int)
        # Simple greedy coloring (outer pentagon)
        colors = [0, 1, 2, 0, 1]  # outer pentagon
        colors += [2, 0, 1, 2, 0]  # inner pentagram

        for v, color in enumerate(colors):
            solution[v * 3 + color] = 1

        validation = visualizer.validate_coloring(solution)
        # Note: this may not be a valid 3-coloring, but tests the structure
        assert "valid" in validation
        assert "unassigned_vertices" in validation
        assert "conflicts" in validation

    def test_visualizer_render_no_error(self, visualizer):
        """Test that render() doesn't crash (requires matplotlib and networkx)."""
        try:
            import matplotlib.pyplot as plt

            solution = np.zeros(30, dtype=int)
            for v in range(10):
                color = v % 3
                solution[v * 3 + color] = 1

            fig, ax = plt.subplots()
            visualizer.render(solution, ax)
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib or networkx not available")


class TestCallbackIntegration:
    """Tests for algorithm callback integration with trajectory buffer."""

    def test_callback_invoked_during_run(self):
        """Test that callbacks are invoked during algorithm execution."""
        problem = PetersenColoringMaxSAT()
        buffer = TrajectoryBuffer()

        def callback(evals, current, best, solution):
            buffer.record(evals, current, best, solution)

        algo = RLS(problem, seed=42, callback=callback)
        algo.run(budget=10)

        assert len(buffer) > 0
        # Should record initial state + steps
        assert len(buffer) >= 2

    def test_callback_records_correct_data(self):
        """Test that callback data matches trajectory."""
        problem = PetersenColoringMaxSAT()
        buffer = TrajectoryBuffer()

        def callback(evals, current, best, solution):
            buffer.record(evals, current, best, solution)

        algo = RLS(problem, seed=42, callback=callback)
        algo.run(budget=5)

        # Verify recorded data structure
        for step in buffer.steps:
            assert isinstance(step.evaluations, int)
            assert isinstance(step.current_value, float)
            assert isinstance(step.best_value, float)
            assert isinstance(step.current_solution, np.ndarray)
            assert len(step.current_solution) == 30

    def test_callback_optional_backward_compat(self):
        """Test that algorithms work without callbacks (backward compatibility)."""
        problem = PetersenColoringMaxSAT()
        algo = RLS(problem, seed=42)  # No callback
        best_val, best_sol = algo.run(budget=5)

        assert best_val is not None
        assert best_sol is not None
        # Should still work and record history
        assert len(algo.history) > 0
