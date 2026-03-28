"""Petersen graph visualization for the MaxSAT coloring problem."""

from typing import Any

import numpy as np

from regret.problems.combinatorial import PetersenColoringMaxSAT


class PetersenGraphVisualizer:
    """Visualizer for 3-coloring of the Petersen graph problem.

    Decodes binary solution vectors into vertex-color assignments and renders
    the Petersen graph structure with matplotlib.

    Attributes:
        problem: PetersenColoringMaxSAT instance defining the graph structure.
        n_vertices: Number of vertices in the Petersen graph (10).
        k_colors: Number of colors available for assignment (3).
        edges: List of (u, v) tuples representing graph edges.
    """

    # Color mapping: 0 -> red, 1 -> green, 2 -> blue
    COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    COLOR_NAMES = ["red", "green", "blue"]
    UNASSIGNED_COLOR = "#CCCCCC"  # gray for unassigned vertices

    def __init__(self, problem: PetersenColoringMaxSAT):
        """Initialize visualizer with a Petersen graph problem instance.

        Args:
            problem: PetersenColoringMaxSAT instance defining the graph structure.
        """
        self.problem = problem
        self.n_vertices = problem.N_VERTICES
        self.k_colors = problem.K_COLORS
        self.edges = problem.EDGES

    def decode_solution(self, x: np.ndarray) -> dict[int, int | None]:
        """Decode a binary bitstring into vertex-color assignments.

        The bitstring has length 30, with variables at index (v * k_colors + c)
        representing whether vertex v is assigned color c.

        Args:
            x: Binary solution array of length 30.

        Returns:
            Dictionary mapping vertex index to assigned color (0, 1, 2) or None
            if no color is assigned.
        """
        coloring: dict[int, int | None] = {}
        for v in range(self.n_vertices):
            base = v * self.k_colors
            colors_for_v = x[base : base + self.k_colors]

            # Count assigned colors
            assigned_indices = np.where(colors_for_v == 1)[0]
            if len(assigned_indices) == 0:
                coloring[v] = None
            elif len(assigned_indices) == 1:
                coloring[v] = int(assigned_indices[0])
            else:
                # Multiple colors assigned (typically shouldn't happen in optimal solution)
                coloring[v] = int(assigned_indices[0])

        return coloring

    def get_vertex_colors(self, x: np.ndarray) -> list[str]:
        """Get matplotlib colors for each vertex based on solution.

        Args:
            x: Binary solution array.

        Returns:
            List of color hex codes (length n_vertices). Unassigned vertices are gray.
        """
        coloring = self.decode_solution(x)
        colors = []
        for v in range(self.n_vertices):
            assigned_color = coloring.get(v)
            if assigned_color is None:
                colors.append(self.UNASSIGNED_COLOR)
            else:
                colors.append(self.COLORS[assigned_color])
        return colors

    def validate_coloring(self, x: np.ndarray) -> dict[str, Any]:
        """Validate a coloring against the Petersen graph constraints.

        Args:
            x: Binary solution array.

        Returns:
            Dictionary with 'valid' (bool), 'unassigned_vertices' (list),
            and 'conflicts' (list of edge tuples with same color).
        """
        coloring = self.decode_solution(x)
        conflicts = []
        unassigned = []

        for v in range(self.n_vertices):
            if coloring[v] is None:
                unassigned.append(v)

        for u, v in self.edges:
            color_u = coloring.get(u)
            color_v = coloring.get(v)
            if color_u is not None and color_v is not None and color_u == color_v:
                conflicts.append((u, v))

        return {
            "valid": len(unassigned) == 0 and len(conflicts) == 0,
            "unassigned_vertices": unassigned,
            "conflicts": conflicts,
        }

    def render(self, x: np.ndarray, ax):
        """Render the Petersen graph with vertex colors from a solution.

        Args:
            x: Binary solution array.
            ax: Matplotlib axes to draw on.
        """
        import networkx as nx

        # Build networkx graph
        G = nx.Graph()
        G.add_nodes_from(range(self.n_vertices))
        G.add_edges_from(self.edges)

        # Layout: outer pentagon on a circle, inner pentagram on smaller circle
        pos = {}
        import math

        # Outer pentagon (vertices 0-4)
        for v in range(5):
            angle = 2 * math.pi * v / 5
            pos[v] = (math.cos(angle), math.sin(angle))

        # Inner pentagram (vertices 5-9)
        for v in range(5):
            angle = 2 * math.pi * v / 5 + math.pi / 5  # offset by 36 degrees
            pos[v + 5] = (0.4 * math.cos(angle), 0.4 * math.sin(angle))

        # Get colors for vertices
        colors = self.get_vertex_colors(x)

        # Validation info
        validation = self.validate_coloring(x)
        conflicts = validation.get("conflicts", [])

        # Draw edges
        regular_edges = [e for e in self.edges if e not in conflicts and tuple(reversed(e)) not in conflicts]
        conflict_edges = conflicts

        nx.draw_networkx_edges(G, pos, edgelist=regular_edges, ax=ax, width=2, edge_color="#333333", alpha=0.6)
        nx.draw_networkx_edges(
            G, pos, edgelist=conflict_edges, ax=ax, width=3, edge_color="#FF0000", style="dashed", alpha=0.8
        )

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800, ax=ax, edgecolors="black", linewidths=2)

        # Draw labels
        labels = {i: str(i) for i in range(self.n_vertices)}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, font_weight="bold")

        ax.set_title("Petersen Graph 3-Coloring")
        ax.axis("off")
        ax.set_aspect("auto")

        # Add legend
        from matplotlib.patches import Patch

        legend_elements: list[Any] = [
            Patch(facecolor=self.COLORS[0], edgecolor="black", label="Color 0 (Red)"),
            Patch(facecolor=self.COLORS[1], edgecolor="black", label="Color 1 (Green)"),
            Patch(facecolor=self.COLORS[2], edgecolor="black", label="Color 2 (Blue)"),
            Patch(facecolor=self.UNASSIGNED_COLOR, edgecolor="black", label="Unassigned"),
        ]
        if conflicts:
            from matplotlib.lines import Line2D

            legend_elements.append(Line2D([0], [0], color="#FF0000", linewidth=2, linestyle="--", label="Conflicts"))

        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
