"""Topology utilities for constructing graphs from place-cell coactivity.

Requires ``networkx``. Install via:

    python3 -m pip install networkx

"""
from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "TopologicalGraph requires networkx. Install it with 'python3 -m pip install networkx'."
    ) from exc


class TopologicalGraph:
    """Graph derived from place-cell centers and their coactivity matrix."""

    def __init__(self, positions: np.ndarray) -> None:
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (num_cells, 2)")

        self.positions = positions
        self.num_cells = positions.shape[0]
        self.graph = nx.Graph()
        for idx, (x, y) in enumerate(self.positions):
            self.graph.add_node(idx, position=(float(x), float(y)))

    def build_from_coactivity(
        self,
        coactivity: np.ndarray,
        c_min: float,
        max_distance: float,
    ) -> None:
        """Populate edges using coactivity counts and spatial proximity.

        Parameters
        ----------
        coactivity:
            Square coactivity count matrix ``C`` of shape (num_cells, num_cells).
        c_min:
            Minimum coactivity count needed to draw an edge between two cells.
        max_distance:
            Maximum Euclidean distance between cell centers for an edge to be eligible.
        """

        if coactivity.shape != (self.num_cells, self.num_cells):
            raise ValueError("coactivity must have shape (num_cells, num_cells)")
        if c_min < 0:
            raise ValueError("c_min must be non-negative")
        if max_distance <= 0:
            raise ValueError("max_distance must be positive")

        self.graph.remove_edges_from(list(self.graph.edges()))

        for i in range(self.num_cells):
            for j in range(i + 1, self.num_cells):
                if coactivity[i, j] < c_min:
                    continue
                distance = np.linalg.norm(self.positions[i] - self.positions[j])
                if distance <= max_distance:
                    self.graph.add_edge(i, j, weight=float(coactivity[i, j]), distance=float(distance))

    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""

        return self.graph.number_of_nodes()

    def num_edges(self) -> int:
        """Return the number of edges in the graph."""

        return self.graph.number_of_edges()

    def num_components(self) -> int:
        """Return the number of connected components."""

        return nx.number_connected_components(self.graph)

    def cycle_basis(self) -> Iterable[list[int]]:
        """Return a simple cycle basis of the underlying undirected graph."""

        return nx.cycle_basis(self.graph)
