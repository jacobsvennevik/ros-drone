"""Topology utilities for constructing graphs from place-cell coactivity.

Requires ``networkx``. Install via:

    python3 -m pip install networkx

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np

if TYPE_CHECKING:
    from .env import Environment

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "TopologicalGraph requires networkx. Install it with 'python3 -m pip install networkx'."
    ) from exc


def _line_intersects_circle(
    p1: np.ndarray,
    p2: np.ndarray,
    circle_center: np.ndarray,
    radius: float,
) -> bool:
    """Check if a line segment intersects (or is inside) a circle.
    
    Parameters
    ----------
    p1, p2:
        Line segment endpoints, each as a 2D array.
    circle_center:
        Center of the circle as a 2D array.
    radius:
        Radius of the circle.
    
    Returns
    -------
    bool
        True if the line segment intersects the circle or is entirely inside it.
    
    Notes
    -----
    This uses the standard line-circle intersection test. The line segment
    intersects the circle if:
    1. Either endpoint is inside the circle, OR
    2. The closest point on the line segment to the circle center is within
       the circle and lies on the segment.
    """
    # Vector from p1 to p2
    d = p2 - p1
    # Vector from p1 to circle center
    f = circle_center - p1
    
    # Check if either endpoint is inside the circle
    dist_p1 = np.linalg.norm(f)
    dist_p2 = np.linalg.norm(p2 - circle_center)
    if dist_p1 <= radius or dist_p2 <= radius:
        return True
    
    # Project circle center onto line segment
    # If d is zero, p1 == p2, already handled above
    if np.linalg.norm(d) < 1e-10:
        return dist_p1 <= radius
    
    # Parameter t along the line (0 = p1, 1 = p2)
    t = np.dot(f, d) / np.dot(d, d)
    
    # Clamp t to [0, 1] to stay on the segment
    t = max(0.0, min(1.0, t))
    
    # Closest point on segment to circle center
    closest = p1 + t * d
    
    # Distance from closest point to circle center
    dist_to_closest = np.linalg.norm(closest - circle_center)
    
    return dist_to_closest <= radius


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
        integration_window: Optional[float] = None,
        current_time: Optional[float] = None,
        integration_times: Optional[dict[tuple[int, int], float]] = None,
        environment: Optional["Environment"] = None,
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
        integration_window:
            Optional integration window (ϖ) in seconds. If provided, edges are only
            admitted if the pair has exceeded the threshold for at least this duration.
            This implements the paper's "integrator" mechanism for stable map learning.
        current_time:
            Current simulation time in seconds. Required if integration_window is provided.
        integration_times:
            Dictionary mapping (i, j) pairs to the time when they first exceeded c_min.
            Required if integration_window is provided.
        environment:
            Optional environment containing obstacles. If provided, edges that would
            cross obstacles are filtered out to preserve hole topology (Solution 1:
            obstacle-aware edge filtering).

        Examples
        --------
        >>> from hippocampus_core.topology import TopologicalGraph
        >>> import numpy as np
        >>> 
        >>> # Basic usage (no integration window)
        >>> positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        >>> graph = TopologicalGraph(positions)
        >>> coactivity = np.array([[0, 5, 3], [5, 0, 2], [3, 2, 0]])
        >>> graph.build_from_coactivity(coactivity, c_min=3.0, max_distance=2.0)
        >>> 
        >>> # With integration window
        >>> integration_times = {(0, 1): 1.5, (0, 2): 2.0}  # Times when pairs exceeded threshold
        >>> graph.build_from_coactivity(
        ...     coactivity,
        ...     c_min=3.0,
        ...     max_distance=2.0,
        ...     integration_window=2.0,  # 2 second window
        ...     current_time=4.0,
        ...     integration_times=integration_times,
        ... )
        """

        if coactivity.shape != (self.num_cells, self.num_cells):
            raise ValueError("coactivity must have shape (num_cells, num_cells)")
        if c_min < 0:
            raise ValueError("c_min must be non-negative")
        if max_distance <= 0:
            raise ValueError("max_distance must be positive")
        
        if integration_window is not None:
            if integration_window < 0:
                raise ValueError("integration_window must be non-negative")
            if current_time is None:
                raise ValueError("current_time must be provided if integration_window is set")
            if integration_times is None:
                raise ValueError("integration_times must be provided if integration_window is set")

        self.graph.remove_edges_from(list(self.graph.edges()))

        for i in range(self.num_cells):
            for j in range(i + 1, self.num_cells):
                if coactivity[i, j] < c_min:
                    continue
                
                # Apply integration window gating (paper's ϖ parameter)
                if integration_window is not None:
                    pair = (i, j)
                    if pair not in integration_times:
                        # Pair hasn't exceeded threshold yet
                        continue
                    first_exceeded_time = integration_times[pair]
                    elapsed_time = current_time - first_exceeded_time
                    if elapsed_time < integration_window:
                        # Pair hasn't exceeded threshold long enough
                        continue
                
                distance = np.linalg.norm(self.positions[i] - self.positions[j])
                if distance <= max_distance:
                    # Solution 1: Obstacle-aware edge filtering
                    # Skip edges that would cross obstacles to preserve hole topology
                    if environment is not None and environment.obstacles:
                        pos_i = self.positions[i]
                        pos_j = self.positions[j]
                        # Check if edge intersects any obstacle
                        edge_intersects_obstacle = False
                        for obstacle in environment.obstacles:
                            circle_center = np.array([obstacle.center_x, obstacle.center_y])
                            if _line_intersects_circle(pos_i, pos_j, circle_center, obstacle.radius):
                                edge_intersects_obstacle = True
                                break
                        if edge_intersects_obstacle:
                            continue  # Skip this edge - it crosses an obstacle
                    
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

    def get_edge_lengths(self) -> np.ndarray:
        """Get the Euclidean lengths of all edges in the graph.

        Returns
        -------
        np.ndarray
            Array of edge lengths in the same units as positions.
        """

        lengths = []
        for i, j in self.graph.edges():
            distance = np.linalg.norm(self.positions[i] - self.positions[j])
            lengths.append(distance)
        return np.array(lengths, dtype=float)

    def get_node_degrees(self) -> np.ndarray:
        """Get the degree (number of connections) for each node.

        Returns
        -------
        np.ndarray
            Array of node degrees, shape (num_nodes,).
        """

        degrees = np.array([self.graph.degree(node) for node in range(self.num_cells)], dtype=int)
        return degrees

    def get_degree_statistics(self) -> dict[str, float]:
        """Get statistics about node degrees.

        Returns
        -------
        dict[str, float]
            Dictionary with keys: 'mean', 'median', 'min', 'max', 'std'
        """

        degrees = self.get_node_degrees()
        return {
            "mean": float(np.mean(degrees)),
            "median": float(np.median(degrees)),
            "min": int(np.min(degrees)),
            "max": int(np.max(degrees)),
            "std": float(np.std(degrees)),
        }

    def get_edge_length_statistics(self) -> dict[str, float]:
        """Get statistics about edge lengths.

        Returns
        -------
        dict[str, float]
            Dictionary with keys: 'mean', 'median', 'min', 'max', 'std'
        """

        lengths = self.get_edge_lengths()
        if len(lengths) == 0:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        return {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "min": float(np.min(lengths)),
            "max": float(np.max(lengths)),
            "std": float(np.std(lengths)),
        }

    def get_maximal_cliques(self) -> list[list[int]]:
        """Extract all maximal cliques from the graph.

        Returns
        -------
        list[list[int]]
            List of maximal cliques, where each clique is a list of node indices.
            A clique is a set of nodes where every pair of nodes is connected by an edge.
            A maximal clique is a clique that cannot be extended by adding another node.

        Notes
        -----
        This method is used to build the clique complex for persistent homology
        computation. In the clique complex, each k-clique (k nodes) corresponds to
        a (k-1)-simplex.
        """

        return list(nx.find_cliques(self.graph))

    def compute_betti_numbers(
        self, max_dim: int = 2, backend: str = "auto"
    ) -> dict[int, int]:
        """Compute Betti numbers from the clique complex of this graph.

        Parameters
        ----------
        max_dim:
            Maximum dimension for which to compute Betti numbers (default: 2).
            b_0, b_1, ..., b_max_dim will be computed.
        backend:
            Backend to use: "ripser", "gudhi", or "auto" (default: "auto").
            If "auto", uses ripser if available, otherwise gudhi.

        Returns
        -------
        dict[int, int]
            Dictionary mapping dimension to Betti number: {0: b_0, 1: b_1, ...}
            - b_0: number of connected components
            - b_1: number of 1D holes (loops)
            - b_2: number of 2D holes (voids)
            - etc.

        Raises
        ------
        ImportError
            If neither ripser nor gudhi is available and backend is "auto",
            or if the specified backend is not available.
        ValueError
            If max_dim is negative.

        Notes
        -----
        This method builds a clique complex from the graph's maximal cliques
        and computes its Betti numbers using persistent homology. This allows
        verification that the learned graph topology matches the physical
        environment (e.g., b_0=1 for connected space, b_1=number of holes).

        The clique complex approach matches the method used in Hoffman et al.
        (2016) for topological mapping in bat hippocampus.

        Important: When the graph has no edges (all nodes isolated), b_0 should
        equal the number of nodes. However, persistent homology computation on
        an empty clique complex may return b_0=1. In this case, prefer using
        `num_components()` which correctly counts isolated nodes.

        Examples
        --------
        >>> from hippocampus_core.topology import TopologicalGraph
        >>> import numpy as np
        >>> 
        >>> # Create a graph from place cell positions
        >>> positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        >>> graph = TopologicalGraph(positions)
        >>> 
        >>> # Build graph from coactivity matrix
        >>> coactivity = np.zeros((4, 4))
        >>> coactivity[0, 1] = coactivity[1, 0] = 5.0
        >>> coactivity[1, 2] = coactivity[2, 1] = 5.0
        >>> coactivity[2, 3] = coactivity[3, 2] = 5.0
        >>> coactivity[3, 0] = coactivity[0, 3] = 5.0
        >>> graph.build_from_coactivity(coactivity, c_min=3.0, max_distance=2.0)
        >>> 
        >>> # Compute Betti numbers (requires ripser or gudhi)
        >>> betti = graph.compute_betti_numbers(max_dim=2)
        >>> print(f"b_0 (components): {betti[0]}")
        >>> print(f"b_1 (holes): {betti[1]}")
        >>> 
        >>> # Verify b_0 equals number of connected components
        >>> assert betti[0] == graph.num_components()
        """
        if max_dim < 0:
            raise ValueError(f"max_dim must be non-negative, got {max_dim}")

        try:
            from .persistent_homology import compute_betti_numbers_from_cliques
        except ImportError as exc:
            raise ImportError(
                "Persistent homology computation requires ripser or gudhi. "
                "Install with: python3 -m pip install ripser"
            ) from exc

        cliques = self.get_maximal_cliques()
        return compute_betti_numbers_from_cliques(cliques, max_dim=max_dim, backend=backend)
