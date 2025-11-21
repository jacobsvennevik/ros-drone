"""Graph Navigation Service for path planning on topological graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

from .data_structures import (
    GraphSnapshot,
    MissionGoal,
    GoalType,
    NodeGoal,
    PointGoal,
)


@dataclass
class NavigationPath:
    """A path through the graph."""

    nodes: List[int]  # Sequence of node IDs
    total_length: float  # Sum of edge lengths
    is_complete: bool  # True if path reaches goal


@dataclass
class WaypointTarget:
    """Target waypoint for navigation."""

    node_id: int
    position: Tuple[float, float]  # (x, y) or (x, y, z) for 3D
    distance: float  # Distance from current pose
    bearing: float  # Bearing from current pose (radians)


class GraphNavigationService:
    """Service for navigating the topological graph."""

    def __init__(
        self,
        graph_snapshot: Optional[GraphSnapshot] = None,
        algorithm: str = "dijkstra",  # "dijkstra", "astar", "greedy"
    ):
        """Initialize navigation service.

        Parameters
        ----------
        graph_snapshot:
            Optional initial graph snapshot.
        algorithm:
            Path planning algorithm to use.
        """
        if nx is None:
            raise ImportError("NetworkX required for GraphNavigationService")
        
        self.graph_snapshot = graph_snapshot
        self.algorithm = algorithm
        self._cached_paths: dict[Tuple[int, int], NavigationPath] = {}
        self._graph_version = 0

    def update_graph(self, new_snapshot: GraphSnapshot) -> None:
        """Update graph snapshot and invalidate caches.

        Parameters
        ----------
        new_snapshot:
            New graph snapshot.
        """
        if new_snapshot.meta.epoch_id != self.graph_snapshot.meta.epoch_id if self.graph_snapshot else -1:
            self.graph_snapshot = new_snapshot
            self._graph_version += 1
            self._cached_paths.clear()  # Invalidate cache

    def find_path(
        self,
        start_node: int,
        goal_node: int,
    ) -> Optional[NavigationPath]:
        """Find path from start to goal node.

        Parameters
        ----------
        start_node:
            Starting node ID.
        goal_node:
            Goal node ID.

        Returns
        -------
        Optional[NavigationPath]
            Path if found, None otherwise.
        """
        if self.graph_snapshot is None:
            return None

        if start_node == goal_node:
            return NavigationPath(nodes=[start_node], total_length=0.0, is_complete=True)

        # Check cache
        cache_key = (start_node, goal_node)
        if cache_key in self._cached_paths:
            cached = self._cached_paths[cache_key]
            if self._is_path_valid(cached):
                return cached

        # Build NetworkX graph
        graph = self._build_networkx_graph()
        if graph is None:
            return None

        # Check if nodes exist
        if start_node not in graph.nodes or goal_node not in graph.nodes:
            return None

        # Find path using selected algorithm
        if self.algorithm == "dijkstra":
            path_nodes = self._dijkstra_path(graph, start_node, goal_node)
        elif self.algorithm == "astar":
            path_nodes = self._astar_path(graph, start_node, goal_node)
        elif self.algorithm == "greedy":
            path_nodes = self._greedy_path(graph, start_node, goal_node)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        if path_nodes is None:
            return None

        # Compute total length
        total_length = self._compute_path_length(path_nodes)

        path = NavigationPath(
            nodes=path_nodes,
            total_length=total_length,
            is_complete=len(path_nodes) > 0 and path_nodes[-1] == goal_node,
        )

        # Cache path
        self._cached_paths[cache_key] = path

        return path

    def select_next_waypoint(
        self,
        current_pose: Tuple[float, float, float],  # (x, y, yaw)
        goal: MissionGoal,
    ) -> Optional[WaypointTarget]:
        """Select next waypoint given current pose and goal.

        Parameters
        ----------
        current_pose:
            Current robot pose (x, y, yaw).
        goal:
            Mission goal.

        Returns
        -------
        Optional[WaypointTarget]
            Next waypoint target or None if goal unreachable.
        """
        if self.graph_snapshot is None or not self.graph_snapshot.V:
            return None

        # Find current node (nearest to current pose)
        current_node = self._find_nearest_node(current_pose[:2])

        # Resolve goal to node
        goal_node = self._resolve_goal_node(goal)
        if goal_node is None:
            return None

        # Find path
        path = self.find_path(current_node, goal_node)
        if path is None or not path.is_complete:
            return None

        # Select next waypoint (first node in path after current)
        if len(path.nodes) < 2:
            # Already at goal
            return None

        next_node_id = path.nodes[1]
        next_node = self._get_node_data(next_node_id)
        if next_node is None:
            return None

        # Compute distance and bearing
        dx = next_node.position[0] - current_pose[0]
        dy = next_node.position[1] - current_pose[1]
        distance = np.sqrt(dx**2 + dy**2)
        bearing = np.arctan2(dy, dx) - current_pose[2]
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))  # Wrap to [-π, π]

        return WaypointTarget(
            node_id=next_node_id,
            position=next_node.position,
            distance=distance,
            bearing=bearing,
        )

    def _build_networkx_graph(self) -> Optional[nx.Graph]:
        """Build NetworkX graph from snapshot.

        Returns
        -------
        Optional[nx.Graph]
            NetworkX graph or None if snapshot invalid.
        """
        if self.graph_snapshot is None:
            return None

        graph = nx.Graph()

        # Add nodes
        for node in self.graph_snapshot.V:
            graph.add_node(node.node_id, position=node.position)

        # Add edges
        for edge in self.graph_snapshot.E:
            if edge.traversable:
                graph.add_edge(edge.u, edge.v, weight=edge.length, length=edge.length)

        return graph

    def _dijkstra_path(
        self,
        graph: nx.Graph,
        start: int,
        goal: int,
    ) -> Optional[List[int]]:
        """Find shortest path using Dijkstra's algorithm."""
        try:
            path = nx.shortest_path(
                graph,
                source=start,
                target=goal,
                weight="length",
            )
            return path
        except nx.NetworkXNoPath:
            return None

    def _astar_path(
        self,
        graph: nx.Graph,
        start: int,
        goal: int,
    ) -> Optional[List[int]]:
        """Find shortest path using A* algorithm."""
        # Get node positions for heuristic
        start_pos = np.array(graph.nodes[start]["position"][:2])
        goal_pos = np.array(graph.nodes[goal]["position"][:2])

        def heuristic(u: int, v: int) -> float:
            """Euclidean distance heuristic."""
            u_pos = np.array(graph.nodes[u]["position"][:2])
            v_pos = np.array(graph.nodes[v]["position"][:2])
            return float(np.linalg.norm(u_pos - v_pos))

        try:
            path = nx.astar_path(
                graph,
                source=start,
                target=goal,
                heuristic=heuristic,
                weight="length",
            )
            return path
        except nx.NetworkXNoPath:
            return None

    def _greedy_path(
        self,
        graph: nx.Graph,
        start: int,
        goal: int,
    ) -> Optional[List[int]]:
        """Greedy path: always move to neighbor closest to goal."""
        goal_pos = np.array(graph.nodes[goal]["position"][:2])
        path = [start]
        current = start
        visited = {start}
        max_iterations = graph.number_of_nodes() * 2  # Prevent infinite loops

        iteration = 0
        while current != goal and iteration < max_iterations:
            iteration += 1
            neighbors = list(graph.neighbors(current))
            unvisited = [n for n in neighbors if n not in visited]

            if not unvisited:
                # Dead end, backtrack
                if len(path) < 2:
                    return None
                path.pop()
                current = path[-1]
                continue

            # Select neighbor closest to goal
            best_neighbor = min(
                unvisited,
                key=lambda n: np.linalg.norm(
                    np.array(graph.nodes[n]["position"][:2]) - goal_pos
                ),
            )

            path.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor

        if iteration >= max_iterations:
            return None

        return path if current == goal else None

    def _compute_path_length(self, path_nodes: List[int]) -> float:
        """Compute total length of path.

        Parameters
        ----------
        path_nodes:
            List of node IDs in path.

        Returns
        -------
        float
            Total path length.
        """
        if len(path_nodes) < 2:
            return 0.0

        total = 0.0
        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]
            u_node = self._get_node_data(u)
            v_node = self._get_node_data(v)
            if u_node and v_node:
                dist = np.linalg.norm(
                    np.array(u_node.position[:2]) - np.array(v_node.position[:2])
                )
                total += dist

        return total

    def _is_path_valid(self, path: NavigationPath) -> bool:
        """Check if cached path is still valid.

        Parameters
        ----------
        path:
            Cached path.

        Returns
        -------
        bool
            True if path is still valid.
        """
        if self.graph_snapshot is None:
            return False

        # Check if all nodes in path still exist
        node_ids = {n.node_id for n in self.graph_snapshot.V}
        return all(node_id in node_ids for node_id in path.nodes)

    def _find_nearest_node(self, position: Tuple[float, float]) -> int:
        """Find graph node nearest to given position.

        Parameters
        ----------
        position:
            Position (x, y).

        Returns
        -------
        int
            Nearest node ID.
        """
        if not self.graph_snapshot or not self.graph_snapshot.V:
            raise ValueError("No nodes in graph")

        min_dist = float("inf")
        nearest = None

        pos_array = np.array(position)
        for node in self.graph_snapshot.V:
            node_pos = np.array(node.position[:2])
            dist = np.linalg.norm(node_pos - pos_array)
            if dist < min_dist:
                min_dist = dist
                nearest = node.node_id

        if nearest is None:
            raise ValueError("Could not find nearest node")

        return nearest

    def _resolve_goal_node(self, goal: MissionGoal) -> Optional[int]:
        """Resolve mission goal to graph node ID.

        Parameters
        ----------
        goal:
            Mission goal.

        Returns
        -------
        Optional[int]
            Node ID or None if cannot resolve.
        """
        if self.graph_snapshot is None or not self.graph_snapshot.V:
            return None

        if goal.type == GoalType.NODE:
            node_goal = goal.value
            # Check if node exists
            if any(n.node_id == node_goal.node_id for n in self.graph_snapshot.V):
                return node_goal.node_id
            return None

        elif goal.type == GoalType.POINT:
            point_goal = goal.value
            # Find nearest node to point
            return self._find_nearest_node(point_goal.position[:2])

        else:
            # Other goal types not yet supported
            return None

    def _get_node_data(self, node_id: int) -> Optional:
        """Get node data by ID.

        Parameters
        ----------
        node_id:
            Node ID.

        Returns
        -------
        Optional[NodeData]
            Node data or None if not found.
        """
        if self.graph_snapshot is None:
            return None

        return next((n for n in self.graph_snapshot.V if n.node_id == node_id), None)

    def is_reachable(
        self,
        start_node: int,
        goal_node: int,
    ) -> bool:
        """Check if goal is reachable from start.

        Parameters
        ----------
        start_node:
            Start node ID.
        goal_node:
            Goal node ID.

        Returns
        -------
        bool
            True if reachable.
        """
        path = self.find_path(start_node, goal_node)
        return path is not None and path.is_complete

