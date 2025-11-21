"""Spatial Feature Service for building policy features."""
from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np

from .data_structures import (
    FeatureVector,
    RobotState,
    Mission,
    MissionGoal,
    GoalType,
    GraphSnapshot,
    LocalContext,
)
from .topology_service import TopologyService


def compute_goal_ego(
    robot_pose: Tuple[float, float, float],  # (x, y, yaw) or (x, y, z, yaw, pitch) for 3D
    goal_position: Tuple[float, ...],  # (x, y) or (x, y, z) for 3D
    max_range: float = 10.0,
    is_3d: bool = False,
) -> List[float]:
    """Compute egocentric goal features.

    Parameters
    ----------
    robot_pose:
        Current robot pose (x, y, yaw) for 2D or (x, y, z, yaw, pitch) for 3D.
    goal_position:
        Goal position (x, y) for 2D or (x, y, z) for 3D.
    max_range:
        Maximum range for distance normalization.
    is_3d:
        Whether to compute 3D features.

    Returns
    -------
    List[float]
        2D: [distance_norm, cos(θ_g), sin(θ_g)]
        3D: [distance_norm, cos(θ_g), sin(θ_g), sin(φ_g), cos(φ_g)]
    """
    # Extract 2D position
    robot_pos = np.array([robot_pose[0], robot_pose[1]])
    goal_pos = np.array([goal_position[0], goal_position[1]])

    # Vector from robot to goal
    vec_to_goal = goal_pos - robot_pos
    distance_2d = np.linalg.norm(vec_to_goal)

    # For 3D, compute full 3D distance
    if is_3d and len(robot_pose) >= 3 and len(goal_position) >= 3:
        robot_z = robot_pose[2] if len(robot_pose) > 2 else 0.0
        goal_z = goal_position[2] if len(goal_position) > 2 else 0.0
        dz = goal_z - robot_z
        distance = np.sqrt(distance_2d**2 + dz**2)
    else:
        distance = distance_2d
        dz = 0.0

    # Normalize distance
    distance_norm = np.clip(distance / max_range, 0.0, 1.0)

    # Bearing relative to robot heading
    goal_bearing = np.arctan2(vec_to_goal[1], vec_to_goal[0])
    robot_yaw = robot_pose[2] if len(robot_pose) > 2 else 0.0
    relative_bearing = goal_bearing - robot_yaw
    relative_bearing = np.arctan2(np.sin(relative_bearing), np.cos(relative_bearing))  # Wrap to [-π, π]

    features = [
        float(distance_norm),
        float(np.cos(relative_bearing)),
        float(np.sin(relative_bearing)),
    ]

    # Add elevation for 3D
    if is_3d and len(robot_pose) >= 3 and len(goal_position) >= 3:
        elevation = np.arctan2(dz, distance_2d) if distance_2d > 0 else 0.0
        features.extend([
            float(np.sin(elevation)),
            float(np.cos(elevation)),
        ])

    return features


def compute_neighbor_features(
    robot_pose: Tuple[float, float, float],
    graph_snapshot: GraphSnapshot,
    k: int = 8,
    current_path: Optional[List[int]] = None,
    max_range: float = 10.0,
) -> List[List[float]]:
    """Compute features for k nearest graph nodes.

    Parameters
    ----------
    robot_pose:
        Current robot pose (x, y, yaw).
    graph_snapshot:
        Current graph snapshot.
    k:
        Number of nearest neighbors to consider.
    current_path:
        Optional current path (for on_path flag).
    max_range:
        Maximum range for distance normalization.

    Returns
    -------
    List[List[float]]
        k × [cos(θ_j), sin(θ_j), d_j_norm, on_path]
    """
    robot_pos = np.array([robot_pose[0], robot_pose[1]])

    if not graph_snapshot.V:
        # No nodes: return dummy features
        return [[0.0, 0.0, 1.0, 0.0]] * k

    # Compute distances to all nodes
    node_distances = []
    for node in graph_snapshot.V:
        node_pos = np.array(node.position[:2])
        dist = np.linalg.norm(node_pos - robot_pos)
        node_distances.append((node.node_id, dist, node_pos))

    # Sort by distance and take k nearest
    node_distances.sort(key=lambda x: x[1])
    k_nearest = node_distances[:k]

    # Pad if fewer than k nodes
    while len(k_nearest) < k:
        k_nearest.append((None, max_range, np.array([0.0, 0.0])))  # Dummy node

    # Compute features for each neighbor
    neighbor_features = []
    path_set = set(current_path) if current_path else set()

    for node_id, dist, node_pos in k_nearest:
        if node_id is None:
            # Dummy node: zero features
            neighbor_features.append([0.0, 0.0, 1.0, 0.0])
            continue

        # Distance (normalized)
        dist_norm = np.clip(dist / max_range, 0.0, 1.0)

        # Bearing relative to robot heading
        vec_to_node = node_pos - robot_pos
        node_bearing = np.arctan2(vec_to_node[1], vec_to_node[0])
        relative_bearing = node_bearing - robot_pose[2]
        relative_bearing = np.arctan2(np.sin(relative_bearing), np.cos(relative_bearing))

        # On path flag
        on_path = 1.0 if node_id in path_set else 0.0

        neighbor_features.append([
            float(np.cos(relative_bearing)),
            float(np.sin(relative_bearing)),
            float(dist_norm),
            float(on_path),
        ])

    return neighbor_features


def compute_topo_context(
    graph_snapshot: GraphSnapshot,
    current_node: Optional[int] = None,
    current_path: Optional[List[int]] = None,
) -> List[float]:
    """Compute topological context features.

    Parameters
    ----------
    graph_snapshot:
        Current graph snapshot.
    current_node:
        Current graph node ID (if known).
    current_path:
        Current path through graph (if known).

    Returns
    -------
    List[float]
        [deg_norm, clustering, path_progress]
    """
    # Node degree (normalized)
    if current_node is not None and graph_snapshot.V:
        node_data = next((n for n in graph_snapshot.V if n.node_id == current_node), None)
        if node_data:
            degree = node_data.degree
            max_degree = max(n.degree for n in graph_snapshot.V) if graph_snapshot.V else 1
            deg_norm = degree / max_degree if max_degree > 0 else 0.0
        else:
            deg_norm = 0.0
    else:
        deg_norm = 0.0

    # Clustering coefficient (simplified)
    clustering = 0.0
    if current_node is not None and graph_snapshot.V:
        # Find neighbors
        neighbors = []
        for e in graph_snapshot.E:
            if e.u == current_node:
                neighbors.append(e.v)
            elif e.v == current_node:
                neighbors.append(e.u)
        neighbors = list(set(neighbors))

        if len(neighbors) > 1:
            # Count edges between neighbors
            neighbor_edges = sum(
                1 for e in graph_snapshot.E
                if (e.u in neighbors and e.v in neighbors)
            )
            max_possible = len(neighbors) * (len(neighbors) - 1) / 2
            clustering = neighbor_edges / max_possible if max_possible > 0 else 0.0

    # Path progress
    path_progress = 0.0
    if current_path and current_node is not None:
        try:
            idx = current_path.index(current_node)
            path_progress = idx / len(current_path) if len(current_path) > 0 else 0.0
        except ValueError:
            path_progress = 0.0

    return [float(deg_norm), float(clustering), float(path_progress)]


def compute_safety_features(
    robot_pose: Tuple[float, float, float],
    sensor_data: Optional[any] = None,
    num_bands: int = 4,
    is_3d: bool = False,
) -> List[float]:
    """Compute obstacle avoidance features.

    Parameters
    ----------
    robot_pose:
        Current robot pose.
    sensor_data:
        Optional sensor data (depth/LiDAR). For Milestone A, assume safe.
    num_bands:
        Number of directional bands (4 for 2D, 5 for 3D).
    is_3d:
        Whether to compute 3D safety features.

    Returns
    -------
    List[float]
        2D: [front, left, right, back]_norm
        3D: [front, left, right, up, down]_norm
    """
    # For Milestone A: assume safe (no obstacles detected)
    # In future: integrate with actual sensor data
    if is_3d and num_bands == 5:
        return [1.0] * 5  # front, left, right, up, down
    else:
        return [1.0] * num_bands  # front, left, right, back


def compute_dynamics_features(
    previous_actions: Optional[List[float]],
    max_linear: float = 0.3,
    max_angular: float = 1.0,
) -> Optional[List[float]]:
    """Compute dynamics features from previous actions.

    Parameters
    ----------
    previous_actions:
        Previous action [v, ω] or None.
    max_linear:
        Maximum linear velocity for normalization.
    max_angular:
        Maximum angular velocity for normalization.

    Returns
    -------
    Optional[List[float]]
        [v_norm, omega_norm] or None if no previous actions.
    """
    if previous_actions is None or len(previous_actions) < 2:
        return None

    v_norm = previous_actions[0] / max_linear if max_linear > 0 else 0.0
    omega_norm = previous_actions[1] / max_angular if max_angular > 0 else 0.0

    return [float(v_norm), float(omega_norm)]


class SpatialFeatureService:
    """Service that builds features from graph, robot state, and mission."""

    def __init__(
        self,
        topology_service: TopologyService,
        k_neighbors: int = 8,
        max_range: float = 10.0,
        is_3d: bool = False,
    ):
        """Initialize feature service.

        Parameters
        ----------
        topology_service:
            TopologyService instance.
        k_neighbors:
            Number of nearest neighbors to consider (8 for 2D, 12 for 3D).
        max_range:
            Maximum range for distance normalization.
        is_3d:
            Whether to compute 3D features.
        """
        self.topology_service = topology_service
        self.k_neighbors = k_neighbors
        self.max_range = max_range
        self.is_3d = is_3d

    def build_features(
        self,
        robot_state: RobotState,
        mission: Mission,
        sensor_data: Optional[any] = None,
    ) -> Tuple[FeatureVector, LocalContext]:
        """Build feature vector from current state.

        Parameters
        ----------
        robot_state:
            Current robot state.
        mission:
            Current mission.
        sensor_data:
            Optional sensor data.

        Returns
        -------
        Tuple[FeatureVector, LocalContext]
            Feature vector and local context.
        """
        # Get graph snapshot
        graph_snapshot = self.topology_service.get_graph_snapshot(robot_state.time)

        # Resolve goal position
        goal_position = self._resolve_goal(mission.goal)

        # Compute feature groups
        goal_ego = compute_goal_ego(
            robot_state.pose,
            goal_position,
            self.max_range,
            is_3d=self.is_3d,
        )

        neighbors_k = compute_neighbor_features(
            robot_state.pose,
            graph_snapshot,
            k=self.k_neighbors,
            current_path=None,  # No path for Milestone A
            max_range=self.max_range,
        )

        topo_ctx = compute_topo_context(
            graph_snapshot,
            current_node=robot_state.current_node,
            current_path=None,
        )

        num_safety_bands = 5 if self.is_3d else 4
        safety = compute_safety_features(
            robot_state.pose,
            sensor_data,
            num_bands=num_safety_bands,
            is_3d=self.is_3d,
        )

        dynamics = compute_dynamics_features(
            robot_state.previous_action.tolist() if robot_state.previous_action is not None else None,
        )

        # Create feature vector
        features = FeatureVector(
            goal_ego=goal_ego,
            neighbors_k=neighbors_k,
            topo_ctx=topo_ctx,
            safety=safety,
            dynamics=dynamics,
        )

        # Build local context
        local_context = LocalContext(
            current_node=robot_state.current_node,
            graph_snapshot=graph_snapshot,
            sensor_data=sensor_data,
        )

        return features, local_context

    def _resolve_goal(self, goal: MissionGoal) -> Tuple[float, float]:
        """Resolve goal to position.

        Parameters
        ----------
        goal:
            Mission goal.

        Returns
        -------
        Tuple[float, float]
            Goal position (x, y).
        """
        if goal.type == GoalType.POINT:
            return goal.value.position[:2]
        elif goal.type == GoalType.NODE:
            # Get node position from graph
            snapshot = self.topology_service.get_graph_snapshot(0.0)
            node_data = next((n for n in snapshot.V if n.node_id == goal.value.node_id), None)
            if node_data:
                return node_data.position[:2]
            else:
                # Fallback: return origin
                return (0.0, 0.0)
        else:
            # Default: return origin
            return (0.0, 0.0)

