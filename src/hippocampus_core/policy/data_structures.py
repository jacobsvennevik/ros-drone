"""Core data structures for the SNN Policy Service."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union, Dict
import numpy as np


class GoalType(str, Enum):
    """Types of mission goals."""

    POINT = "point"
    NODE = "node"
    REGION = "region"
    SEQUENTIAL = "sequential"
    EXPLORE = "explore"


@dataclass
class PointGoal:
    """Point goal: target position."""

    position: Tuple[float, ...]  # (x, y) for 2D or (x, y, z) for 3D
    tolerance: float = 0.1  # Distance tolerance (m)
    frame_id: str = "map"  # Coordinate frame


@dataclass
class NodeGoal:
    """Node goal: target graph node."""

    node_id: int
    tolerance: float = 0.05  # Distance tolerance (m)


@dataclass
class MissionGoal:
    """Unified mission goal representation."""

    type: GoalType
    value: Union[PointGoal, NodeGoal]  # Simplified for Milestone A
    priority: int = 1
    timeout: Optional[float] = None
    created_at: float = 0.0

    def is_reached(
        self,
        current_pose: Tuple[float, float, float],
        current_node: Optional[int] = None,
    ) -> bool:
        """Check if goal is reached."""
        if self.type == GoalType.POINT:
            goal = self.value
            dist = np.linalg.norm(
                np.array(current_pose[:2]) - np.array(goal.position[:2])
            )
            return dist <= goal.tolerance
        elif self.type == GoalType.NODE:
            goal = self.value
            if current_node == goal.node_id:
                return True
        return False


@dataclass
class MissionConstraints:
    """Constraints for mission execution."""

    no_fly_zones: List = field(default_factory=list)
    altitude_min: Optional[float] = None
    altitude_max: Optional[float] = None
    keepout_nodes: List[int] = field(default_factory=list)
    max_velocity: Optional[float] = None
    max_acceleration: Optional[float] = None
    geofence: Optional = None  # RegionGoal (simplified for Milestone A)


@dataclass
class Mission:
    """Complete mission specification."""

    goal: MissionGoal
    constraints: MissionConstraints = field(default_factory=MissionConstraints)
    metadata: Dict = field(default_factory=dict)

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate mission is feasible."""
        if self.goal.type == GoalType.POINT:
            goal = self.goal.value
            # Basic validation
            if goal.tolerance <= 0:
                return False, "Invalid tolerance"
        if self.goal.timeout is not None and self.goal.timeout <= 0:
            return False, "Invalid timeout"
        return True, None


@dataclass
class RobotState:
    """Robot state information."""

    pose: Tuple[float, ...]  # (x, y, yaw) for 2D or (x, y, z, yaw, pitch) for 3D
    twist: Optional[Tuple[float, ...]] = None  # (vx, vy, omega) for 2D or (vx, vy, vz, omega, pitch_rate) for 3D
    health: Dict[str, bool] = field(default_factory=lambda: {"sensor_ok": True, "localization_ok": True})
    time: float = 0.0
    current_node: Optional[int] = None
    covariance: Optional[np.ndarray] = None
    previous_action: Optional[np.ndarray] = None


@dataclass
class NodeData:
    """Data for a single graph node."""

    node_id: int
    position: Tuple[float, ...]  # (x, y) for 2D or (x, y, z) for 3D
    normal: Optional[Tuple[float, float, float]] = None  # Surface normal for 3D
    degree: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class EdgeData:
    """Data for a single graph edge."""

    u: int
    v: int
    length: float
    traversable: bool = True
    integrator_value: float = 0.0
    updated_at: float = 0.0


@dataclass
class GraphSnapshotMetadata:
    """Metadata for a graph snapshot."""

    epoch_id: int
    frame_id: str
    stamp: float
    last_updated: float
    update_rate: float
    staleness_warning: bool
    node_visit_counts: Dict[int, int] = field(default_factory=dict)


@dataclass
class GraphSnapshot:
    """Immutable snapshot of the topological graph."""

    V: List[NodeData]
    E: List[EdgeData]
    meta: GraphSnapshotMetadata


@dataclass
class FeatureVector:
    """Complete feature vector for SNN policy."""

    goal_ego: List[float]  # [distance_norm, cos(θ_g), sin(θ_g), ...]
    neighbors_k: List[List[float]]  # k × [cos(θ_j), sin(θ_j), d_j_norm, on_path]
    topo_ctx: List[float]  # [deg_norm, clustering, path_progress]
    safety: List[float]  # [front, left, right, back]_norm
    dynamics: Optional[List[float]] = None  # [prev_v_norm, prev_ω_norm, ...]

    def to_array(self) -> np.ndarray:
        """Convert to flat numpy array."""
        features = []
        features.extend(self.goal_ego)
        for neighbor in self.neighbors_k:
            features.extend(neighbor)
        features.extend(self.topo_ctx)
        features.extend(self.safety)
        if self.dynamics:
            features.extend(self.dynamics)
        return np.array(features, dtype=np.float32)

    @property
    def dim(self) -> int:
        """Total feature dimensionality."""
        dim = len(self.goal_ego)
        dim += sum(len(n) for n in self.neighbors_k)
        dim += len(self.topo_ctx)
        dim += len(self.safety)
        if self.dynamics:
            dim += len(self.dynamics)
        return dim


@dataclass
class ActionProposal:
    """Proposed action from policy."""

    v: float  # Linear velocity (m/s)
    omega: float  # Angular velocity (rad/s)
    vz: Optional[float] = None  # Vertical velocity (m/s) for 3D
    pitch_rate: Optional[float] = None  # Pitch rate (rad/s) for 3D


@dataclass
class PolicyDecision:
    """Policy decision output."""

    next_waypoint: Optional[int] = None
    action_proposal: ActionProposal
    confidence: float  # [0, 1]
    reason: str = "snn"  # "snn", "heuristic", "fallback"


@dataclass
class SafeCommand:
    """Safe command after arbitration."""

    cmd: Tuple[float, ...]  # (v, ω) for 2D or (v, ω, vz) for 3D
    safety_flags: Dict[str, bool] = field(default_factory=lambda: {"clamped": False, "slowed": False, "stop": False})
    latency_ms: float = 0.0


@dataclass
class LocalContext:
    """Local context for feature computation."""

    current_node: Optional[int] = None
    graph_snapshot: Optional[GraphSnapshot] = None
    sensor_data: Optional[object] = None  # Placeholder for sensor data

