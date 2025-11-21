# Feature Engineering Specification

This document details the feature engineering pipeline for the SNN Policy Service, including normalization schemes, temporal context, and feature selection.

## 1. Overview

The **Spatial Feature Service (SFS)** converts topological graph snapshots, robot state, and mission goals into feature vectors suitable for SNN input.

**Location**: `src/hippocampus_core/policy/features.py`

## 2. Feature Vector Structure

### 2.1 Complete Feature Vector

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class FeatureVector:
    """Complete feature vector for SNN policy."""
    
    # Goal-relative features (egocentric)
    goal_ego: List[float]  # [distance_norm, cos(θ_g), sin(θ_g), sin(φ_g), cos(φ_g)] for 3D
    
    # Neighbor features (k nearest graph nodes)
    neighbors_k: List[List[float]]  # k × [cos(θ_j), sin(θ_j), d_j_norm, on_path]
    
    # Topological context
    topo_ctx: List[float]  # [deg_norm, clustering, path_progress]
    
    # Safety features (obstacle bands)
    safety: List[float]  # [front, left, right, up, down]_norm
    
    # Dynamics (previous commands)
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
```

### 2.2 Feature Dimensions

**2D Configuration** (default):
- `goal_ego`: 3 features (distance, cos(θ), sin(θ))
- `neighbors_k`: k × 4 features (k=8 → 32 features)
- `topo_ctx`: 3 features
- `safety`: 4 features (front, left, right, back)
- `dynamics`: 2 features (optional)
- **Total**: ~44 features (without dynamics) or ~46 (with dynamics)

**3D Configuration**:
- `goal_ego`: 5 features (distance, cos(θ), sin(θ), sin(φ), cos(φ))
- `neighbors_k`: k × 5 features (k=12 → 60 features)
- `topo_ctx`: 3 features
- `safety`: 5 features (front, left, right, up, down)
- `dynamics`: 3 features (optional)
- **Total**: ~73 features (without dynamics) or ~76 (with dynamics)

## 3. Feature Computation

### 3.1 Goal-Relative Features

```python
def compute_goal_ego(
    robot_pose: Tuple[float, float, float],  # (x, y, yaw)
    goal_position: Tuple[float, float],  # (x, y) or (x, y, z)
    is_3d: bool = False,
) -> List[float]:
    """Compute egocentric goal features.
    
    Returns
    -------
    List[float]
        [distance_norm, cos(θ_g), sin(θ_g)] for 2D
        [distance_norm, cos(θ_g), sin(θ_g), sin(φ_g), cos(φ_g)] for 3D
    """
    robot_pos = np.array([robot_pose[0], robot_pose[1]])
    goal_pos = np.array([goal_position[0], goal_position[1]])
    
    # Vector from robot to goal
    vec_to_goal = goal_pos - robot_pos
    distance = np.linalg.norm(vec_to_goal)
    
    # Normalize distance (assume max range, e.g., arena diagonal)
    max_range = 10.0  # Configurable
    distance_norm = np.clip(distance / max_range, 0.0, 1.0)
    
    # Bearing relative to robot heading
    goal_bearing = np.arctan2(vec_to_goal[1], vec_to_goal[0])
    relative_bearing = goal_bearing - robot_pose[2]
    relative_bearing = np.arctan2(np.sin(relative_bearing), np.cos(relative_bearing))  # Wrap to [-π, π]
    
    features = [
        distance_norm,
        np.cos(relative_bearing),
        np.sin(relative_bearing),
    ]
    
    if is_3d and len(goal_position) > 2:
        # Elevation angle
        dz = goal_position[2] - robot_pose[2] if len(robot_pose) > 2 else 0.0
        elevation = np.arctan2(dz, distance)
        features.extend([
            np.sin(elevation),
            np.cos(elevation),
        ])
    
    return features
```

### 3.2 Neighbor Features

```python
def compute_neighbor_features(
    robot_pose: Tuple[float, float, float],
    graph_snapshot: GraphSnapshot,
    k: int = 8,
    current_path: Optional[List[int]] = None,
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
        
    Returns
    -------
    List[List[float]]
        k × [cos(θ_j), sin(θ_j), d_j_norm, on_path]
    """
    robot_pos = np.array([robot_pose[0], robot_pose[1]])
    
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
        k_nearest.append((None, 1.0, np.array([0.0, 0.0])))  # Dummy node
    
    # Compute features for each neighbor
    neighbor_features = []
    max_range = 10.0  # Same as goal features
    
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
            np.cos(relative_bearing),
            np.sin(relative_bearing),
            dist_norm,
            on_path,
        ])
    
    return neighbor_features
```

### 3.3 Topological Context

```python
def compute_topo_context(
    graph_snapshot: GraphSnapshot,
    current_node: Optional[int] = None,
    current_path: Optional[List[int]] = None,
) -> List[float]:
    """Compute topological context features.
    
    Returns
    -------
    List[float]
        [deg_norm, clustering, path_progress]
    """
    # Node degree (normalized)
    if current_node is not None:
        node_data = next(n for n in graph_snapshot.V if n.node_id == current_node)
        degree = node_data.degree
        max_degree = max(n.degree for n in graph_snapshot.V) if graph_snapshot.V else 1
        deg_norm = degree / max_degree if max_degree > 0 else 0.0
    else:
        deg_norm = 0.0
    
    # Clustering coefficient (local connectivity)
    # Simplified: fraction of neighbors that are connected
    if current_node is not None and graph_snapshot.V:
        neighbors = [e.v for e in graph_snapshot.E if e.u == current_node]
        neighbors.extend([e.u for e in graph_snapshot.E if e.v == current_node])
        neighbors = list(set(neighbors))
        
        if len(neighbors) > 1:
            # Count edges between neighbors
            neighbor_edges = sum(
                1 for e in graph_snapshot.E
                if (e.u in neighbors and e.v in neighbors)
            )
            max_possible = len(neighbors) * (len(neighbors) - 1) / 2
            clustering = neighbor_edges / max_possible if max_possible > 0 else 0.0
        else:
            clustering = 0.0
    else:
        clustering = 0.0
    
    # Path progress (if on a path)
    path_progress = 0.0
    if current_path and current_node is not None:
        try:
            idx = current_path.index(current_node)
            path_progress = idx / len(current_path) if len(current_path) > 0 else 0.0
        except ValueError:
            path_progress = 0.0
    
    return [deg_norm, clustering, path_progress]
```

### 3.4 Safety Features

```python
def compute_safety_features(
    robot_pose: Tuple[float, float, float],
    sensor_data: "SensorData",  # Depth/LiDAR data
    num_bands: int = 4,  # front, left, right, back (or +up/down for 3D)
) -> List[float]:
    """Compute obstacle avoidance features.
    
    Parameters
    ----------
    robot_pose:
        Current robot pose (x, y, yaw).
    sensor_data:
        Depth or LiDAR scan data.
    num_bands:
        Number of directional bands (4 for 2D, 5 for 3D).
        
    Returns
    -------
    List[float]
        [front, left, right, back]_norm (or +up/down for 3D)
    """
    # Divide sensor field into bands
    # For 2D: front, left, right, back
    # For 3D: front, left, right, up, down
    
    if sensor_data.type == "depth":
        # Depth image: divide into regions
        depth_image = sensor_data.data  # (H, W) or (H, W, 1)
        h, w = depth_image.shape[:2]
        
        # Front band (center region)
        front_region = depth_image[h//3:2*h//3, w//3:2*w//3]
        front_min = np.min(front_region) if front_region.size > 0 else float('inf')
        
        # Left band (left region)
        left_region = depth_image[:, :w//3]
        left_min = np.min(left_region) if left_region.size > 0 else float('inf')
        
        # Right band (right region)
        right_region = depth_image[:, 2*w//3:]
        right_min = np.min(right_region) if right_region.size > 0 else float('inf')
        
        # Back band (bottom region, assuming forward is top)
        back_region = depth_image[2*h//3:, :]
        back_min = np.min(back_region) if back_region.size > 0 else float('inf')
        
        # Normalize to [0, 1] (closer = lower value, higher danger)
        max_range = sensor_data.max_range
        safety_features = [
            np.clip(front_min / max_range, 0.0, 1.0),
            np.clip(left_min / max_range, 0.0, 1.0),
            np.clip(right_min / max_range, 0.0, 1.0),
            np.clip(back_min / max_range, 0.0, 1.0),
        ]
        
    elif sensor_data.type == "lidar":
        # LiDAR scan: divide into angular sectors
        ranges = sensor_data.ranges  # (N,) array
        angles = sensor_data.angles  # (N,) array relative to robot heading
        
        # Define sectors
        front_mask = np.abs(angles) < np.pi / 4
        left_mask = (angles >= np.pi / 4) & (angles < 3 * np.pi / 4)
        right_mask = (angles >= -3 * np.pi / 4) & (angles < -np.pi / 4)
        back_mask = np.abs(angles) >= 3 * np.pi / 4
        
        front_min = np.min(ranges[front_mask]) if np.any(front_mask) else float('inf')
        left_min = np.min(ranges[left_mask]) if np.any(left_mask) else float('inf')
        right_min = np.min(ranges[right_mask]) if np.any(right_mask) else float('inf')
        back_min = np.min(ranges[back_mask]) if np.any(back_mask) else float('inf')
        
        max_range = sensor_data.max_range
        safety_features = [
            np.clip(front_min / max_range, 0.0, 1.0),
            np.clip(left_min / max_range, 0.0, 1.0),
            np.clip(right_min / max_range, 0.0, 1.0),
            np.clip(back_min / max_range, 0.0, 1.0),
        ]
    else:
        # No sensor data: assume safe
        safety_features = [1.0] * num_bands
    
    return safety_features
```

### 3.5 Dynamics Features

```python
def compute_dynamics_features(
    previous_actions: Optional[List[float]],  # [v, ω] or [v, ω, vz]
    max_linear: float = 0.3,
    max_angular: float = 1.0,
) -> Optional[List[float]]:
    """Compute dynamics features from previous actions.
    
    Returns None if no previous actions available.
    """
    if previous_actions is None or len(previous_actions) < 2:
        return None
    
    v_norm = previous_actions[0] / max_linear if max_linear > 0 else 0.0
    omega_norm = previous_actions[1] / max_angular if max_angular > 0 else 0.0
    
    features = [v_norm, omega_norm]
    
    if len(previous_actions) > 2:
        # 3D: include vertical velocity
        max_vertical = 0.2  # Configurable
        vz_norm = previous_actions[2] / max_vertical if max_vertical > 0 else 0.0
        features.append(vz_norm)
    
    return features
```

## 4. Normalization Schemes

### 4.1 Z-Score Normalization

```python
class ZScoreNormalizer:
    """Z-score normalization: (x - μ) / σ"""
    
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = np.maximum(std, 1e-6)  # Avoid division by zero
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.std
    
    @classmethod
    def fit(cls, feature_matrix: np.ndarray) -> "ZScoreNormalizer":
        """Fit normalizer to data."""
        mean = np.mean(feature_matrix, axis=0)
        std = np.std(feature_matrix, axis=0)
        return cls(mean, std)
```

**Use case**: Features with unknown distributions, training data available

### 4.2 Min-Max Normalization

```python
class MinMaxNormalizer:
    """Min-max normalization: (x - min) / (max - min)"""
    
    def __init__(self, min_val: np.ndarray, max_val: np.ndarray):
        self.min_val = min_val
        self.max_val = max_val
        self.range = np.maximum(max_val - min_val, 1e-6)
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        return (features - self.min_val) / self.range
    
    @classmethod
    def fit(cls, feature_matrix: np.ndarray) -> "MinMaxNormalizer":
        """Fit normalizer to data."""
        min_val = np.min(feature_matrix, axis=0)
        max_val = np.max(feature_matrix, axis=0)
        return cls(min_val, max_val)
```

**Use case**: Features with known bounds (e.g., angles, distances)

### 4.3 Unit Vector Normalization

```python
class UnitVectorNormalizer:
    """Unit vector normalization: x / ||x||"""
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(features)
        if norm < 1e-6:
            return features
        return features / norm
```

**Use case**: Direction vectors (bearings, orientations)

### 4.4 Feature-Specific Normalization

Different features may need different normalization:

```python
class FeatureNormalizer:
    """Normalizer that applies different schemes to different feature groups."""
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.goal_normalizer = UnitVectorNormalizer()  # Bearings are unit vectors
        self.neighbor_normalizer = MinMaxNormalizer(...)  # Distances have bounds
        self.safety_normalizer = MinMaxNormalizer(...)  # Safety bands [0, 1]
        # etc.
    
    def normalize(self, features: FeatureVector) -> FeatureVector:
        """Normalize each feature group appropriately."""
        # Goal features: unit vector for bearings, min-max for distance
        goal_norm = self._normalize_goal(features.goal_ego)
        
        # Neighbors: min-max for all
        neighbors_norm = [self.neighbor_normalizer.normalize(np.array(n)) for n in features.neighbors_k]
        
        # Safety: already [0, 1], no normalization needed
        safety_norm = features.safety
        
        return FeatureVector(
            goal_ego=goal_norm,
            neighbors_k=neighbors_norm,
            topo_ctx=features.topo_ctx,  # Already normalized
            safety=safety_norm,
            dynamics=features.dynamics,  # Already normalized
        )
```

## 5. Temporal Context

### 5.1 History Buffer

```python
from collections import deque
from typing import Deque, Optional

class TemporalContext:
    """Maintains temporal context for features."""
    
    def __init__(self, history_length: int = 10):
        self.history_length = history_length
        self.feature_history: Deque[FeatureVector] = deque(maxlen=history_length)
        self.action_history: Deque[np.ndarray] = deque(maxlen=history_length)
        self.time_history: Deque[float] = deque(maxlen=history_length)
    
    def update(
        self,
        features: FeatureVector,
        action: np.ndarray,
        time: float,
    ) -> None:
        """Update history buffers."""
        self.feature_history.append(features)
        self.action_history.append(action)
        self.time_history.append(time)
    
    def get_temporal_features(
        self,
        current_features: FeatureVector,
    ) -> FeatureVector:
        """Create feature vector with temporal context.
        
        Adds features from history to current features.
        """
        if not self.feature_history:
            return current_features
        
        # Compute temporal derivatives
        prev_features = self.feature_history[-1]
        
        # Velocity features (change in goal distance, bearing)
        goal_velocity = [
            current_features.goal_ego[0] - prev_features.goal_ego[0],  # Distance change
            current_features.goal_ego[1] - prev_features.goal_ego[1],  # Bearing change (cos)
            current_features.goal_ego[2] - prev_features.goal_ego[2],  # Bearing change (sin)
        ]
        
        # Add to current features (extend goal_ego)
        extended_goal = list(current_features.goal_ego) + goal_velocity
        
        return FeatureVector(
            goal_ego=extended_goal,
            neighbors_k=current_features.neighbors_k,
            topo_ctx=current_features.topo_ctx,
            safety=current_features.safety,
            dynamics=current_features.dynamics,
        )
    
    def reset(self) -> None:
        """Reset history."""
        self.feature_history.clear()
        self.action_history.clear()
        self.time_history.clear()
```

### 5.2 Temporal Aggregation

For multi-step SNN inference, aggregate features over time:

```python
def aggregate_temporal_features(
    feature_history: List[FeatureVector],
    method: str = "mean",  # "mean", "max", "last"
) -> FeatureVector:
    """Aggregate features over time window."""
    if method == "mean":
        # Average features
        goal_ego = np.mean([f.goal_ego for f in feature_history], axis=0)
        # etc.
    elif method == "max":
        # Max pooling (for safety features)
        safety = np.max([f.safety for f in feature_history], axis=0)
    elif method == "last":
        # Use most recent
        return feature_history[-1]
    
    return FeatureVector(...)
```

## 6. Feature Selection & Ablation

### 6.1 Feature Groups

Features can be grouped for ablation studies:

```python
FEATURE_GROUPS = {
    "goal": ["goal_ego"],
    "neighbors": ["neighbors_k"],
    "topology": ["topo_ctx"],
    "safety": ["safety"],
    "dynamics": ["dynamics"],
}

def create_ablated_features(
    features: FeatureVector,
    disabled_groups: List[str],
) -> FeatureVector:
    """Create feature vector with some groups disabled (zeroed)."""
    goal_ego = features.goal_ego if "goal" not in disabled_groups else [0.0] * len(features.goal_ego)
    neighbors_k = features.neighbors_k if "neighbors" not in disabled_groups else [[0.0] * 4] * len(features.neighbors_k)
    # etc.
    return FeatureVector(...)
```

### 6.2 Ablation Study Plan

1. **Baseline**: All features
2. **No goal**: Remove goal_ego → test reactive behavior
3. **No neighbors**: Remove neighbors_k → test without graph
4. **No safety**: Remove safety → test obstacle avoidance dependency
5. **No dynamics**: Remove dynamics → test temporal dependency

## 7. Feature Validation

### 7.1 Range Checks

```python
def validate_features(features: FeatureVector) -> bool:
    """Validate feature ranges."""
    # Goal features: distance [0, 1], bearings [-1, 1]
    if not all(0.0 <= f <= 1.0 for f in [features.goal_ego[0]]):
        return False
    if not all(-1.0 <= f <= 1.0 for f in features.goal_ego[1:]):
        return False
    
    # Safety: [0, 1]
    if not all(0.0 <= f <= 1.0 for f in features.safety):
        return False
    
    # etc.
    return True
```

### 7.2 NaN/Inf Checks

```python
def check_finite(features: FeatureVector) -> bool:
    """Check for NaN or Inf values."""
    feature_array = features.to_array()
    return np.all(np.isfinite(feature_array))
```

## 8. Implementation

### 8.1 SpatialFeatureService

```python
class SpatialFeatureService:
    """Service that builds features from graph, robot state, and mission."""
    
    def __init__(
        self,
        topology_service: TopologyService,
        normalizer: Optional[FeatureNormalizer] = None,
        temporal_context: Optional[TemporalContext] = None,
        k_neighbors: int = 8,
    ):
        self.topology_service = topology_service
        self.normalizer = normalizer
        self.temporal_context = temporal_context
        self.k_neighbors = k_neighbors
    
    def build_features(
        self,
        robot_state: RobotState,
        mission: Mission,
        sensor_data: Optional[SensorData] = None,
        previous_action: Optional[np.ndarray] = None,
    ) -> Tuple[FeatureVector, LocalContext]:
        """Build feature vector from current state."""
        
        # Get graph snapshot
        graph_snapshot = self.topology_service.get_graph_snapshot(robot_state.time)
        
        # Resolve goal
        goal_position = self._resolve_goal(mission)
        
        # Compute feature groups
        goal_ego = compute_goal_ego(robot_state.pose, goal_position)
        neighbors_k = compute_neighbor_features(
            robot_state.pose,
            graph_snapshot,
            k=self.k_neighbors,
        )
        topo_ctx = compute_topo_context(graph_snapshot, robot_state.current_node)
        safety = compute_safety_features(robot_state.pose, sensor_data) if sensor_data else [1.0] * 4
        dynamics = compute_dynamics_features(previous_action) if previous_action is not None else None
        
        # Create feature vector
        features = FeatureVector(
            goal_ego=goal_ego,
            neighbors_k=neighbors_k,
            topo_ctx=topo_ctx,
            safety=safety,
            dynamics=dynamics,
        )
        
        # Apply temporal context
        if self.temporal_context:
            features = self.temporal_context.get_temporal_features(features)
        
        # Normalize
        if self.normalizer:
            features = self.normalizer.normalize(features)
        
        # Validate
        if not validate_features(features):
            raise ValueError("Invalid feature values")
        
        # Build local context
        local_context = LocalContext(
            current_node=robot_state.current_node,
            graph_snapshot=graph_snapshot,
            sensor_data=sensor_data,
        )
        
        return features, local_context
```

## 9. Summary

**Key Components**:

1. **Feature Groups**: Goal, neighbors, topology, safety, dynamics
2. **Normalization**: Z-score, min-max, unit vector (feature-specific)
3. **Temporal Context**: History buffers, temporal derivatives, aggregation
4. **Validation**: Range checks, NaN/Inf detection
5. **Ablation**: Feature group disabling for studies

**Feature Dimensions**:
- 2D: ~44-46 features
- 3D: ~73-76 features

**Implementation Phases**:
- **Milestone A**: Basic features (goal, neighbors, safety)
- **Milestone B**: Temporal context, normalization
- **Milestone C**: 3D features, advanced topology
- **Milestone D**: Ablation studies, optimization

This specification provides a complete feature engineering pipeline for the SNN policy.

