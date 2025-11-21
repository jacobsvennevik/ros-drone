# SNN Policy Architecture: Integration Analysis

This document specifies how the SNN Policy components integrate with the existing codebase, particularly `TopologicalGraph`, `PlaceCellController`, and `BrainNode`.

## 1. Topology Service (TS) Integration

### 1.1 Wrapping TopologicalGraph

The **Topology Service** wraps the existing `TopologicalGraph` class rather than duplicating its functionality.

**Location**: `src/hippocampus_core/policy/topology_service.py`

**Design**:

```python
from typing import Optional
from dataclasses import dataclass
import numpy as np
from ..topology import TopologicalGraph

@dataclass
class GraphSnapshotMetadata:
    """Metadata for a graph snapshot."""
    epoch_id: int
    frame_id: str
    stamp: float  # ROS timestamp in seconds
    last_updated: float  # Simulation time when graph was last built
    update_rate: float  # Hz - how often graph is updated
    staleness_warning: bool  # True if graph is stale
    node_visit_counts: dict[int, int]  # Track exploration

class TopologyService:
    """Service that wraps TopologicalGraph and provides snapshots for policy."""
    
    def __init__(self, graph: TopologicalGraph, frame_id: str = "map"):
        self._graph = graph
        self._frame_id = frame_id
        self._epoch_id = 0
        self._last_update_time = 0.0
        self._node_visit_counts: dict[int, int] = {}
        self._staleness_threshold = 5.0  # seconds
        
    def get_graph_snapshot(self, current_time: float) -> GraphSnapshot:
        """Get an immutable snapshot of the current graph state.
        
        Returns a GraphSnapshot with:
        - V: List of nodes with (node_id, position, normal, degree, tags)
        - E: List of edges with (u, v, length, traversable, integrator_value, updated_at)
        - meta: Metadata including epoch_id, frame_id, stamp
        """
        # Check staleness
        time_since_update = current_time - self._last_update_time
        staleness_warning = time_since_update > self._staleness_threshold
        
        # Build node list
        nodes = []
        positions = self._graph.positions
        degrees = self._graph.get_node_degrees()
        for i in range(self._graph.num_nodes()):
            pos = positions[i]
            nodes.append({
                'node_id': i,
                'position': (float(pos[0]), float(pos[1])),
                'normal': None,  # Optional surface normal for 3D
                'degree': int(degrees[i]),
                'tags': [],  # Future: obstacle_adjacent, explored, etc.
            })
        
        # Build edge list
        edges = []
        for u, v in self._graph.graph.edges():
            edge_data = self._graph.graph[u][v]
            distance = np.linalg.norm(positions[u] - positions[v])
            edges.append({
                'u': u,
                'v': v,
                'length': float(distance),
                'traversable': True,  # Future: could mark blocked edges
                'integrator_value': edge_data.get('weight', 0.0),
                'updated_at': self._last_update_time,
            })
        
        return GraphSnapshot(
            V=nodes,
            E=edges,
            meta=GraphSnapshotMetadata(
                epoch_id=self._epoch_id,
                frame_id=self._frame_id,
                stamp=current_time,
                last_updated=self._last_update_time,
                update_rate=1.0 / max(time_since_update, 0.1),
                staleness_warning=staleness_warning,
                node_visit_counts=self._node_visit_counts.copy(),
            )
        )
    
    def update_from_controller(self, controller: PlaceCellController) -> None:
        """Update the graph from a PlaceCellController.
        
        This should be called periodically (e.g., every 1-5 seconds)
        to refresh the graph snapshot.
        """
        self._graph = controller.get_graph()
        self._last_update_time = controller.current_time
        self._epoch_id += 1
```

### 1.2 GraphSnapshot Data Contract

```python
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass(frozen=True)
class NodeData:
    """Data for a single graph node."""
    node_id: int
    position: Tuple[float, float]  # (x, y) or (x, y, z) for 3D
    normal: Optional[Tuple[float, float, float]] = None  # Surface normal (3D)
    degree: int
    tags: List[str]  # e.g., ["obstacle_adjacent", "explored"]

@dataclass(frozen=True)
class EdgeData:
    """Data for a single graph edge."""
    u: int
    v: int
    length: float
    traversable: bool
    integrator_value: float  # Coactivity count
    updated_at: float  # Timestamp

@dataclass(frozen=True)
class GraphSnapshot:
    """Immutable snapshot of the topological graph."""
    V: List[NodeData]
    E: List[EdgeData]
    meta: GraphSnapshotMetadata
```

## 2. Vehicle Interface (VI) Integration

### 2.1 Extending BrainNode

The **Vehicle Interface** extends the existing `BrainNode` to subscribe to policy decisions rather than directly calling controllers.

**Location**: `ros2_ws/src/hippocampus_ros2/hippocampus_ros2/nodes/policy_node.py`

**Design**:

```python
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from hippocampus_ros2.msg import PolicyDecision, SafeCommand

class PolicyNode(Node):
    """ROS 2 node that subscribes to policy decisions and publishes cmd_vel.
    
    This extends BrainNode's functionality by:
    1. Subscribing to /policy/decision (from SPS)
    2. Subscribing to /odom (robot pose)
    3. Publishing /cmd_vel (velocity commands)
    4. Publishing /policy/status (diagnostics)
    """
    
    def __init__(self):
        super().__init__("policy_node")
        
        # Subscriptions
        self._decision_sub = self.create_subscription(
            PolicyDecision,
            "/policy/decision",
            self._decision_callback,
            10
        )
        self._odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self._odom_callback,
            10
        )
        
        # Publishers
        self._cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._status_pub = self.create_publisher(
            PolicyStatus,
            "/policy/status",
            10
        )
        
        # State
        self._last_decision: Optional[PolicyDecision] = None
        self._last_pose: Optional[Tuple[float, float, float]] = None  # x, y, yaw
        
    def _decision_callback(self, msg: PolicyDecision) -> None:
        """Handle policy decision from SPS."""
        self._last_decision = msg
        
        if self._last_pose is None:
            return
            
        # Convert decision to Twist
        twist = Twist()
        twist.linear.x = float(msg.action_proposal.linear_x)
        twist.linear.y = float(msg.action_proposal.linear_y)
        twist.linear.z = float(msg.action_proposal.linear_z) if hasattr(msg.action_proposal, 'linear_z') else 0.0
        twist.angular.z = float(msg.action_proposal.angular_z)
        
        self._cmd_vel_pub.publish(twist)
        
    def _odom_callback(self, msg: Odometry) -> None:
        """Update robot pose."""
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        yaw = self._quat_to_yaw(orient.x, orient.y, orient.z, orient.w)
        self._last_pose = (float(pos.x), float(pos.y), yaw)
```

### 2.2 Alternative: Policy Service Node

Alternatively, create a separate service node that orchestrates all policy components:

**Location**: `ros2_ws/src/hippocampus_ros2/hippocampus_ros2/nodes/policy_service_node.py`

This node would:
- Subscribe to `/odom`, `/scan` (or `/depth`), `/topo_graph`
- Call SFS, SPS, AAS internally
- Publish `/cmd_vel`, `/policy/decision`, `/policy/status`

This keeps all policy logic in one node, making it easier to manage state and timing.

## 3. Spiking Policy Service (SPS) Integration

### 3.1 Following SNNController Interface

The **SPS** should follow the existing `SNNController` interface pattern.

**Location**: `src/hippocampus_core/policy/spiking_policy_service.py`

**Design**:

```python
from ..controllers.base import SNNController
from .features import FeatureVector, SpatialFeatureService
from .topology_service import GraphSnapshot
from .mission import Mission

class SpikingPolicyService(SNNController):
    """Spiking neural network policy that decides where to fly.
    
    This implements the SNNController interface so it can be used
    in the same way as PlaceCellController or SnnTorchController.
    """
    
    def __init__(
        self,
        feature_service: SpatialFeatureService,
        snn_model: Optional[SNNModel] = None,  # Future: trained SNN
        config: Optional[PolicyConfig] = None,
    ):
        self._feature_service = feature_service
        self._snn_model = snn_model
        self._config = config or PolicyConfig()
        self._decision_history: deque = deque(maxlen=10)
        
    def reset(self) -> None:
        """Reset policy state."""
        self._decision_history.clear()
        if self._snn_model:
            self._snn_model.reset()
    
    def step(self, obs: np.ndarray, dt: float) -> np.ndarray:
        """Advance policy by one step.
        
        Parameters
        ----------
        obs:
            Observation vector [x, y, cos(heading), sin(heading), ...]
        dt:
            Time step duration
            
        Returns
        -------
        np.ndarray
            Action vector [v, ω] or [v, ω, vz] for 3D
        """
        # Extract robot state from observation
        robot_state = self._parse_observation(obs)
        
        # Get graph snapshot (from TS)
        graph_snapshot = self._get_graph_snapshot()
        
        # Get mission (from mission service or config)
        mission = self._get_current_mission()
        
        # Build features
        features, local_context = self._feature_service.build_features(
            graph_snapshot=graph_snapshot,
            robot_state=robot_state,
            mission=mission,
        )
        
        # Make decision
        decision = self.decide(features, local_context, dt)
        
        # Store in history
        self._decision_history.append(decision)
        
        # Return action proposal as numpy array
        action = decision.action_proposal
        return np.array([action.v, action.omega], dtype=float)
    
    def decide(
        self,
        features: FeatureVector,
        local_context: LocalContext,
        dt: float,
    ) -> PolicyDecision:
        """Make a policy decision from features.
        
        This is the core decision-making method that can be called
        independently of the step() method.
        """
        if self._snn_model is None:
            # Heuristic stub for Milestone A
            return self._heuristic_decide(features, local_context)
        
        # Future: SNN inference
        # spikes = self._encode_features(features, dt)
        # output_spikes = self._snn_model.forward(spikes)
        # decision = self._decode_spikes(output_spikes, features, local_context)
        # return decision
        
        raise NotImplementedError("SNN inference not yet implemented")
    
    def _heuristic_decide(
        self,
        features: FeatureVector,
        local_context: LocalContext,
    ) -> PolicyDecision:
        """Heuristic decision maker (Milestone A stub)."""
        # Simple: move toward goal, avoid obstacles
        goal_ego = features.goal_ego
        dg = goal_ego[0]  # Distance to goal
        theta_g = np.arctan2(goal_ego[2], goal_ego[1])  # Bearing to goal
        
        # Basic velocity command
        v = min(dg * 0.5, 0.3)  # Proportional to distance, capped
        omega = theta_g * 0.5  # Proportional to bearing error
        
        return PolicyDecision(
            next_waypoint=None,  # No explicit waypoint in reactive mode
            action_proposal=ActionProposal(v=v, omega=omega),
            confidence=0.5,  # Low confidence for heuristic
            reason=DecisionReason.HEURISTIC,
        )
```

### 3.2 Integration with PlaceCellController

The SPS can work alongside `PlaceCellController`:

```python
# In a ROS node or simulation:
place_controller = PlaceCellController(environment, config, rng)

# TS wraps the graph from place_controller
ts = TopologyService(place_controller.get_graph())

# SFS uses TS
sfs = SpatialFeatureService(ts)

# SPS uses SFS
sps = SpikingPolicyService(sfs)

# In control loop:
for step in range(num_steps):
    # Update mapping
    position = agent.step(dt)
    place_controller.step(position, dt)
    
    # Update TS
    ts.update_from_controller(place_controller)
    
    # Get policy decision
    obs = np.array([position[0], position[1], cos(heading), sin(heading)])
    action = sps.step(obs, dt)
    
    # Apply action to agent
    agent.apply_action(action)
```

## 4. Data Flow Integration

### 4.1 Complete Integration Flow

```
PlaceCellController (existing)
    ↓ get_graph()
TopologicalGraph (existing)
    ↓ wrapped by
TopologyService
    ↓ get_graph_snapshot()
GraphSnapshot
    ↓ used by
SpatialFeatureService
    ↓ build_features()
FeatureVector
    ↓ used by
SpikingPolicyService
    ↓ decide()
PolicyDecision
    ↓ filtered by
ActionArbitrationSafety
    ↓ filter()
SafeCommand
    ↓ published by
PolicyNode (extends BrainNode)
    ↓ /cmd_vel
Robot
```

### 4.2 ROS 2 Topic Integration

**Existing topics** (from BrainNode):
- `/odom` (subscribed)
- `/cmd_vel` (published)

**New topics** (for policy system):
- `/topo_graph` (latched, published by mapping node)
- `/mission` (subscribed, goal updates)
- `/policy/decision` (published by SPS node)
- `/policy/status` (published by policy node)
- `/policy/diagnostics` (published by policy node)

## 5. Configuration Integration

### 5.1 Reusing Existing Config Patterns

The policy system should follow the same configuration pattern as `PlaceCellControllerConfig`:

```python
@dataclass
class PolicyConfig:
    """Configuration for SNN Policy Service."""
    
    # Feature parameters
    k_neighbors: int = 8  # Number of nearest neighbors to consider
    feature_normalization: str = "z_score"  # "z_score", "min_max", "unit_vector"
    
    # SNN parameters (future)
    snn_model_path: Optional[str] = None
    spike_encoding: str = "rate"  # "rate", "temporal", "population"
    
    # Decision parameters
    decision_rate: float = 10.0  # Hz
    confidence_threshold: float = 0.3
    
    # Safety parameters
    max_linear_velocity: float = 0.3
    max_angular_velocity: float = 1.0
    collision_margin: float = 0.1
```

## 6. Testing Integration

### 6.1 Unit Tests

Tests should verify integration points:

```python
def test_topology_service_wraps_graph():
    """Test that TS correctly wraps TopologicalGraph."""
    graph = TopologicalGraph(positions)
    ts = TopologyService(graph)
    snapshot = ts.get_graph_snapshot(0.0)
    assert len(snapshot.V) == graph.num_nodes()
    assert len(snapshot.E) == graph.num_edges()

def test_sps_follows_snn_controller_interface():
    """Test that SPS implements SNNController correctly."""
    sps = SpikingPolicyService(feature_service)
    assert isinstance(sps, SNNController)
    sps.reset()
    action = sps.step(obs, dt)
    assert action.shape == (2,)
```

### 6.2 Integration Tests

End-to-end tests with existing components:

```python
def test_policy_with_place_cell_controller():
    """Test full integration: PlaceCellController → TS → SFS → SPS."""
    # Setup
    env = Environment(width=1.0, height=1.0)
    place_ctrl = PlaceCellController(env, config)
    ts = TopologyService(place_ctrl.get_graph())
    sfs = SpatialFeatureService(ts)
    sps = SpikingPolicyService(sfs)
    
    # Run simulation
    for step in range(100):
        position = agent.step(dt)
        place_ctrl.step(position, dt)
        ts.update_from_controller(place_ctrl)
        obs = np.array([position[0], position[1], 0.0, 0.0])
        action = sps.step(obs, dt)
        # Verify action is valid
        assert -0.3 <= action[0] <= 0.3
        assert -1.0 <= action[1] <= 1.0
```

## 7. Migration Path

### Phase 1: Milestone A
- Create `TopologyService` that wraps `TopologicalGraph`
- Create `SpatialFeatureService` that uses `TopologyService`
- Create `SpikingPolicyService` with heuristic stub
- Create `PolicyNode` that extends `BrainNode`
- Integration test with `PlaceCellController`

### Phase 2: Milestone B
- Add SNN inference to `SpikingPolicyService`
- Can reuse existing `SnnTorchController` patterns
- Add temporal context handling

### Phase 3: Milestone C
- Add 3D support to all services
- Add `GraphNavigationService` for path planning

## 8. Summary

**Key Integration Points**:

1. **TS wraps TopologicalGraph**: Don't duplicate, wrap and add metadata
2. **SPS follows SNNController**: Same interface as existing controllers
3. **VI extends BrainNode**: Reuse ROS infrastructure, add policy subscriptions
4. **Reuse configuration patterns**: Follow `PlaceCellControllerConfig` style
5. **Test integration**: Unit tests for wrappers, integration tests for full flow

This design ensures the policy system integrates smoothly with the existing codebase while maintaining clean separation of concerns.

