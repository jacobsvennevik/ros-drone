# Graph Navigation Service (GNS) Design

This document specifies the Graph Navigation Service that handles path planning on the topological graph, integrating with the SNN Policy Service for waypoint selection.

## 1. Overview

The **Graph Navigation Service (GNS)** provides path planning capabilities on the topological graph. It bridges high-level waypoint selection with the reactive SNN policy.

**Responsibilities**:
- Find paths between graph nodes
- Select next waypoint given current pose and goal
- Handle disconnected graphs
- Integrate with SPS for hierarchical planning (optional)

**Location**: `src/hippocampus_core/policy/graph_navigation.py`

## 2. Core Interface

```python
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from ..topology import TopologicalGraph
from .topology_service import GraphSnapshot

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
    position: Tuple[float, float]  # (x, y) or (x, y, z)
    distance: float  # Distance from current pose
    bearing: float  # Bearing from current pose (radians)

class GraphNavigationService:
    """Service for navigating the topological graph."""
    
    def __init__(
        self,
        graph_snapshot: GraphSnapshot,
        algorithm: str = "dijkstra",  # "dijkstra", "astar", "greedy"
    ):
        self.graph_snapshot = graph_snapshot
        self.algorithm = algorithm
        
    def find_path(
        self,
        start_node: int,
        goal_node: int,
    ) -> Optional[NavigationPath]:
        """Find path from start to goal node.
        
        Returns None if no path exists.
        """
        # Implementation depends on algorithm
        pass
    
    def select_next_waypoint(
        self,
        current_pose: Tuple[float, float, float],  # (x, y, yaw)
        goal: "MissionGoal",  # From mission representation
    ) -> Optional[WaypointTarget]:
        """Select next waypoint given current pose and goal.
        
        Returns None if goal is unreachable or already reached.
        """
        # 1. Find current node (nearest to current pose)
        current_node = self._find_nearest_node(current_pose[:2])
        
        # 2. Find goal node (from mission goal)
        goal_node = self._resolve_goal_node(goal)
        
        if goal_node is None:
            return None
        
        # 3. Find path
        path = self.find_path(current_node, goal_node)
        
        if path is None or not path.is_complete:
            return None
        
        # 4. Select next waypoint (first node in path after current)
        if len(path.nodes) < 2:
            # Already at goal
            return None
        
        next_node_id = path.nodes[1]
        next_node = self._get_node_data(next_node_id)
        
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
```

## 3. Path Planning Algorithms

### 3.1 Dijkstra's Algorithm

**Use case**: Optimal paths when all edges have equal or similar weights.

```python
import networkx as nx

def dijkstra_path(
    graph: nx.Graph,
    start: int,
    goal: int,
    weight: str = "distance",
) -> Optional[List[int]]:
    """Find shortest path using Dijkstra's algorithm."""
    try:
        path = nx.shortest_path(
            graph,
            source=start,
            target=goal,
            weight=weight,
        )
        return path
    except nx.NetworkXNoPath:
        return None
```

**Complexity**: O(E log V) where E=edges, V=vertices

### 3.2 A* Algorithm

**Use case**: Faster than Dijkstra when heuristic is available (Euclidean distance).

```python
def astar_path(
    graph: nx.Graph,
    start: int,
    goal: int,
    positions: np.ndarray,
    weight: str = "distance",
) -> Optional[List[int]]:
    """Find shortest path using A* algorithm."""
    
    def heuristic(u: int, v: int) -> float:
        """Euclidean distance heuristic."""
        pos_u = positions[u]
        pos_v = positions[v]
        return np.linalg.norm(pos_u - pos_v)
    
    try:
        path = nx.astar_path(
            graph,
            source=start,
            target=goal,
            heuristic=heuristic,
            weight=weight,
        )
        return path
    except nx.NetworkXNoPath:
        return None
```

**Complexity**: O(E) in best case, O(E log V) worst case

### 3.3 Greedy Nearest-Neighbor

**Use case**: Fast, suboptimal but good for reactive control.

```python
def greedy_path(
    graph: nx.Graph,
    start: int,
    goal: int,
    positions: np.ndarray,
) -> Optional[List[int]]:
    """Greedy path: always move to neighbor closest to goal."""
    path = [start]
    current = start
    visited = {start}
    
    while current != goal:
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
        goal_pos = positions[goal]
        best_neighbor = min(
            unvisited,
            key=lambda n: np.linalg.norm(positions[n] - goal_pos),
        )
        
        path.append(best_neighbor)
        visited.add(best_neighbor)
        current = best_neighbor
        
        # Prevent infinite loops
        if len(path) > graph.number_of_nodes():
            return None
    
    return path
```

**Complexity**: O(V) worst case, but may not find optimal path

## 4. Handling Disconnected Graphs

### 4.1 Component Detection

```python
def find_connected_component(
    graph: nx.Graph,
    node: int,
) -> set[int]:
    """Find all nodes in the same connected component."""
    return nx.node_connected_component(graph, node)

def is_reachable(
    graph: nx.Graph,
    start: int,
    goal: int,
) -> bool:
    """Check if goal is reachable from start."""
    try:
        nx.shortest_path_length(graph, start, goal)
        return True
    except nx.NetworkXNoPath:
        return False
```

### 4.2 Fallback Strategies

When graph is disconnected:

1. **Nearest component**: Move to nearest node in goal's component
2. **Exploration**: Move toward unexplored regions
3. **Reactive fallback**: Use pure reactive control (no waypoints)

```python
def handle_disconnected(
    self,
    current_node: int,
    goal_node: int,
    current_pose: Tuple[float, float, float],
) -> Optional[WaypointTarget]:
    """Handle case when goal is in different component."""
    # Strategy 1: Find nearest node to goal in current component
    current_component = find_connected_component(self.graph, current_node)
    goal_component = find_connected_component(self.graph, goal_node)
    
    if current_component == goal_component:
        # Should not happen if we're here
        return None
    
    # Find bridge node: node in current component closest to goal component
    goal_pos = self._get_node_data(goal_node).position
    bridge_node = min(
        current_component,
        key=lambda n: np.linalg.norm(
            np.array(self._get_node_data(n).position) - np.array(goal_pos)
        ),
    )
    
    # Return waypoint toward bridge node
    return self._create_waypoint_target(bridge_node, current_pose)
```

## 5. Integration with SNN Policy

### 5.1 Hierarchical Planning Mode

When using hierarchical planning, GNS provides waypoints that SPS uses as guidance:

```python
class HierarchicalPolicy:
    """Combines GNS path planning with SPS reactive control."""
    
    def __init__(self, gns: GraphNavigationService, sps: SpikingPolicyService):
        self.gns = gns
        self.sps = sps
        self.current_waypoint: Optional[WaypointTarget] = None
        
    def decide(
        self,
        current_pose: Tuple[float, float, float],
        goal: MissionGoal,
        features: FeatureVector,
    ) -> PolicyDecision:
        """Hierarchical decision: waypoint from GNS, action from SPS."""
        
        # Update waypoint if needed
        if self._should_replan(current_pose, goal):
            self.current_waypoint = self.gns.select_next_waypoint(current_pose, goal)
        
        # If no waypoint, fall back to reactive
        if self.current_waypoint is None:
            return self.sps.decide(features, local_context, dt)
        
        # Add waypoint bias to features
        waypoint_features = self._add_waypoint_bias(features, self.current_waypoint)
        
        # Get action from SPS
        decision = self.sps.decide(waypoint_features, local_context, dt)
        
        # Override waypoint in decision
        decision.next_waypoint = self.current_waypoint.node_id
        
        return decision
    
    def _add_waypoint_bias(
        self,
        features: FeatureVector,
        waypoint: WaypointTarget,
    ) -> FeatureVector:
        """Add waypoint information to features."""
        # Modify goal_ego to point toward waypoint instead of final goal
        # This biases the policy toward the waypoint
        waypoint_ego = [
            waypoint.distance,
            np.cos(waypoint.bearing),
            np.sin(waypoint.bearing),
        ]
        # Create modified feature vector
        modified = FeatureVector(
            goal_ego=waypoint_ego,  # Override with waypoint
            neighbors_k=features.neighbors_k,
            topo_ctx=features.topo_ctx,
            safety=features.safety,
            dynamics=features.dynamics,
        )
        return modified
```

### 5.2 Reactive Mode (No GNS)

For pure reactive control (Milestone A), SPS works without GNS:

```python
# SPS directly uses goal from mission
decision = sps.decide(features, local_context, dt)
# decision.next_waypoint is None
```

## 6. Dynamic Graph Updates

### 6.1 Graph Staleness

When graph updates, invalidate cached paths:

```python
class GraphNavigationService:
    def __init__(self, ...):
        self._cached_paths: dict[Tuple[int, int], NavigationPath] = {}
        self._graph_version = 0
        
    def update_graph(self, new_snapshot: GraphSnapshot) -> None:
        """Update graph snapshot and invalidate caches."""
        if new_snapshot.meta.epoch_id != self.graph_snapshot.meta.epoch_id:
            self.graph_snapshot = new_snapshot
            self._graph_version += 1
            self._cached_paths.clear()  # Invalidate cache
```

### 6.2 Incremental Updates

For efficiency, only recompute paths when necessary:

```python
def find_path_cached(
    self,
    start: int,
    goal: int,
) -> Optional[NavigationPath]:
    """Find path with caching."""
    cache_key = (start, goal)
    
    if cache_key in self._cached_paths:
        cached = self._cached_paths[cache_key]
        # Verify path is still valid
        if self._is_path_valid(cached):
            return cached
    
    # Compute new path
    path = self._compute_path(start, goal)
    if path:
        self._cached_paths[cache_key] = path
    return path
```

## 7. Node Resolution

### 7.1 Finding Nearest Node

```python
def _find_nearest_node(
    self,
    position: Tuple[float, float],
) -> int:
    """Find graph node nearest to given position."""
    min_dist = float('inf')
    nearest = None
    
    for node in self.graph_snapshot.V:
        node_pos = node.position
        dist = np.linalg.norm(
            np.array(position) - np.array(node_pos[:2])
        )
        if dist < min_dist:
            min_dist = dist
            nearest = node.node_id
    
    return nearest
```

### 7.2 Resolving Goal to Node

```python
def _resolve_goal_node(
    self,
    goal: "MissionGoal",
) -> Optional[int]:
    """Resolve mission goal to graph node ID."""
    if goal.type == "node":
        # Direct node ID
        return goal.value
    elif goal.type == "point":
        # Find nearest node to point
        return self._find_nearest_node(goal.value)
    elif goal.type == "region":
        # Find node in region (or nearest to region center)
        center = self._compute_region_center(goal.value)
        return self._find_nearest_node(center)
    else:
        return None
```

## 8. Edge Cases

### 8.1 Empty Graph

```python
if len(self.graph_snapshot.V) == 0:
    return None  # No nodes to navigate
```

### 8.2 Single Node

```python
if len(self.graph_snapshot.V) == 1:
    # Only one node, check if it's the goal
    if self._resolve_goal_node(goal) == 0:
        return None  # Already at goal
    else:
        return None  # Goal unreachable (no edges)
```

### 8.3 No Edges

```python
if len(self.graph_snapshot.E) == 0:
    # All nodes isolated
    # Fall back to reactive control
    return None
```

### 8.4 Goal Outside Graph

```python
# If goal is far from all nodes, may want to create virtual node
# For now, use nearest node as proxy
goal_node = self._find_nearest_node(goal_position)
if distance_to_nearest > threshold:
    # Goal is too far, may be unreachable
    return None
```

## 9. Performance Optimization

### 9.1 Precomputation

For static graphs, precompute all-pairs shortest paths:

```python
def precompute_all_paths(self) -> dict[Tuple[int, int], List[int]]:
    """Precompute all-pairs shortest paths."""
    all_paths = {}
    nodes = [n.node_id for n in self.graph_snapshot.V]
    
    for start in nodes:
        for goal in nodes:
            if start != goal:
                path = self._compute_path(start, goal)
                if path:
                    all_paths[(start, goal)] = path.nodes
    
    return all_paths
```

**Use case**: Small graphs (<100 nodes), static environments

### 9.2 Lazy Evaluation

Only compute paths when requested:

```python
# Don't precompute, compute on-demand
path = self.find_path(start, goal)  # Computed only when needed
```

**Use case**: Large graphs, dynamic environments

## 10. Testing

### 10.1 Unit Tests

```python
def test_dijkstra_path():
    """Test Dijkstra path finding."""
    graph = create_test_graph()
    gns = GraphNavigationService(graph, algorithm="dijkstra")
    path = gns.find_path(0, 5)
    assert path is not None
    assert path.nodes[0] == 0
    assert path.nodes[-1] == 5
    assert path.is_complete

def test_disconnected_graph():
    """Test handling of disconnected graph."""
    graph = create_disconnected_graph()
    gns = GraphNavigationService(graph)
    path = gns.find_path(0, 5)  # Different components
    assert path is None

def test_waypoint_selection():
    """Test waypoint selection."""
    gns = GraphNavigationService(graph)
    current_pose = (0.5, 0.5, 0.0)
    goal = MissionGoal(type="point", value=(0.9, 0.9))
    waypoint = gns.select_next_waypoint(current_pose, goal)
    assert waypoint is not None
    assert waypoint.node_id in graph.nodes
```

### 10.2 Integration Tests

```python
def test_hierarchical_policy():
    """Test GNS + SPS integration."""
    gns = GraphNavigationService(graph)
    sps = SpikingPolicyService(...)
    policy = HierarchicalPolicy(gns, sps)
    
    decision = policy.decide(current_pose, goal, features)
    assert decision.next_waypoint is not None
    assert decision.action_proposal.v >= 0
```

## 11. Summary

**Key Components**:

1. **Path Planning**: Dijkstra (optimal), A* (fast), Greedy (reactive)
2. **Waypoint Selection**: Nearest node → path → next waypoint
3. **Disconnected Handling**: Component detection, fallback strategies
4. **Integration**: Hierarchical (GNS + SPS) or reactive (SPS only)
5. **Performance**: Caching, lazy evaluation, precomputation (optional)

**Implementation Phases**:

- **Milestone A**: Not needed (pure reactive)
- **Milestone C**: Full implementation with A*/Dijkstra
- **Milestone D**: Optimization, caching, precomputation

This service enables hierarchical planning while maintaining the flexibility of reactive control when needed.

