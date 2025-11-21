# Mission Representation & Goal Management

This document specifies the mission and goal representation system for the SNN Policy Service.

## 1. Overview

The **Mission** system defines what the robot should accomplish, including goals, constraints, and dynamic updates.

**Location**: `src/hippocampus_core/policy/mission.py`

## 2. Core Data Structures

### 2.1 Mission Goal Types

```python
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
from enum import Enum
import numpy as np

class GoalType(str, Enum):
    """Types of mission goals."""
    POINT = "point"  # Target (x, y) or (x, y, z)
    NODE = "node"  # Target graph node ID
    REGION = "region"  # Target region (polygon or circle)
    SEQUENTIAL = "sequential"  # Sequence of waypoints
    EXPLORE = "explore"  # Exploration goal (no specific target)

@dataclass
class PointGoal:
    """Point goal: target position."""
    position: Tuple[float, float]  # (x, y) or (x, y, z) for 3D
    tolerance: float = 0.1  # Distance tolerance (m)
    frame_id: str = "map"  # Coordinate frame

@dataclass
class NodeGoal:
    """Node goal: target graph node."""
    node_id: int
    tolerance: float = 0.05  # Distance tolerance (m)

@dataclass
class RegionGoal:
    """Region goal: target region."""
    region_type: str  # "circle", "polygon", "rectangle"
    center: Tuple[float, float]  # For circle/rectangle
    radius: Optional[float] = None  # For circle
    width: Optional[float] = None  # For rectangle
    height: Optional[float] = None  # For rectangle
    vertices: Optional[List[Tuple[float, float]]] = None  # For polygon
    frame_id: str = "map"

@dataclass
class SequentialGoal:
    """Sequential goal: multiple waypoints in order."""
    waypoints: List[Union[PointGoal, NodeGoal, RegionGoal]]
    current_index: int = 0
    loop: bool = False  # Loop back to start after last waypoint

@dataclass
class ExploreGoal:
    """Exploration goal: explore unknown areas."""
    target_coverage: float = 0.8  # Target coverage fraction [0, 1]
    unexplored_regions: Optional[List[Tuple[float, float, float]]] = None  # (x, y, radius)

@dataclass
class MissionGoal:
    """Unified mission goal representation."""
    type: GoalType
    value: Union[PointGoal, NodeGoal, RegionGoal, SequentialGoal, ExploreGoal]
    priority: int = 1  # Higher = more important
    timeout: Optional[float] = None  # Timeout in seconds
    created_at: float = 0.0  # Timestamp
    
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
            # Also check distance
            # (implementation depends on graph access)
            return False
        elif self.type == GoalType.REGION:
            goal = self.value
            return self._point_in_region(current_pose[:2], goal)
        elif self.type == GoalType.SEQUENTIAL:
            goal = self.value
            if goal.current_index >= len(goal.waypoints):
                return True
            current_waypoint = goal.waypoints[goal.current_index]
            # Check if current waypoint is reached
            # (recursive check)
            return False
        elif self.type == GoalType.EXPLORE:
            # Exploration goals are never "reached" in the traditional sense
            # Check coverage instead
            return False
        return False
    
    def _point_in_region(
        self,
        point: Tuple[float, float],
        region: RegionGoal,
    ) -> bool:
        """Check if point is in region."""
        if region.region_type == "circle":
            dist = np.linalg.norm(
                np.array(point) - np.array(region.center)
            )
            return dist <= region.radius
        elif region.region_type == "polygon":
            # Point-in-polygon test
            vertices = region.vertices
            if vertices is None:
                return False
            return self._point_in_polygon(point, vertices)
        elif region.region_type == "rectangle":
            x, y = point
            cx, cy = region.center
            w, h = region.width, region.height
            return (cx - w/2 <= x <= cx + w/2 and
                    cy - h/2 <= y <= cy + h/2)
        return False
    
    def _point_in_polygon(
        self,
        point: Tuple[float, float],
        vertices: List[Tuple[float, float]],
    ) -> bool:
        """Ray casting algorithm for point-in-polygon."""
        x, y = point
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
```

### 2.2 Mission Constraints

```python
@dataclass
class MissionConstraints:
    """Constraints for mission execution."""
    no_fly_zones: List[RegionGoal] = None  # Prohibited regions
    altitude_min: Optional[float] = None  # Minimum altitude (3D)
    altitude_max: Optional[float] = None  # Maximum altitude (3D)
    keepout_nodes: List[int] = None  # Prohibited graph nodes
    max_velocity: Optional[float] = None  # Maximum velocity
    max_acceleration: Optional[float] = None  # Maximum acceleration
    geofence: Optional[RegionGoal] = None  # Operating boundary
    
    def __post_init__(self):
        if self.no_fly_zones is None:
            self.no_fly_zones = []
        if self.keepout_nodes is None:
            self.keepout_nodes = []
    
    def is_valid_position(
        self,
        position: Tuple[float, float],
        altitude: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Check if position is valid given constraints.
        
        Returns
        -------
        Tuple[bool, Optional[str]]
            (is_valid, reason) where reason is None if valid.
        """
        # Check geofence
        if self.geofence is not None:
            if not self._point_in_region(position, self.geofence):
                return False, "outside_geofence"
        
        # Check no-fly zones
        for nfz in self.no_fly_zones:
            if self._point_in_region(position, nfz):
                return False, "in_no_fly_zone"
        
        # Check altitude
        if altitude is not None:
            if self.altitude_min is not None and altitude < self.altitude_min:
                return False, "below_min_altitude"
            if self.altitude_max is not None and altitude > self.altitude_max:
                return False, "above_max_altitude"
        
        return True, None
```

### 2.3 Complete Mission

```python
@dataclass
class Mission:
    """Complete mission specification."""
    goal: MissionGoal
    constraints: MissionConstraints
    metadata: dict = None  # Additional metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate mission is feasible.
        
        Returns
        -------
        Tuple[bool, Optional[str]]
            (is_valid, error_message)
        """
        # Check goal is valid
        if self.goal.type == GoalType.POINT:
            goal = self.goal.value
            is_valid, reason = self.constraints.is_valid_position(goal.position)
            if not is_valid:
                return False, f"Goal position {reason}"
        
        # Check timeout
        if self.goal.timeout is not None and self.goal.timeout <= 0:
            return False, "Invalid timeout"
        
        return True, None
```

## 3. Goal Resolution

### 3.1 Resolving Goals to Positions

```python
class GoalResolver:
    """Resolves mission goals to target positions."""
    
    def __init__(self, graph_snapshot: Optional[GraphSnapshot] = None):
        self.graph_snapshot = graph_snapshot
    
    def resolve_goal_position(
        self,
        goal: MissionGoal,
        current_pose: Tuple[float, float, float],
    ) -> Optional[Tuple[float, float]]:
        """Resolve goal to target position (x, y).
        
        Returns None if goal cannot be resolved.
        """
        if goal.type == GoalType.POINT:
            return goal.value.position[:2]
        
        elif goal.type == GoalType.NODE:
            if self.graph_snapshot is None:
                return None
            node_goal = goal.value
            node_data = next(
                (n for n in self.graph_snapshot.V if n.node_id == node_goal.node_id),
                None
            )
            if node_data is None:
                return None
            return node_data.position[:2]
        
        elif goal.type == GoalType.REGION:
            # Return center of region
            return goal.value.center
        
        elif goal.type == GoalType.SEQUENTIAL:
            seq_goal = goal.value
            if seq_goal.current_index >= len(seq_goal.waypoints):
                return None
            current_waypoint = seq_goal.waypoints[seq_goal.current_index]
            # Recursively resolve
            sub_goal = MissionGoal(type=..., value=current_waypoint)
            return self.resolve_goal_position(sub_goal, current_pose)
        
        elif goal.type == GoalType.EXPLORE:
            # Return nearest unexplored region center
            explore_goal = goal.value
            if explore_goal.unexplored_regions:
                # Find nearest
                current_pos = np.array(current_pose[:2])
                nearest = min(
                    explore_goal.unexplored_regions,
                    key=lambda r: np.linalg.norm(np.array(r[:2]) - current_pos),
                )
                return nearest[:2]
            return None
        
        return None
```

## 4. Dynamic Goal Updates

### 4.1 Goal Update Interface

```python
class MissionManager:
    """Manages mission goals and updates."""
    
    def __init__(self):
        self.current_mission: Optional[Mission] = None
        self.goal_history: List[MissionGoal] = []
    
    def set_mission(self, mission: Mission) -> Tuple[bool, Optional[str]]:
        """Set new mission."""
        is_valid, error = mission.validate()
        if not is_valid:
            return False, error
        
        if self.current_mission is not None:
            self.goal_history.append(self.current_mission.goal)
        
        self.current_mission = mission
        return True, None
    
    def update_goal(self, new_goal: MissionGoal) -> Tuple[bool, Optional[str]]:
        """Update current goal (keeping constraints)."""
        if self.current_mission is None:
            return False, "No active mission"
        
        # Create new mission with updated goal
        updated_mission = Mission(
            goal=new_goal,
            constraints=self.current_mission.constraints,
        )
        
        return self.set_mission(updated_mission)
    
    def advance_sequential_goal(self) -> bool:
        """Advance to next waypoint in sequential goal."""
        if self.current_mission is None:
            return False
        
        goal = self.current_mission.goal
        if goal.type != GoalType.SEQUENTIAL:
            return False
        
        seq_goal = goal.value
        if seq_goal.current_index < len(seq_goal.waypoints) - 1:
            seq_goal.current_index += 1
            return True
        elif seq_goal.loop:
            seq_goal.current_index = 0
            return True
        
        return False
    
    def check_timeout(self, current_time: float) -> bool:
        """Check if current goal has timed out."""
        if self.current_mission is None:
            return False
        
        goal = self.current_mission.goal
        if goal.timeout is None:
            return False
        
        elapsed = current_time - goal.created_at
        return elapsed > goal.timeout
```

### 4.2 ROS 2 Integration

```python
# ROS 2 message for mission updates
# hippocampus_ros2/msg/Mission.msg
"""
MissionGoal goal
MissionConstraints constraints
"""

# In ROS node:
class MissionSubscriber:
    def __init__(self, mission_manager: MissionManager):
        self.mission_manager = mission_manager
        self.subscription = node.create_subscription(
            Mission,
            "/mission",
            self.mission_callback,
            10,
        )
    
    def mission_callback(self, msg: Mission) -> None:
        """Handle mission update."""
        mission = self._msg_to_mission(msg)
        success, error = self.mission_manager.set_mission(mission)
        if not success:
            node.get_logger().error(f"Failed to set mission: {error}")
```

## 5. Goal Validation

### 5.1 Reachability Check

```python
def check_goal_reachable(
    goal: MissionGoal,
    graph_snapshot: GraphSnapshot,
    current_node: int,
) -> Tuple[bool, Optional[str]]:
    """Check if goal is reachable from current position.
    
    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_reachable, reason)
    """
    if goal.type == GoalType.NODE:
        goal_node = goal.value.node_id
        # Check if nodes are in same connected component
        # (implementation depends on graph structure)
        return True, None
    
    elif goal.type == GoalType.POINT:
        # Find nearest node to goal
        goal_pos = goal.value.position
        # Check if there's a path from current_node to goal_node
        return True, None
    
    # etc.
    return True, None
```

### 5.2 Constraint Validation

```python
def validate_goal_constraints(
    goal: MissionGoal,
    constraints: MissionConstraints,
) -> Tuple[bool, Optional[str]]:
    """Validate goal satisfies constraints."""
    if goal.type == GoalType.POINT:
        is_valid, reason = constraints.is_valid_position(goal.value.position)
        if not is_valid:
            return False, f"Goal position violates constraints: {reason}"
    
    elif goal.type == GoalType.NODE:
        if goal.value.node_id in constraints.keepout_nodes:
            return False, "Goal node is in keepout list"
    
    # etc.
    return True, None
```

## 6. Summary

**Key Components**:

1. **Goal Types**: Point, Node, Region, Sequential, Explore
2. **Constraints**: No-fly zones, altitude limits, geofence, keepout nodes
3. **Validation**: Reachability, constraint checking, timeout handling
4. **Dynamic Updates**: Mission manager, goal history, sequential advancement
5. **ROS Integration**: Mission messages, subscribers, callbacks

**Implementation Phases**:

- **Milestone A**: Point and Node goals, basic constraints
- **Milestone B**: Region goals, sequential goals
- **Milestone C**: 3D goals (altitude), exploration goals
- **Milestone D**: Advanced validation, dynamic replanning

This specification provides a complete mission and goal management system for the policy.

