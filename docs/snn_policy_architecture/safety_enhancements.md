# Safety Enhancements Specification

This document specifies safety mechanisms for the SNN Policy Service, including graph staleness detection, localization drift handling, sensor fusion, and recovery behaviors.

## 1. Overview

The **Action Arbitration & Safety (AAS)** component enforces safety constraints and handles failure modes to ensure safe operation.

**Location**: `src/hippocampus_core/policy/safety.py`

## 2. Graph Staleness Detection

### 2.1 Staleness Metrics

```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class StalenessStatus:
    """Status of graph staleness."""
    is_stale: bool
    time_since_update: float  # seconds
    staleness_level: str  # "fresh", "warning", "stale", "critical"
    recommended_action: str  # "continue", "degrade", "hold", "estop"

class GraphStalenessDetector:
    """Detects when topological graph is stale."""
    
    def __init__(
        self,
        warning_threshold: float = 2.0,  # seconds
        stale_threshold: float = 5.0,
        critical_threshold: float = 10.0,
    ):
        self.warning_threshold = warning_threshold
        self.stale_threshold = stale_threshold
        self.critical_threshold = critical_threshold
    
    def check_staleness(
        self,
        graph_snapshot: GraphSnapshot,
        current_time: float,
    ) -> StalenessStatus:
        """Check if graph is stale."""
        time_since_update = current_time - graph_snapshot.meta.last_updated
        
        if time_since_update < self.warning_threshold:
            level = "fresh"
            action = "continue"
        elif time_since_update < self.stale_threshold:
            level = "warning"
            action = "degrade"
        elif time_since_update < self.critical_threshold:
            level = "stale"
            action = "hold"
        else:
            level = "critical"
            action = "estop"
        
        return StalenessStatus(
            is_stale=time_since_update >= self.stale_threshold,
            time_since_update=time_since_update,
            staleness_level=level,
            recommended_action=action,
        )
```

### 2.2 Degradation Strategies

```python
class SafetyMode(str, Enum):
    """Safety operation modes."""
    NORMAL = "normal"
    CONSERVATIVE = "conservative"  # Reduced speed, tighter constraints
    HOLD = "hold"  # Hover/stop, no movement
    E_STOP = "estop"  # Emergency stop, zero commands

def apply_staleness_degradation(
    decision: PolicyDecision,
    staleness: StalenessStatus,
) -> PolicyDecision:
    """Apply degradation based on staleness."""
    if staleness.recommended_action == "continue":
        return decision
    elif staleness.recommended_action == "degrade":
        # Reduce velocity by 50%
        decision.action_proposal.v *= 0.5
        decision.action_proposal.omega *= 0.5
        decision.confidence *= 0.7
    elif staleness.recommended_action == "hold":
        # Zero velocity, maintain position
        decision.action_proposal.v = 0.0
        decision.action_proposal.omega = 0.0
        decision.confidence = 0.0
        decision.reason = "graph_stale"
    elif staleness.recommended_action == "estop":
        # Emergency stop
        decision.action_proposal.v = 0.0
        decision.action_proposal.omega = 0.0
        decision.action_proposal.vz = 0.0 if decision.action_proposal.vz else None
        decision.confidence = 0.0
        decision.reason = "graph_critical_stale"
    
    return decision
```

## 3. Localization Drift Detection

### 3.1 Drift Estimation

```python
@dataclass
class LocalizationStatus:
    """Status of localization system."""
    is_healthy: bool
    drift_estimate: float  # meters
    covariance: Optional[np.ndarray] = None  # Position covariance
    last_update_time: float

class LocalizationMonitor:
    """Monitors localization quality."""
    
    def __init__(
        self,
        max_drift: float = 0.5,  # meters
        max_covariance: float = 0.1,  # m²
    ):
        self.max_drift = max_drift
        self.max_covariance = max_covariance
        self._pose_history: List[Tuple[float, float, float, float]] = []  # (x, y, yaw, time)
    
    def update_pose(
        self,
        pose: Tuple[float, float, float],
        covariance: Optional[np.ndarray],
        time: float,
    ) -> LocalizationStatus:
        """Update pose and check for drift."""
        self._pose_history.append((*pose, time))
        # Keep only recent history
        if len(self._pose_history) > 100:
            self._pose_history.pop(0)
        
        # Estimate drift from pose history
        drift = self._estimate_drift()
        
        # Check covariance
        cov_norm = np.trace(covariance) if covariance is not None else 0.0
        is_healthy = (drift < self.max_drift and 
                     cov_norm < self.max_covariance)
        
        return LocalizationStatus(
            is_healthy=is_healthy,
            drift_estimate=drift,
            covariance=covariance,
            last_update_time=time,
        )
    
    def _estimate_drift(self) -> float:
        """Estimate localization drift from pose history."""
        if len(self._pose_history) < 10:
            return 0.0
        
        # Simple: check for sudden jumps
        positions = np.array([(p[0], p[1]) for p in self._pose_history])
        velocities = np.diff(positions, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Large speed changes indicate drift/jump
        max_speed = np.max(speeds) if len(speeds) > 0 else 0.0
        # Assume max reasonable speed is 1 m/s
        # Anything above indicates potential drift
        drift_estimate = max(0.0, max_speed - 1.0)
        
        return drift_estimate
```

### 3.2 Drift Mitigation

```python
def apply_drift_mitigation(
    decision: PolicyDecision,
    localization: LocalizationStatus,
) -> PolicyDecision:
    """Apply mitigation for localization drift."""
    if not localization.is_healthy:
        # Reduce confidence and speed
        decision.confidence *= 0.5
        decision.action_proposal.v *= 0.5
        decision.action_proposal.omega *= 0.5
        decision.reason = "localization_drift"
    
    return decision
```

## 4. Sensor Fusion

### 4.1 Multi-Sensor Integration

```python
@dataclass
class SensorFusion:
    """Fused sensor data."""
    obstacle_map: np.ndarray  # 2D occupancy grid
    confidence: float  # [0, 1]
    sensor_sources: List[str]  # ["depth", "lidar", "sonar"]

class SensorFusionManager:
    """Manages fusion of multiple sensor sources."""
    
    def __init__(
        self,
        resolution: float = 0.1,  # meters per cell
        map_size: Tuple[int, int] = (100, 100),  # cells
    ):
        self.resolution = resolution
        self.map_size = map_size
        self.obstacle_map = np.zeros(map_size, dtype=np.float32)
        self.confidence_map = np.zeros(map_size, dtype=np.float32)
    
    def update_depth(
        self,
        depth_image: np.ndarray,
        pose: Tuple[float, float, float],
        max_range: float = 5.0,
    ) -> None:
        """Update obstacle map from depth image."""
        # Project depth image to 2D map
        # (implementation depends on camera model)
        pass
    
    def update_lidar(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        pose: Tuple[float, float, float],
        max_range: float = 10.0,
    ) -> None:
        """Update obstacle map from LiDAR scan."""
        # Project LiDAR points to 2D map
        for r, theta in zip(ranges, angles):
            if r > max_range:
                continue
            x = pose[0] + r * np.cos(pose[2] + theta)
            y = pose[1] + r * np.sin(pose[2] + theta)
            # Update obstacle map
            self._update_cell(x, y, occupied=True)
    
    def fuse(self) -> SensorFusion:
        """Fuse all sensor data."""
        # Combine obstacle maps from different sources
        # Weight by confidence
        fused_map = self.obstacle_map  # Simplified
        confidence = np.mean(self.confidence_map)
        
        return SensorFusion(
            obstacle_map=fused_map,
            confidence=confidence,
            sensor_sources=["depth", "lidar"],  # Example
        )
    
    def _update_cell(self, x: float, y: float, occupied: bool) -> None:
        """Update a cell in the obstacle map."""
        i = int(x / self.resolution) + self.map_size[0] // 2
        j = int(y / self.resolution) + self.map_size[1] // 2
        
        if 0 <= i < self.map_size[0] and 0 <= j < self.map_size[1]:
            if occupied:
                self.obstacle_map[i, j] = 1.0
            else:
                self.obstacle_map[i, j] = 0.0
```

### 4.2 Sensor Failure Detection

```python
class SensorHealthMonitor:
    """Monitors health of sensor systems."""
    
    def __init__(self):
        self.sensor_status: dict[str, bool] = {}
        self.last_update_times: dict[str, float] = {}
        self.timeout = 1.0  # seconds
    
    def update_sensor(
        self,
        sensor_name: str,
        data: Any,
        time: float,
    ) -> None:
        """Update sensor status."""
        self.sensor_status[sensor_name] = data is not None
        self.last_update_times[sensor_name] = time
    
    def check_health(self, current_time: float) -> dict[str, bool]:
        """Check health of all sensors."""
        health = {}
        for sensor_name in self.sensor_status:
            time_since_update = current_time - self.last_update_times.get(sensor_name, 0.0)
            is_healthy = (
                self.sensor_status.get(sensor_name, False) and
                time_since_update < self.timeout
            )
            health[sensor_name] = is_healthy
        return health
```

## 5. Recovery Behaviors

### 5.1 Stuck Detection

```python
class StuckDetector:
    """Detects when robot is stuck."""
    
    def __init__(
        self,
        stuck_threshold: float = 0.1,  # meters
        stuck_duration: float = 5.0,  # seconds
    ):
        self.stuck_threshold = stuck_threshold
        self.stuck_duration = stuck_duration
        self._position_history: List[Tuple[float, float, float]] = []  # (x, y, time)
    
    def check_stuck(
        self,
        current_pose: Tuple[float, float],
        current_time: float,
    ) -> Tuple[bool, Optional[str]]:
        """Check if robot is stuck."""
        self._position_history.append((*current_pose, current_time))
        # Keep only recent history
        self._position_history = [
            p for p in self._position_history
            if current_time - p[2] < self.stuck_duration
        ]
        
        if len(self._position_history) < 10:
            return False, None
        
        # Check if position has changed significantly
        positions = np.array([(p[0], p[1]) for p in self._position_history])
        max_displacement = np.max(np.linalg.norm(
            positions - positions[0], axis=1
        ))
        
        if max_displacement < self.stuck_threshold:
            return True, "position_stuck"
        
        return False, None
```

### 5.2 Recovery Actions

```python
class RecoveryBehavior:
    """Recovery behaviors for stuck/failure situations."""
    
    def __init__(self):
        self.recovery_attempts = 0
        self.max_attempts = 3
    
    def get_recovery_action(
        self,
        stuck_reason: str,
        current_pose: Tuple[float, float, float],
    ) -> Optional[ActionProposal]:
        """Get recovery action based on stuck reason."""
        if self.recovery_attempts >= self.max_attempts:
            return None  # Give up
        
        self.recovery_attempts += 1
        
        if stuck_reason == "position_stuck":
            # Spiral search: rotate and move backward
            return ActionProposal(
                v=-0.1,  # Move backward
                omega=0.5,  # Rotate
            )
        elif stuck_reason == "obstacle_blocked":
            # Back up and turn
            return ActionProposal(
                v=-0.2,
                omega=1.0,  # Turn sharply
            )
        
        return None
    
    def reset(self) -> None:
        """Reset recovery attempts."""
        self.recovery_attempts = 0
```

## 6. Action Arbitration

### 6.1 Safety Filtering

```python
class ActionArbitrationSafety:
    """Filters policy decisions through safety constraints."""
    
    def __init__(
        self,
        max_linear: float = 0.3,
        max_angular: float = 1.0,
        collision_margin: float = 0.1,
        geofence: Optional[RegionGoal] = None,
    ):
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.collision_margin = collision_margin
        self.geofence = geofence
        self.staleness_detector = GraphStalenessDetector()
        self.localization_monitor = LocalizationMonitor()
        self.stuck_detector = StuckDetector()
        self.recovery_behavior = RecoveryBehavior()
    
    def filter(
        self,
        decision: PolicyDecision,
        robot_state: RobotState,
        graph_snapshot: GraphSnapshot,
        sensor_fusion: Optional[SensorFusion],
        mission: Mission,
    ) -> SafeCommand:
        """Filter policy decision through safety constraints."""
        
        # 1. Check graph staleness
        staleness = self.staleness_detector.check_staleness(
            graph_snapshot,
            robot_state.time,
        )
        decision = apply_staleness_degradation(decision, staleness)
        
        # 2. Check localization
        localization = self.localization_monitor.update_pose(
            robot_state.pose,
            robot_state.covariance,
            robot_state.time,
        )
        decision = apply_drift_mitigation(decision, localization)
        
        # 3. Check for stuck
        is_stuck, stuck_reason = self.stuck_detector.check_stuck(
            robot_state.pose[:2],
            robot_state.time,
        )
        if is_stuck:
            recovery_action = self.recovery_behavior.get_recovery_action(
                stuck_reason,
                robot_state.pose,
            )
            if recovery_action:
                decision.action_proposal = recovery_action
        
        # 4. Collision checking
        if sensor_fusion:
            decision = self._check_collisions(decision, robot_state, sensor_fusion)
        
        # 5. Geofence checking
        if self.geofence:
            decision = self._check_geofence(decision, robot_state, mission)
        
        # 6. Rate limiting
        decision = self._apply_rate_limiting(decision, robot_state)
        
        # 7. Hard limits
        decision.action_proposal.v = np.clip(
            decision.action_proposal.v,
            -self.max_linear,
            self.max_linear,
        )
        decision.action_proposal.omega = np.clip(
            decision.action_proposal.omega,
            -self.max_angular,
            self.max_angular,
        )
        
        # Build safe command
        return SafeCommand(
            cmd=(decision.action_proposal.v, decision.action_proposal.omega),
            safety_flags={
                "clamped": False,  # Set if limits applied
                "slowed": staleness.staleness_level != "fresh",
                "stop": staleness.recommended_action in ["hold", "estop"],
            },
            latency_ms=0.0,  # Measured during execution
        )
    
    def _check_collisions(
        self,
        decision: PolicyDecision,
        robot_state: RobotState,
        sensor_fusion: SensorFusion,
    ) -> PolicyDecision:
        """Check for collisions and adjust action."""
        # Predict next position
        dt = 0.1  # Assume 10Hz
        next_x = robot_state.pose[0] + decision.action_proposal.v * np.cos(robot_state.pose[2]) * dt
        next_y = robot_state.pose[1] + decision.action_proposal.v * np.sin(robot_state.pose[2]) * dt
        
        # Check obstacle map
        # (simplified - would need proper coordinate transformation)
        if sensor_fusion.confidence > 0.5:
            # If collision predicted, reduce speed
            decision.action_proposal.v *= 0.5
            decision.confidence *= 0.7
        
        return decision
    
    def _check_geofence(
        self,
        decision: PolicyDecision,
        robot_state: RobotState,
        mission: Mission,
    ) -> PolicyDecision:
        """Check geofence constraints."""
        is_valid, reason = mission.constraints.is_valid_position(robot_state.pose[:2])
        if not is_valid:
            # Stop or reverse
            decision.action_proposal.v = 0.0
            decision.action_proposal.omega = 0.0
            decision.reason = f"geofence_violation: {reason}"
        
        return decision
    
    def _apply_rate_limiting(
        self,
        decision: PolicyDecision,
        robot_state: RobotState,
    ) -> PolicyDecision:
        """Apply rate limiting to prevent sudden changes."""
        if robot_state.previous_action is not None:
            prev_v, prev_omega = robot_state.previous_action
            max_dv = 0.1  # m/s per step
            max_domega = 0.5  # rad/s per step
            
            dv = decision.action_proposal.v - prev_v
            domega = decision.action_proposal.omega - prev_omega
            
            if abs(dv) > max_dv:
                decision.action_proposal.v = prev_v + np.sign(dv) * max_dv
            if abs(domega) > max_domega:
                decision.action_proposal.omega = prev_omega + np.sign(domega) * max_domega
        
        return decision
```

## 7. Summary

**Key Safety Mechanisms**:

1. **Graph Staleness**: Detection, degradation, hold/estop
2. **Localization Drift**: Monitoring, mitigation
3. **Sensor Fusion**: Multi-sensor integration, failure detection
4. **Recovery Behaviors**: Stuck detection, recovery actions
5. **Action Arbitration**: Collision checking, geofence, rate limiting, hard limits

**Safety Modes**:
- NORMAL: Full operation
- CONSERVATIVE: Reduced speed, tighter constraints
- HOLD: Stop/hover
- E_STOP: Emergency stop

**Implementation Phases**:
- **Milestone A**: Basic limits, collision checking
- **Milestone B**: Staleness detection, localization monitoring
- **Milestone C**: Sensor fusion, recovery behaviors
- **Milestone D**: Advanced failure modes, comprehensive testing

This specification provides comprehensive safety mechanisms for the policy system.

