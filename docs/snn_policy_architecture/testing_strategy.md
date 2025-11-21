# Testing Strategy for SNN Policy System

This document specifies comprehensive testing strategies for the SNN Policy Service.

## 1. Overview

Testing covers unit tests, integration tests, property tests, latency tests, and scenario-based tests.

**Test Organization**:
```
tests/
├── unit/
│   ├── test_topology_service.py
│   ├── test_feature_service.py
│   ├── test_spike_encoding.py
│   ├── test_snn_network.py
│   ├── test_decision_decoding.py
│   ├── test_graph_navigation.py
│   ├── test_mission_representation.py
│   └── test_safety.py
├── integration/
│   ├── test_policy_pipeline.py
│   ├── test_ros_integration.py
│   └── test_end_to_end.py
├── property/
│   ├── test_feature_invariants.py
│   └── test_coordinate_transforms.py
├── latency/
│   ├── test_timing_budgets.py
│   └── test_performance_regression.py
└── scenarios/
    ├── test_graph_connectivity.py
    ├── test_goal_unreachability.py
    ├── test_sensor_failure.py
    └── test_recovery_behaviors.py
```

## 2. Unit Tests

### 2.1 Topology Service

```python
def test_topology_service_wraps_graph():
    """Test TS correctly wraps TopologicalGraph."""
    graph = TopologicalGraph(positions)
    ts = TopologyService(graph)
    snapshot = ts.get_graph_snapshot(0.0)
    assert len(snapshot.V) == graph.num_nodes()
    assert len(snapshot.E) == graph.num_edges()

def test_graph_snapshot_metadata():
    """Test snapshot metadata is correct."""
    ts = TopologyService(graph)
    snapshot = ts.get_graph_snapshot(10.0)
    assert snapshot.meta.stamp == 10.0
    assert snapshot.meta.frame_id == "map"
    assert snapshot.meta.epoch_id >= 0

def test_staleness_detection():
    """Test staleness detection."""
    ts = TopologyService(graph)
    snapshot = ts.get_graph_snapshot(0.0)
    # Update time
    snapshot2 = ts.get_graph_snapshot(6.0)  # 6 seconds later
    assert snapshot2.meta.staleness_warning
```

### 2.2 Feature Service

```python
def test_goal_ego_features():
    """Test goal-relative feature computation."""
    robot_pose = (0.0, 0.0, 0.0)  # (x, y, yaw)
    goal_position = (1.0, 0.0)  # 1m ahead
    features = compute_goal_ego(robot_pose, goal_position)
    assert len(features) == 3
    assert features[0] > 0  # Distance > 0
    assert abs(features[1] - 1.0) < 0.01  # cos(0) = 1
    assert abs(features[2] - 0.0) < 0.01  # sin(0) = 0

def test_neighbor_features():
    """Test neighbor feature computation."""
    robot_pose = (0.5, 0.5, 0.0)
    graph_snapshot = create_test_graph_snapshot()
    neighbors = compute_neighbor_features(robot_pose, graph_snapshot, k=8)
    assert len(neighbors) == 8
    for neighbor in neighbors:
        assert len(neighbor) == 4  # [cos, sin, dist, on_path]

def test_feature_normalization():
    """Test feature normalization."""
    normalizer = ZScoreNormalizer.fit(feature_matrix)
    normalized = normalizer.normalize(features)
    assert np.all(np.isfinite(normalized))
    assert np.std(normalized) < 2.0  # Should be roughly unit variance
```

### 2.3 Spike Encoding

```python
def test_rate_encoding():
    """Test rate coding."""
    features = torch.tensor([[0.5, 0.8, 0.2]])
    spikes = encode_rate(features, num_steps=10)
    assert spikes.shape == (10, 1, 3)
    assert spikes.dtype == torch.bool
    # Higher values should spike more
    assert spikes[:, 0, 1].sum() > spikes[:, 0, 2].sum()

def test_latency_encoding():
    """Test latency coding."""
    features = torch.tensor([[0.9, 0.1]])
    spikes = encode_latency(features, num_steps=10)
    # Higher value should spike earlier
    first_spike_high = (spikes[:, 0, 0] == 1).nonzero()[0].item()
    first_spike_low = (spikes[:, 0, 1] == 1).nonzero()[0].item()
    assert first_spike_high < first_spike_low
```

### 2.4 SNN Network

```python
def test_snn_forward_step():
    """Test SNN single-step forward."""
    model = PolicySNN(feature_dim=10, hidden_dim=32, output_dim=2)
    input_spikes = torch.zeros(1, 10)
    membrane = model.init_state(1, torch.device("cpu"))
    action, next_membrane = model.forward_step(input_spikes, membrane)
    assert action.shape == (1, 2)
    assert next_membrane.shape == (1, 32)
    assert torch.all(torch.abs(action) <= 1.0)  # tanh output

def test_snn_sequence():
    """Test SNN multi-step forward."""
    model = PolicySNN(feature_dim=10, hidden_dim=32, output_dim=2)
    spike_train = torch.zeros(5, 1, 10)
    actions, final_membrane = model.forward_sequence(spike_train)
    assert actions.shape == (5, 1, 2)
    assert final_membrane.shape == (1, 32)
```

### 2.5 Decision Decoding

```python
def test_decision_decoding():
    """Test decision decoding."""
    decoder = DecisionDecoder(max_linear=0.3, max_angular=1.0)
    snn_output = torch.tensor([[0.5, -0.3]])  # Normalized actions
    decision = decoder.decode(snn_output, features, context, mission)
    assert decision.action_proposal.v == 0.15  # 0.5 * 0.3
    assert decision.action_proposal.omega == -0.3  # -0.3 * 1.0
    assert 0.0 <= decision.confidence <= 1.0
```

## 3. Integration Tests

### 3.1 Policy Pipeline

```python
def test_full_policy_pipeline():
    """Test end-to-end policy pipeline."""
    # Setup
    place_ctrl = PlaceCellController(env, config)
    ts = TopologyService(place_ctrl.get_graph())
    sfs = SpatialFeatureService(ts)
    sps = SpikingPolicyService(sfs)
    aas = ActionArbitrationSafety()
    
    # Run simulation
    for step in range(100):
        position = agent.step(dt)
        place_ctrl.step(position, dt)
        ts.update_from_controller(place_ctrl)
        
        robot_state = RobotState(pose=(*position, 0.0), time=step*dt)
        mission = Mission(goal=PointGoal((0.9, 0.9)))
        
        features, context = sfs.build_features(robot_state, mission)
        decision = sps.decide(features, context, dt)
        safe_cmd = aas.filter(decision, robot_state, ts.get_graph_snapshot(robot_state.time), None, mission)
        
        # Verify outputs
        assert -0.3 <= safe_cmd.cmd[0] <= 0.3
        assert -1.0 <= safe_cmd.cmd[1] <= 1.0
```

### 3.2 ROS Integration

```python
def test_ros_node_integration():
    """Test ROS 2 node integration."""
    # Create test node
    node = PolicyNode()
    
    # Publish odometry
    odom_msg = Odometry()
    odom_msg.pose.pose.position.x = 0.5
    odom_msg.pose.pose.position.y = 0.5
    node._odom_callback(odom_msg)
    
    # Publish mission
    mission_msg = Mission()
    mission_msg.goal.type = "point"
    mission_msg.goal.value.position = [0.9, 0.9]
    node._mission_callback(mission_msg)
    
    # Verify cmd_vel is published
    # (would need ROS test framework)
```

## 4. Property Tests

### 4.1 Feature Invariants

```python
def test_rotation_invariance():
    """Test features rotate correctly with world rotation."""
    robot_pose1 = (0.0, 0.0, 0.0)
    robot_pose2 = (0.0, 0.0, np.pi/2)  # Rotated 90 degrees
    goal_position = (1.0, 0.0)
    
    features1 = compute_goal_ego(robot_pose1, goal_position)
    features2 = compute_goal_ego(robot_pose2, goal_position)
    
    # Bearing should differ by 90 degrees
    bearing1 = np.arctan2(features1[2], features1[1])
    bearing2 = np.arctan2(features2[2], features2[1])
    assert abs(bearing2 - bearing1 - np.pi/2) < 0.01

def test_translation_invariance():
    """Test features are translation-invariant for relative positions."""
    # Features should only depend on relative positions, not absolute
    robot_pose1 = (0.0, 0.0, 0.0)
    robot_pose2 = (1.0, 1.0, 0.0)  # Translated
    goal1 = (1.0, 0.0)
    goal2 = (2.0, 1.0)  # Same relative position
    
    features1 = compute_goal_ego(robot_pose1, goal1)
    features2 = compute_goal_ego(robot_pose2, goal2)
    
    # Should be identical (same relative position)
    assert np.allclose(features1, features2)
```

### 4.2 Coordinate Transforms

```python
def test_egocentric_transform():
    """Test allocentric to egocentric transform."""
    # World frame: goal at (1, 0) relative to origin
    # Robot at (0, 0) facing +x: goal should be (1, 0) in ego frame
    # Robot at (0, 0) facing +y: goal should be (0, 1) in ego frame
    
    robot_pose1 = (0.0, 0.0, 0.0)  # Facing +x
    robot_pose2 = (0.0, 0.0, np.pi/2)  # Facing +y
    goal = (1.0, 0.0)
    
    features1 = compute_goal_ego(robot_pose1, goal)
    features2 = compute_goal_ego(robot_pose2, goal)
    
    # In ego frame 1: goal is ahead (bearing = 0)
    # In ego frame 2: goal is to the left (bearing = -π/2)
    bearing1 = np.arctan2(features1[2], features1[1])
    bearing2 = np.arctan2(features2[2], features2[1])
    assert abs(bearing1 - 0.0) < 0.01
    assert abs(bearing2 - (-np.pi/2)) < 0.01
```

## 5. Latency Tests

### 5.1 Timing Budgets

```python
def test_sfs_timing_budget():
    """Test SFS meets timing budget."""
    sfs = SpatialFeatureService(ts)
    robot_state = RobotState(...)
    mission = Mission(...)
    
    times = []
    for _ in range(100):
        start = time.perf_counter()
        features, context = sfs.build_features(robot_state, mission)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    
    p50 = np.percentile(times, 50)
    p99 = np.percentile(times, 99)
    
    assert p50 < 1.0, f"SFS p50: {p50:.2f}ms (target: <1.0ms)"
    assert p99 < 2.0, f"SFS p99: {p99:.2f}ms (target: <2.0ms)"

def test_sps_timing_budget():
    """Test SPS meets timing budget."""
    sps = SpikingPolicyService(sfs)
    features = FeatureVector(...)
    
    times = []
    for _ in range(100):
        start = time.perf_counter()
        decision = sps.decide(features, context, dt)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    p50 = np.percentile(times, 50)
    p99 = np.percentile(times, 99)
    
    assert p50 < 2.0, f"SPS p50: {p50:.2f}ms (target: <2.0ms)"
    assert p99 < 5.0, f"SPS p99: {p99:.2f}ms (target: <5.0ms)"
```

### 5.2 Performance Regression

```python
def test_performance_regression():
    """Test for performance regressions."""
    baseline_times = load_baseline_times("baseline_performance.json")
    current_times = measure_current_performance()
    
    for component in ["sfs", "sps", "aas"]:
        baseline_p99 = baseline_times[component]["p99"]
        current_p99 = current_times[component]["p99"]
        
        regression_threshold = baseline_p99 * 1.2  # 20% regression allowed
        assert current_p99 < regression_threshold, (
            f"{component} p99 regressed: {current_p99:.2f}ms > {regression_threshold:.2f}ms"
        )
```

## 6. Scenario Tests

### 6.1 Graph Connectivity

```python
def test_disconnected_graph():
    """Test handling of disconnected graph."""
    # Create disconnected graph (two components)
    graph = create_disconnected_graph()
    gns = GraphNavigationService(graph)
    
    # Try to find path between components
    path = gns.find_path(0, 5)  # Nodes in different components
    assert path is None
    
    # Should fall back to reactive control
    waypoint = gns.select_next_waypoint(current_pose, goal)
    assert waypoint is None or waypoint.node_id in same_component

def test_empty_graph():
    """Test handling of empty graph."""
    graph = GraphSnapshot(V=[], E=[], meta=...)
    gns = GraphNavigationService(graph)
    waypoint = gns.select_next_waypoint(current_pose, goal)
    assert waypoint is None
```

### 6.2 Goal Unreachability

```python
def test_goal_in_no_fly_zone():
    """Test goal in no-fly zone."""
    mission = Mission(
        goal=PointGoal((0.5, 0.5)),
        constraints=MissionConstraints(
            no_fly_zones=[RegionGoal(region_type="circle", center=(0.5, 0.5), radius=0.2)],
        ),
    )
    
    is_valid, error = mission.validate()
    assert not is_valid
    assert "no_fly_zone" in error.lower()

def test_goal_outside_geofence():
    """Test goal outside geofence."""
    mission = Mission(
        goal=PointGoal((10.0, 10.0)),  # Outside arena
        constraints=MissionConstraints(
            geofence=RegionGoal(region_type="rectangle", center=(0.5, 0.5), width=1.0, height=1.0),
        ),
    )
    
    is_valid, error = mission.validate()
    assert not is_valid
```

### 6.3 Sensor Failure

```python
def test_sensor_failure_handling():
    """Test handling of sensor failures."""
    sensor_monitor = SensorHealthMonitor()
    
    # Simulate sensor failure (no updates)
    sensor_monitor.update_sensor("depth", None, 0.0)
    time.sleep(1.1)  # Exceed timeout
    
    health = sensor_monitor.check_health(1.1)
    assert not health["depth"]
    
    # Policy should degrade but continue
    decision = sps.decide(features, context, dt)
    # Safety features should be conservative (assume obstacles)
    assert decision.confidence < 1.0
```

### 6.4 Recovery Behaviors

```python
def test_stuck_recovery():
    """Test recovery from stuck situation."""
    stuck_detector = StuckDetector()
    recovery = RecoveryBehavior()
    
    # Simulate stuck (no movement)
    for t in range(10):
        stuck_detector.check_stuck((0.5, 0.5), t * 0.1)
    
    is_stuck, reason = stuck_detector.check_stuck((0.5, 0.5), 1.0)
    assert is_stuck
    
    # Get recovery action
    recovery_action = recovery.get_recovery_action(reason, (0.5, 0.5, 0.0))
    assert recovery_action is not None
    assert recovery_action.v < 0  # Should back up
    assert abs(recovery_action.omega) > 0  # Should rotate
```

## 7. Replay Tests

### 7.1 Deterministic Replay

```python
def test_deterministic_replay():
    """Test deterministic replay produces identical decisions."""
    # Record run
    decisions1 = []
    for step in range(100):
        decision = sps.decide(features[step], context[step], dt)
        decisions1.append(decision)
    
    # Replay with same inputs
    sps.reset()
    decisions2 = []
    for step in range(100):
        decision = sps.decide(features[step], context[step], dt)
        decisions2.append(decision)
    
    # Should be identical
    for d1, d2 in zip(decisions1, decisions2):
        assert d1.action_proposal.v == d2.action_proposal.v
        assert d1.action_proposal.omega == d2.action_proposal.omega
```

## 8. Test Execution

### 8.1 Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -e .[dev]
      - run: pytest tests/unit/ -v
      - run: pytest tests/integration/ -v
      - run: pytest tests/property/ -v
      - run: pytest tests/latency/ -v --benchmark-only
      - run: pytest tests/scenarios/ -v
```

### 8.2 Test Coverage

Target: >80% code coverage

```bash
pytest --cov=src/hippocampus_core/policy --cov-report=html
```

## 9. Summary

**Test Categories**:

1. **Unit Tests**: Component-level functionality
2. **Integration Tests**: End-to-end pipeline
3. **Property Tests**: Invariants and transforms
4. **Latency Tests**: Timing budgets and regression
5. **Scenario Tests**: Failure modes and edge cases
6. **Replay Tests**: Determinism and reproducibility

**Coverage Goals**:
- Unit tests: >90% coverage
- Integration tests: All main workflows
- Scenario tests: All failure modes
- Latency tests: All components

This specification provides comprehensive testing coverage for the policy system.

