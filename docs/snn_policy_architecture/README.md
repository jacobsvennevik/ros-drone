# SNN Policy Architecture Documentation

This directory contains comprehensive architecture specifications for the SNN Policy Service that chooses where to fly and produces flight commands.

## Overview

The SNN Policy System is a hierarchical architecture that:
1. Uses the existing topological mapping system (`TopologicalGraph`) as a source of spatial knowledge
2. Converts graph + robot state into policy features
3. Uses a Spiking Neural Network (SNN) to make navigation decisions
4. Applies safety constraints and filters
5. Publishes velocity commands to the robot

## Architecture Documents

### Core Components

1. **[Integration Analysis](integration_analysis.md)**
   - How components integrate with existing codebase
   - TS wraps `TopologicalGraph`
   - VI extends `BrainNode`
   - SPS follows `SNNController` interface
   - Data flow and ROS 2 integration

2. **[SNN Specification](snn_specification.md)**
   - Spike encoding schemes (rate, latency, delta)
   - LIF neuron dynamics
   - Network architecture
   - Decision decoding
   - Temporal context
   - Training interface (future)

3. **[Graph Navigation Service](graph_navigation_service.md)**
   - Path planning algorithms (Dijkstra, A*, Greedy)
   - Waypoint selection
   - Handling disconnected graphs
   - Integration with SNN policy
   - Dynamic graph updates

4. **[Feature Engineering](feature_engineering.md)**
   - Feature vector structure
   - Feature computation (goal, neighbors, topology, safety, dynamics)
   - Normalization schemes (Z-score, min-max, unit vector)
   - Temporal context and history buffers
   - Feature selection and ablation

5. **[Mission Representation](mission_representation.md)**
   - Goal types (Point, Node, Region, Sequential, Explore)
   - Mission constraints (no-fly zones, geofence, altitude)
   - Goal validation and reachability
   - Dynamic goal updates
   - ROS 2 integration

6. **[Safety Enhancements](safety_enhancements.md)**
   - Graph staleness detection
   - Localization drift monitoring
   - Sensor fusion and failure detection
   - Recovery behaviors
   - Action arbitration and filtering

7. **[Performance Optimization](performance_optimization.md)**
   - Graph snapshot caching
   - Feature computation caching
   - Vectorized operations
   - Memory management
   - TorchScript compilation
   - Profiling and monitoring

8. **[Testing Strategy](testing_strategy.md)**
   - Unit tests
   - Integration tests
   - Property tests
   - Latency tests
   - Scenario tests (connectivity, unreachability, sensor failure)
   - Replay tests

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Topology Service (TS)                       │
│         Wraps TopologicalGraph                           │
│         Provides GraphSnapshot                           │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│         Spatial Feature Service (SFS)                    │
│         GraphSnapshot + RobotState + Mission             │
│         → FeatureVector                                  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│         Spiking Policy Service (SPS)                     │
│         FeatureVector → SpikeTrain → SNN → Decision      │
│         Implements SNNController interface               │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│    Action Arbitration & Safety (AAS)                     │
│    Filters decisions through safety constraints          │
│    → SafeCommand                                         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│         Vehicle Interface (VI)                           │
│         Extends BrainNode                                │
│         Publishes /cmd_vel                               │
└──────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Milestone A: Heuristic Stub
- TS + SFS + AAS + VI integrated
- SPS implemented as heuristic stub (nearest neighbor toward goal)
- CI with latency & replay tests
- **Status**: Ready to implement

### Milestone B: SNN Runtime
- Swap SPS to SNN runtime wrapper
- Spike encoding/decoding infrastructure
- Temporal context handling
- Diagnostics & confidence outputs
- **Status**: Specified, ready for implementation

### Milestone C: 3D & Navigation
- 3D enablement across services
- Graph Navigation Service (GNS)
- Hierarchical planning
- **Status**: Specified, ready for implementation

### Milestone D: Optimization & Testing
- Performance optimizations
- Telemetry dashboards
- Scenario harness
- Stability under failure modes
- **Status**: Specified, ready for implementation

## Key Design Decisions

1. **Integration over Duplication**: TS wraps `TopologicalGraph` rather than duplicating it
2. **Interface Consistency**: SPS follows `SNNController` interface like existing controllers
3. **Reactive First**: Start with reactive control, add hierarchical planning later
4. **Safety First**: Comprehensive safety mechanisms from the start
5. **Performance Targets**: Hard real-time constraints (<10ms p99)
6. **2D/3D Parity**: Design supports both 2D and 3D from the start

## Data Contracts

### GraphSnapshot
- V: List of nodes (node_id, position, degree, tags)
- E: List of edges (u, v, length, traversable, integrator_value)
- meta: Metadata (epoch_id, frame_id, stamp, staleness_warning)

### FeatureVector
- goal_ego: [distance, cos(θ), sin(θ), ...]
- neighbors_k: k × [cos(θ), sin(θ), distance, on_path]
- topo_ctx: [degree, clustering, path_progress]
- safety: [front, left, right, back]_norm
- dynamics: [prev_v, prev_ω, ...] (optional)

### PolicyDecision
- next_waypoint: Optional[int]
- action_proposal: (v, ω, vz)
- confidence: [0, 1]
- reason: str

### SafeCommand
- cmd: (v, ω, vz)
- safety_flags: {clamped, slowed, stop}
- latency_ms: float

## Performance Targets

- **Control loop**: 10 Hz (100ms period)
- **End-to-end latency**: <5ms p50, <10ms p99
- **Component budgets**:
  - SFS: 1.0 / 2.0 ms (p50 / p99)
  - SPS: 2.0 / 5.0 ms
  - AAS: 0.5 / 1.0 ms
  - VI + overhead: 0.5 / 1.0 ms

## Testing Coverage

- **Unit tests**: >90% code coverage
- **Integration tests**: All main workflows
- **Property tests**: Invariants (rotation, translation)
- **Latency tests**: Timing budgets and regression
- **Scenario tests**: Failure modes (disconnected graph, sensor failure, stuck)
- **Replay tests**: Deterministic replay

## References

- **snnTorch**: https://snntorch.readthedocs.io/
- **snnTorch Tutorials**: https://github.com/snntorch/Spiking-Neural-Networks-Tutorials
- **Existing Implementation**: `src/hippocampus_core/controllers/snntorch_controller.py`
- **Topological Mapping**: `src/hippocampus_core/topology.py`
- **ROS 2 Integration**: `ros2_ws/src/hippocampus_ros2/`

## Next Steps

1. Review and approve architecture documents
2. Create implementation plan for Milestone A
3. Set up project structure (`src/hippocampus_core/policy/`)
4. Begin implementation with TS and SFS
5. Integrate with existing `PlaceCellController`
6. Test end-to-end pipeline

## Questions & Clarifications

For questions about the architecture, see individual documents or refer to:
- Integration points: `integration_analysis.md`
- SNN implementation: `snn_specification.md`
- Feature design: `feature_engineering.md`
- Safety mechanisms: `safety_enhancements.md`

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-27  
**Status**: Architecture specified, ready for implementation

