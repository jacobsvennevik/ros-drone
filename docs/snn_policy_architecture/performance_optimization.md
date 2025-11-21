# Performance Optimization Specification

This document specifies performance optimizations for the SNN Policy Service to meet real-time constraints.

## 1. Overview

**Target Performance**:
- Control loop: 10 Hz (100ms period)
- End-to-end latency: <5ms p50, <10ms p99
- Component budgets:
  - SFS: 1.0 / 2.0 ms (p50 / p99)
  - SPS: 2.0 / 5.0 ms
  - AAS: 0.5 / 1.0 ms
  - VI + overhead: 0.5 / 1.0 ms

## 2. Graph Snapshot Caching

### 2.1 Snapshot Caching

```python
from typing import Optional
from dataclasses import dataclass
import time

@dataclass
class CachedSnapshot:
    """Cached graph snapshot."""
    snapshot: GraphSnapshot
    timestamp: float
    hit_count: int = 0

class CachedTopologyService:
    """Topology service with snapshot caching."""
    
    def __init__(
        self,
        topology_service: TopologyService,
        cache_ttl: float = 0.1,  # seconds
    ):
        self.topology_service = topology_service
        self.cache_ttl = cache_ttl
        self._cache: Optional[CachedSnapshot] = None
    
    def get_graph_snapshot(self, current_time: float) -> GraphSnapshot:
        """Get snapshot with caching."""
        # Check cache
        if self._cache is not None:
            age = current_time - self._cache.timestamp
            if age < self.cache_ttl:
                self._cache.hit_count += 1
                return self._cache.snapshot
        
        # Cache miss: fetch new snapshot
        snapshot = self.topology_service.get_graph_snapshot(current_time)
        self._cache = CachedSnapshot(
            snapshot=snapshot,
            timestamp=current_time,
        )
        return snapshot
    
    def invalidate_cache(self) -> None:
        """Invalidate cache (e.g., on graph update)."""
        self._cache = None
```

### 2.2 Incremental Updates

Only rebuild graph when necessary:

```python
class IncrementalGraphBuilder:
    """Incremental graph construction."""
    
    def __init__(self):
        self._last_build_time = 0.0
        self._build_interval = 1.0  # Rebuild every 1 second
    
    def should_rebuild(self, current_time: float) -> bool:
        """Check if graph should be rebuilt."""
        return (current_time - self._last_build_time) >= self._build_interval
    
    def update(self, current_time: float) -> None:
        """Mark graph as updated."""
        self._last_build_time = current_time
```

## 3. Feature Computation Caching

### 3.1 Feature Caching

```python
class CachedFeatureService:
    """Feature service with computation caching."""
    
    def __init__(
        self,
        feature_service: SpatialFeatureService,
        cache_ttl: float = 0.05,  # 50ms
    ):
        self.feature_service = feature_service
        self.cache_ttl = cache_ttl
        self._cache: Optional[Tuple[FeatureVector, float, str]] = None  # (features, time, cache_key)
    
    def build_features(
        self,
        robot_state: RobotState,
        mission: Mission,
        sensor_data: Optional[SensorData] = None,
    ) -> Tuple[FeatureVector, LocalContext]:
        """Build features with caching."""
        # Create cache key from state
        cache_key = self._create_cache_key(robot_state, mission)
        
        # Check cache
        if self._cache is not None:
            cached_features, cached_time, cached_key = self._cache
            age = robot_state.time - cached_time
            if age < self.cache_ttl and cache_key == cached_key:
                # Return cached features (but still need fresh local_context)
                _, local_context = self.feature_service.build_features(
                    robot_state, mission, sensor_data
                )
                return cached_features, local_context
        
        # Cache miss: compute features
        features, local_context = self.feature_service.build_features(
            robot_state, mission, sensor_data
        )
        
        # Update cache
        self._cache = (features, robot_state.time, cache_key)
        
        return features, local_context
    
    def _create_cache_key(
        self,
        robot_state: RobotState,
        mission: Mission,
    ) -> str:
        """Create cache key from state."""
        # Use quantized position and goal
        pos_quantized = (
            int(robot_state.pose[0] * 10) / 10,  # 10cm resolution
            int(robot_state.pose[1] * 10) / 10,
        )
        goal_key = f"{mission.goal.type}_{mission.goal.value}"
        return f"{pos_quantized}_{goal_key}"
```

### 3.2 Vectorized Operations

Use NumPy vectorization for feature computation:

```python
def compute_neighbor_features_vectorized(
    robot_pose: Tuple[float, float, float],
    node_positions: np.ndarray,  # (N, 2)
    k: int = 8,
) -> np.ndarray:
    """Vectorized neighbor feature computation."""
    robot_pos = np.array([robot_pose[0], robot_pose[1]])
    
    # Compute all distances at once
    distances = np.linalg.norm(node_positions - robot_pos, axis=1)
    
    # Get k nearest indices
    k_nearest_indices = np.argpartition(distances, k)[:k]
    k_nearest_indices = k_nearest_indices[np.argsort(distances[k_nearest_indices])]
    
    # Compute bearings for k nearest
    vecs_to_nodes = node_positions[k_nearest_indices] - robot_pos
    bearings = np.arctan2(vecs_to_nodes[:, 1], vecs_to_nodes[:, 0]) - robot_pose[2]
    bearings = np.arctan2(np.sin(bearings), np.cos(bearings))  # Wrap
    
    # Build feature array
    features = np.zeros((k, 4))
    features[:, 0] = np.cos(bearings)
    features[:, 1] = np.sin(bearings)
    features[:, 2] = distances[k_nearest_indices] / 10.0  # Normalize
    features[:, 3] = 0.0  # on_path (would need path info)
    
    return features
```

## 4. Inference Batching

### 4.1 Batch Processing

For multiple queries, batch SNN inference:

```python
class BatchedSNNInference:
    """Batched SNN inference for efficiency."""
    
    def __init__(self, snn_model: PolicySNN, batch_size: int = 4):
        self.snn_model = snn_model
        self.batch_size = batch_size
        self._batch_buffer: List[torch.Tensor] = []
    
    def infer(
        self,
        spike_input: torch.Tensor,  # (1, feature_dim)
        membrane: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infer with batching."""
        self._batch_buffer.append((spike_input, membrane))
        
        if len(self._batch_buffer) >= self.batch_size:
            return self._flush_batch()
        
        # For single queries, process immediately
        return self.snn_model.forward_step(spike_input, membrane)
    
    def _flush_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process batched inputs."""
        # Stack inputs
        batch_inputs = torch.stack([x[0] for x in self._batch_buffer], dim=0)  # (batch, feature_dim)
        batch_membranes = torch.stack(
            [x[1] if x[1] is not None else torch.zeros(...) for x in self._batch_buffer],
            dim=0
        )
        
        # Batch inference
        batch_outputs, batch_membranes = self.snn_model.forward_step(
            batch_inputs, batch_membranes
        )
        
        # Clear buffer
        self._batch_buffer.clear()
        
        return batch_outputs, batch_membranes
```

**Note**: For real-time control, batching may not be applicable (single queries). Useful for training or offline processing.

## 5. Memory Management

### 5.1 History Buffer Limits

```python
class BoundedTemporalContext:
    """Temporal context with memory limits."""
    
    def __init__(
        self,
        max_history: int = 10,
        max_memory_mb: float = 10.0,
    ):
        self.max_history = max_history
        self.max_memory_mb = max_memory_mb
        self.feature_history: deque = deque(maxlen=max_history)
    
    def estimate_memory(self) -> float:
        """Estimate memory usage in MB."""
        total_bytes = 0
        for fv in self.feature_history:
            total_bytes += fv.dim * 4  # float32 = 4 bytes
        return total_bytes / (1024 * 1024)
    
    def update(self, features: FeatureVector) -> None:
        """Update with memory check."""
        # Check memory before adding
        if self.estimate_memory() > self.max_memory_mb:
            # Remove oldest
            if self.feature_history:
                self.feature_history.popleft()
        
        self.feature_history.append(features)
```

### 5.2 Graph Pruning

For large graphs, prune old/unused nodes:

```python
class GraphPruner:
    """Prunes graph to reduce memory."""
    
    def __init__(
        self,
        max_nodes: int = 1000,
        min_visit_count: int = 5,
    ):
        self.max_nodes = max_nodes
        self.min_visit_count = min_visit_count
    
    def prune(
        self,
        graph_snapshot: GraphSnapshot,
    ) -> GraphSnapshot:
        """Prune graph by removing low-visit nodes."""
        if len(graph_snapshot.V) <= self.max_nodes:
            return graph_snapshot
        
        # Sort nodes by visit count
        nodes_with_visits = [
            (n, graph_snapshot.meta.node_visit_counts.get(n.node_id, 0))
            for n in graph_snapshot.V
        ]
        nodes_with_visits.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top N nodes
        kept_nodes = [n for n, _ in nodes_with_visits[:self.max_nodes]]
        kept_node_ids = {n.node_id for n in kept_nodes}
        
        # Filter edges to only connect kept nodes
        kept_edges = [
            e for e in graph_snapshot.E
            if e.u in kept_node_ids and e.v in kept_node_ids
        ]
        
        # Create pruned snapshot
        return GraphSnapshot(
            V=kept_nodes,
            E=kept_edges,
            meta=graph_snapshot.meta,
        )
```

## 6. TorchScript Optimization

### 6.1 Script Compilation

For deployment, use TorchScript:

```python
def compile_snn_model(model: PolicySNN) -> torch.jit.ScriptModule:
    """Compile SNN model to TorchScript."""
    model.eval()
    
    # Create example input
    example_input = torch.zeros(1, model.feature_dim)
    example_membrane = torch.zeros(1, model.hidden_dim)
    
    # Trace
    traced = torch.jit.trace(
        lambda x, m: model.forward_step(x, m)[0],
        (example_input, example_membrane),
    )
    
    return traced

# Usage
scripted_model = compile_snn_model(snn_model)
scripted_model.save("policy_snn.pt")
```

**Benefits**:
- Faster inference (optimized)
- Lower memory footprint
- No Python overhead

## 7. Profiling & Monitoring

### 7.1 Performance Profiling

```python
import time
from contextlib import contextmanager

class PerformanceProfiler:
    """Profiles component performance."""
    
    def __init__(self):
        self.timings: dict[str, List[float]] = {}
    
    @contextmanager
    def profile(self, component: str):
        """Context manager for profiling."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if component not in self.timings:
                self.timings[component] = []
            self.timings[component].append(elapsed * 1000)  # Convert to ms
    
    def get_stats(self, component: str) -> dict:
        """Get statistics for component."""
        if component not in self.timings:
            return {}
        
        timings = self.timings[component]
        return {
            "mean": np.mean(timings),
            "median": np.median(timings),
            "p99": np.percentile(timings, 99),
            "max": np.max(timings),
            "count": len(timings),
        }
```

### 7.2 Usage

```python
profiler = PerformanceProfiler()

# In control loop
with profiler.profile("sfs"):
    features, context = sfs.build_features(robot_state, mission)

with profiler.profile("sps"):
    decision = sps.decide(features, context, dt)

with profiler.profile("aas"):
    safe_cmd = aas.filter(decision, robot_state, ...)

# Check stats
sfs_stats = profiler.get_stats("sfs")
print(f"SFS: {sfs_stats['mean']:.2f}ms mean, {sfs_stats['p99']:.2f}ms p99")
```

## 8. Optimization Strategies by Phase

### Milestone A
- Basic caching (graph snapshots)
- Vectorized neighbor computation
- Simple profiling

### Milestone B
- Feature caching
- TorchScript compilation
- Memory limits

### Milestone C
- Incremental graph updates
- Graph pruning
- Advanced profiling

### Milestone D
- Full optimization suite
- Performance regression tests
- Real-time validation

## 9. Summary

**Key Optimizations**:

1. **Caching**: Graph snapshots, feature vectors
2. **Vectorization**: NumPy operations for neighbors
3. **Memory Management**: History limits, graph pruning
4. **TorchScript**: Compiled models for deployment
5. **Profiling**: Performance monitoring and validation

**Performance Targets**:
- SFS: <2ms p99
- SPS: <5ms p99
- AAS: <1ms p99
- Total: <10ms p99

This specification provides comprehensive performance optimization strategies for the policy system.

