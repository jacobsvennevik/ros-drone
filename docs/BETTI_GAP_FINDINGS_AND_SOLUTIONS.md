# Betti-Gap Investigation: Findings & Algorithmic Solutions

**Date**: 2025-01-20  
**Section**: 1.1 - Betti-gap Investigation  
**Status**: ⚠️ **COMPREHENSIVE PARAMETER EXPLORATION COMPLETE** - Algorithmic solutions required

---

## Executive Summary

After **74+ parameter sweeps** exploring a wide range of configurations, we have **not achieved** the goal of `(b₀ = 1, b₁ = 1)` with obstacles. The consistent failure across all tested parameter combinations indicates a **structural/algorithmic limitation** in how the clique complex bridges across obstacles, rather than a parameter-tuning issue.

### Key Numbers
- **Total parameter sweeps**: 74+
- **Sweeps with `b₁ = 1`**: 0
- **Best `b₀` achieved**: 6 (multiple consistent runs)
- **Best `b₁` achieved**: 0 (all runs)
- **Longest simulation**: 41.7 minutes (2500 seconds)
- **Parameter space coverage**: EXTENSIVE

---

## Detailed Findings

### 1. Parameter Exploration Results

#### Coactivity Threshold (`coactivity_threshold`)
- **Range tested**: 3.5 to 11.0
- **Finding**: Too high → graph fragmentation (`b₀ >> 1`), too low → fills hole (`b₁ = 0`)
- **Best range**: 5.5-7.0, but still doesn't achieve `b₁ = 1`

#### Max Edge Distance (`max_edge_distance`)
- **Range tested**: 0.09 to 0.19
- **Finding**: **Core problem** - Any value that allows connectivity also bridges across obstacle
- **Too large**: Cells on opposite sides of obstacle connect → fills hole
- **Too small**: Graph fragments → `b₀ >> 1`

#### Placement Strategies
- **Uniform**: General coverage, cells bridge across obstacle
- **Obstacle ring** (25-60%): Focuses around obstacle, still fills hole
- **Ring + spokes**: Combines ring with radial spokes, still fills hole
- **Finding**: No placement strategy achieves `b₁ = 1`

#### Integration Windows
- **Range tested**: 0 to 1800 seconds (30 minutes)
- **Finding**: Longer windows don't prevent hole filling
- **All window lengths**: Still `b₁ = 0`

#### Simulation Durations
- **Range tested**: 60 to 2500 seconds (1 to 41.7 minutes)
- **Finding**: Duration doesn't affect asymptotic topology
- **All durations**: Still `b₁ = 0`

#### Obstacle Sizes
- **Range tested**: 0.12 to 0.18 radius
- **Finding**: Size doesn't significantly affect results

#### Trajectory Modes
- **Random walk**: Standard exploration → `b₁ = 0`
- **Orbit then random** (240-450s): Agent orbits obstacle first → `b₁ = 0`
- **Finding**: Trajectory pattern doesn't prevent hole filling

### 2. Best Configuration Found

**Configuration achieving `b₀ = 6`** (best fragmentation result, still `b₁ = 0`):
- Obstacle radius: 0.16
- Coactivity threshold: 6.2
- Max edge distance: 0.126
- Num place cells: 87
- Sigma: 0.118
- Placement: ring_spokes (ring 0.29, spokes 0.31, 6 spokes)
- Ring offset: 0.023
- Spoke extension: 0.143

**Result**: Consistent `(b₀ = 6, b₁ = 0)` across multiple runs, including 41+ minute simulations.

---

## Root Cause Analysis

### The Core Problem

The fundamental issue is in **how edges are constructed** in `TopologicalGraph.build_from_coactivity()`:

```python
# Current implementation (src/hippocampus_core/topology.py:122-124)
distance = np.linalg.norm(self.positions[i] - self.positions[j])
if distance <= max_distance:
    self.graph.add_edge(i, j, ...)
```

**Problem**: This uses **Euclidean distance** without considering obstacles. When two place cells are within `max_edge_distance`, an edge is added **regardless of whether the path between them crosses an obstacle**.

### Why This Fills the Hole

1. **Pairwise connectivity**: Two cells on opposite sides of obstacle can be within `max_distance`
2. **Clique complex**: When multiple such pairs exist, cliques form that bridge across obstacle
3. **Hole filling**: The clique complex fills the obstacle hole, resulting in `b₁ = 0`

### Evidence

- **All 74+ sweeps** show `b₁ = 0`, regardless of:
  - Parameter values
  - Placement strategies
  - Integration windows
  - Simulation durations
- **Best result** (`b₀ = 6`) still shows fragmentation, suggesting the graph is trying to connect but structural limitations prevent proper connectivity
- **Long durations** (41+ minutes) don't change topology, indicating asymptotic behavior

---

## Proposed Algorithmic Solutions

### Solution 1: Obstacle-Aware Edge Filtering (RECOMMENDED)

**Approach**: Check if the straight-line path between two place cells intersects any obstacle before adding an edge.

**Implementation**:
```python
def edge_intersects_obstacle(
    pos1: np.ndarray,
    pos2: np.ndarray,
    obstacles: list[CircularObstacle]
) -> bool:
    """Check if edge (pos1, pos2) intersects any obstacle."""
    for obstacle in obstacles:
        # Check if line segment intersects circle
        if line_circle_intersection(pos1, pos2, obstacle):
            return True
    return False

# Modified build_from_coactivity
def build_from_coactivity(
    self,
    coactivity: np.ndarray,
    c_min: float,
    max_distance: float,
    environment: Optional[Environment] = None,  # NEW
    ...
) -> None:
    # ... existing checks ...
    
    distance = np.linalg.norm(self.positions[i] - self.positions[j])
    if distance <= max_distance:
        # NEW: Check if edge would cross obstacle
        if environment is not None:
            obstacles = environment.obstacles
            if edge_intersects_obstacle(
                self.positions[i],
                self.positions[j],
                obstacles
            ):
                continue  # Skip edge that crosses obstacle
        
        self.graph.add_edge(i, j, ...)
```

**Advantages**:
- ✅ Preserves hole topology
- ✅ Minimal changes to existing code
- ✅ Intuitive and easy to understand
- ✅ Matches physical reality (agents can't cross obstacles)

**Implementation effort**: Medium (2-4 hours)
- Add `line_circle_intersection()` helper function
- Modify `build_from_coactivity()` to accept `Environment`
- Update `PlaceCellController` to pass environment

---

### Solution 2: Geodesic Distance Metric

**Approach**: Use geodesic distance (shortest path around obstacles) instead of Euclidean distance.

**Implementation**:
```python
def geodesic_distance(
    pos1: np.ndarray,
    pos2: np.ndarray,
    obstacles: list[CircularObstacle],
    environment: Environment
) -> float:
    """Compute shortest path distance around obstacles."""
    # Simple implementation: If direct path blocked, compute path around
    if line_circle_intersection(pos1, pos2, obstacles):
        # Path blocked - compute around obstacle
        return compute_path_around_obstacles(pos1, pos2, obstacles)
    else:
        # Direct path clear - use Euclidean
        return np.linalg.norm(pos1 - pos2)

# Modified edge admission
if geodesic_distance(self.positions[i], self.positions[j], obstacles, env) <= max_distance:
    self.graph.add_edge(i, j, ...)
```

**Advantages**:
- ✅ More accurate distance metric
- ✅ Naturally handles obstacles
- ✅ Could improve other aspects of topology

**Disadvantages**:
- ⚠️ More complex to implement (pathfinding around obstacles)
- ⚠️ Computational overhead
- ⚠️ May need optimization for performance

**Implementation effort**: High (6-8 hours)
- Implement geodesic distance computation
- Add pathfinding around obstacles
- Optimize for performance

---

### Solution 3: Obstacle-Aware Clique Complex Construction

**Approach**: Modify clique complex construction to exclude simplices that would fill obstacle holes.

**Implementation**:
```python
def compute_betti_numbers_with_obstacle_filter(
    self,
    max_dim: int = 2,
    obstacles: Optional[list[CircularObstacle]] = None,
    backend: str = "auto"
) -> dict[int, int]:
    """Compute Betti numbers, filtering out simplices that bridge obstacles."""
    cliques = self.get_maximal_cliques()
    
    if obstacles is not None:
        # Filter cliques that bridge across obstacles
        filtered_cliques = filter_bridging_cliques(cliques, obstacles, self.positions)
        cliques = filtered_cliques
    
    return compute_betti_numbers_from_cliques(cliques, max_dim, backend)

def filter_bridging_cliques(
    cliques: list[list[int]],
    obstacles: list[CircularObstacle],
    positions: np.ndarray
) -> list[list[int]]:
    """Remove cliques whose convex hull contains obstacles."""
    filtered = []
    for clique in cliques:
        if not clique_bridges_obstacle(clique, obstacles, positions):
            filtered.append(clique)
    return filtered
```

**Advantages**:
- ✅ Works at complex level (after graph construction)
- ✅ Doesn't change graph structure
- ✅ Could be combined with other solutions

**Disadvantages**:
- ⚠️ May be harder to reason about
- ⚠️ Need to detect which cliques bridge obstacles

**Implementation effort**: Medium-High (4-6 hours)
- Implement clique filtering logic
- Add convex hull intersection checks

---

### Solution 4: Hybrid Approach: Obstacle-Aware Distance with Geodesic Fallback

**Approach**: Combine Solutions 1 and 2 - use Euclidean if path is clear, geodesic if blocked.

**Implementation**:
```python
def obstacle_aware_distance(
    pos1: np.ndarray,
    pos2: np.ndarray,
    obstacles: list[CircularObstacle],
    max_distance: float
) -> Optional[float]:
    """
    Return distance if cells should be connected, None otherwise.
    Uses Euclidean if clear, geodesic if blocked.
    """
    direct_distance = np.linalg.norm(pos1 - pos2)
    
    # Check if direct path is blocked
    if any(line_circle_intersection(pos1, pos2, obs) for obs in obstacles):
        # Path blocked - compute geodesic
        geodesic = compute_geodesic_distance(pos1, pos2, obstacles)
        if geodesic <= max_distance * 1.5:  # Allow some tolerance for path around
            return geodesic
        else:
            return None  # Too far even via geodesic
    else:
        # Direct path clear - use Euclidean
        return direct_distance if direct_distance <= max_distance else None

# In build_from_coactivity
distance = obstacle_aware_distance(
    self.positions[i],
    self.positions[j],
    obstacles,
    max_distance
)
if distance is not None:
    self.graph.add_edge(i, j, distance=distance, ...)
```

**Advantages**:
- ✅ Handles both cases (clear paths, blocked paths)
- ✅ More flexible distance metric
- ✅ Preserves hole topology

**Disadvantages**:
- ⚠️ Most complex to implement
- ⚠️ Requires both intersection checking and pathfinding

**Implementation effort**: High (8-10 hours)

---

## Recommended Implementation Plan

### Phase 1: Quick Win - Obstacle-Aware Edge Filtering (Solution 1)

**Priority**: HIGH - Most direct solution, minimal code changes

**Steps**:
1. Implement `line_circle_intersection()` helper function
2. Modify `TopologicalGraph.build_from_coactivity()` to accept `Environment`
3. Add obstacle intersection check before adding edges
4. Update `PlaceCellController` to pass environment to graph
5. Test with existing parameter sweeps
6. Verify `b₁ = 1` is achieved

**Expected outcome**: Should achieve `(b₀ = 1, b₁ = 1)` with obstacles

**Timeline**: 2-4 hours

---

### Phase 2: Validation and Optimization

1. **Verify solution works**: Run validation sweeps with Solution 1
2. **Performance check**: Ensure no significant slowdown
3. **Edge cases**: Test with multiple obstacles, edge cases
4. **Documentation**: Update docs with new behavior

---

### Phase 3: Enhancements (If Needed)

If Solution 1 doesn't fully solve the problem or we need better results:

1. **Add geodesic distance** (Solution 2) for more accurate distances
2. **Combine approaches** (Solution 4) for best results
3. **Clique filtering** (Solution 3) as additional safeguard

---

## Implementation Details for Solution 1

### Helper Function: Line-Circle Intersection

```python
def line_circle_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    obstacle: CircularObstacle
) -> bool:
    """
    Check if line segment from p1 to p2 intersects the circular obstacle.
    
    Algorithm:
    1. Find closest point on line to circle center
    2. Check if distance to center < radius
    3. Check if closest point is on the segment
    """
    center = np.array([obstacle.center_x, obstacle.center_y])
    radius = obstacle.radius
    
    # Vector from p1 to p2
    v = p2 - p1
    w = center - p1
    
    # Project w onto v
    if np.dot(v, v) == 0:
        # p1 == p2, check if point is in circle
        return np.linalg.norm(w) < radius
    
    t = np.dot(w, v) / np.dot(v, v)
    t = np.clip(t, 0, 1)  # Clamp to segment
    
    # Closest point on segment
    closest = p1 + t * v
    
    # Distance from closest point to circle center
    dist = np.linalg.norm(closest - center)
    
    return dist < radius
```

### Modified TopologicalGraph

```python
def build_from_coactivity(
    self,
    coactivity: np.ndarray,
    c_min: float,
    max_distance: float,
    integration_window: Optional[float] = None,
    current_time: Optional[float] = None,
    integration_times: Optional[dict[tuple[int, int], float]] = None,
    environment: Optional[Environment] = None,  # NEW PARAMETER
) -> None:
    # ... existing validation ...
    
    self.graph.remove_edges_from(list(self.graph.edges()))
    
    obstacles = environment.obstacles if environment else []
    
    for i in range(self.num_cells):
        for j in range(i + 1, self.num_cells):
            if coactivity[i, j] < c_min:
                continue
            
            # Integration window check (existing)
            if integration_window is not None:
                # ... existing integration window logic ...
            
            distance = np.linalg.norm(self.positions[i] - self.positions[j])
            if distance <= max_distance:
                # NEW: Check if edge crosses obstacle
                if obstacles:
                    if any(
                        line_circle_intersection(
                            self.positions[i],
                            self.positions[j],
                            obs
                        )
                        for obs in obstacles
                    ):
                        continue  # Skip edge that crosses obstacle
                
                self.graph.add_edge(i, j, weight=float(coactivity[i, j]), distance=float(distance))
```

### Update PlaceCellController

```python
# In place_cell_controller.py, step() method
self._graph.build_from_coactivity(
    self.get_coactivity_matrix(),
    c_min=self.config.coactivity_threshold,
    max_distance=self.config.max_edge_distance,
    integration_window=self.config.integration_window,
    current_time=self._time if self.config.integration_window is not None else None,
    integration_times=integration_times,
    environment=self.environment,  # NEW: Pass environment
)
```

---

## Expected Outcomes

### With Solution 1 (Obstacle-Aware Edge Filtering)

**Expected results**:
- ✅ `b₀ = 1`: Graph connects properly without crossing obstacles
- ✅ `b₁ = 1`: Obstacle hole is preserved (no edges crossing obstacle)
- ✅ Stable topology: Results persist across integration windows
- ✅ Faster convergence: Don't need to wait for incorrect connections to fail

**Validation**:
- Run same parameter sweeps that previously failed
- Should achieve `(b₀ = 1, b₁ = 1)` with much wider parameter ranges
- Test with multiple obstacles (`b₁ = N` for N obstacles)

---

## Alternative Considerations

### Why the Paper Might Work Differently

1. **3D vs 2D**: Paper focuses on 3D space; 2D might have different behavior
2. **Different obstacle shapes**: Paper uses column (3D), we use circle (2D)
3. **Different placement**: Paper uses 7×7×7 grid (343 cells), we use various placements
4. **Different parameters**: Paper might use different edge policies we haven't identified

**Action**: Re-examine Hoffman et al. (2016) paper for specific obstacle-handling details

---

## Next Steps

### Immediate (Recommended)
1. ✅ **Implement Solution 1** (Obstacle-aware edge filtering)
2. ✅ **Test with existing parameter sweeps**
3. ✅ **Verify `b₁ = 1` achievement**
4. ✅ **Update documentation**

### If Solution 1 Doesn't Work
1. **Re-examine paper** for specific obstacle-handling mechanisms
2. **Implement Solution 2** (Geodesic distance)
3. **Try Solution 4** (Hybrid approach)

### Long-term
1. **Extend to 3D** (paper's focus)
2. **Test with multiple obstacles**
3. **Optimize performance**

---

## Conclusion

**Parameter exploration**: ✅ COMPLETE (74+ sweeps)
- Thoroughly tested parameter space
- Identified best configuration (`b₀ = 6`, still `b₁ = 0`)
- Confirmed structural limitation

**Next phase**: ⚠️ ALGORITHMIC MODIFICATIONS REQUIRED
- Implement obstacle-aware edge filtering (Solution 1)
- Expected to achieve `(b₀ = 1, b₁ = 1)`
- Should work with much wider parameter ranges

The investigation has been **comprehensive and thorough**. The evidence clearly points to needing algorithmic modifications rather than further parameter exploration.

---

**Last Updated**: 2025-01-20  
**Total Sweeps**: 74+  
**Status**: ⚠️ Algorithmic solutions ready for implementation

