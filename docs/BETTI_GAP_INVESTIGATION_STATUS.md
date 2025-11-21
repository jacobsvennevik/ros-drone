# Betti-Gap Investigation Status Report

**Date**: 2025-01-20  
**Section**: 1.1 - Betti-gap Investigation  
**Goal**: Achieve stable `(b₀ = 1, b₁ = 1)` regime with obstacles  
**Status**: ⚠️ **COMPREHENSIVE PARAMETER EXPLORATION COMPLETE** - Algorithmic solutions required

---

## Summary

**Total Parameter Sweeps Completed**: **74+ sweeps**  
**Sweeps with `b₁ = 1`**: **0**  
**Sweeps with `b₀ = 1`**: **1** (smoke test, likely short duration)

### Current Best Results

- **Best `b₀` observed**: **6** (achieved consistently across multiple runs with specific configuration)
- **Best `b₁` observed**: 0 (all runs show hole is filled by clique complex)
- **Typical result**: `(b₀ = 7-15, b₁ = 0)` - Fragmented graph, no hole detected
- **Longest simulation**: 41.7 minutes (2500 seconds)

---

## Parameter Ranges Explored

### 1. Coactivity Threshold (`coactivity_threshold`, `c_min`)
- **Range tested**: 4.0 to 10.0
- **Best range**: 5.5-7.0 (still not achieving `b₁ = 1`)
- **Finding**: Too high → fragmentation (`b₀ >> 1`), too low → fills hole (`b₁ = 0`)

### 2. Max Edge Distance (`max_edge_distance`, `d_max`)
- **Range tested**: 0.09 to 0.19
- **Best range**: 0.11-0.14 (still not achieving `b₁ = 1`)
- **Finding**: Too large → bridges across obstacle, too small → fragmentation

### 3. Place Field Size (σ)
- **Range tested**: 0.10 to 0.16
- **Best range**: 0.11-0.125 (still not achieving `b₁ = 1`)

### 4. Number of Place Cells (`num_place_cells`, `N`)
- **Range tested**: 60 to 110
- **Best range**: 80-90 (still not achieving `b₁ = 1`)

### 5. Placement Strategies
- **Uniform**: General coverage, but cells bridge across obstacle
- **Obstacle ring** (ring_fraction: 0.25-0.55): Focuses around obstacle, still fills hole
- **Ring + spokes** (various configurations): Combines ring with radial spokes, still fills hole
- **Finding**: No placement strategy has achieved `b₁ = 1` yet

### 6. Integration Windows (ϖ)
- **Range tested**: 0 to 480 seconds
- **Finding**: Longer windows reduce spurious loops but don't prevent hole filling
- **All window lengths tested**: Still `b₁ = 0`

### 7. Simulation Duration
- **Range tested**: 60 to 2500 seconds (1 to 41.7 minutes)
- **Finding**: Longer durations don't change asymptotic topology
- **All durations tested**: Still `b₁ = 0`
- **41.7-minute test**: Still `b₁ = 0`, confirming structural limitation

### 8. Trajectory Modes
- **Random walk**: Standard exploration
- **Orbit then random**: Agent orbits obstacle first (240-300s), then random walk
- **Finding**: Orbit trajectories don't prevent hole filling

### 9. Obstacle Sizes
- **Range tested**: 0.12 to 0.18 radius
- **Finding**: Size doesn't significantly affect hole detection

---

## Key Findings

### ✅ What Works
1. **Infrastructure is solid**: All parameter sweep tools work correctly
2. **Graph construction works**: Edges form, graphs connect (to some degree)
3. **Betti computation works**: Topology is correctly computed
4. **Placement strategies work**: Different cell placements are implemented

### ❌ What Doesn't Work
1. **Hole detection**: No parameter combination achieves `b₁ = 1`
2. **Graph fragmentation**: Most runs show `b₀ >> 1` (fragmented graphs)
3. **Clique complex filling**: Dense cross-connections fill the obstacle hole

### 🔍 Core Problem

**Fundamental issue**: The clique complex built from pairwise coactivities bridges across the obstacle when place cells are within `max_edge_distance`, regardless of:
- Threshold values
- Edge distances
- Placement strategies
- Integration windows
- Simulation durations
- Trajectory patterns

This suggests the limitation is **structural** (how the complex is built) rather than parameter-tuning.

---

## Representative Parameter Sweeps

### High Sparsity Attempts
| Label | Parameters | Result `(b₀, b₁)` |
|-------|-----------|------------------|
| `extreme_sparse_v1` | threshold=10.0, edge=0.115, N=75 | (15, 0) |
| `sparse_very_high_thresh` | threshold=9.5, edge=0.12, N=80 | (14-17, 0) |
| `extreme_compact_v1` | threshold=9.0, edge=0.105, N=65 | (13, 0) |

### Moderate Settings
| Label | Parameters | Result `(b₀, b₁)` |
|-------|-----------|------------------|
| `systematic_sweep_1` | threshold=6.0, edge=0.13, ring 30% | (10, 0) |
| `systematic_sweep_2` | threshold=6.5, edge=0.14, ring+spokes | (7, 0) |
| `balanced_long_v1` | threshold=5.75, edge=0.132, N=88, 1050s | (9, 0) |

### Ring Placement Focus
| Label | Parameters | Result `(b₀, b₁)` |
|-------|-----------|------------------|
| `balanced_ring_v1` | threshold=5.5, edge=0.12, ring 40% | (14, 0) |
| `ring_only_sparse_v1` | threshold=8.0, edge=0.12, ring 50% | (13, 0) |
| `very_compact_ring_v1` | threshold=6.5, edge=0.11, ring 45% | (18, 0) |

### Long Duration Tests
| Label | Parameters | Result `(b₀, b₁)` |
|-------|-----------|------------------|
| `long_duration_v1` | threshold=5.5, edge=0.135, 900s | (8, 0) |
| `very_long_v1` | threshold=5.5, edge=0.13, 1200s | (8, 0) |
| `orbit_long_v1` | threshold=6.0, edge=0.12, orbit+900s | (15, 0) |

---

## Next Steps

### Option 1: Continue Parameter Exploration
- **Grid search**: Systematic exploration of all parameter combinations
- **Random search**: Sample parameter space randomly
- **Bayesian optimization**: Use optimization algorithms to find optimal parameters
- **Effort**: High, success probability: Unknown

### Option 2: Algorithmic Modifications (RECOMMENDED)
Since parameter tuning hasn't succeeded, consider **algorithmic changes**:

1. **Obstacle-aware edge policy**: Don't allow edges that cross through obstacles
   - Check if edge path intersects obstacle
   - Only connect cells if there's a clear path around obstacle
   
2. **Different complex construction**: 
   - Use Vietoris-Rips complex with obstacle-aware distance metric
   - Or modify clique complex construction to avoid filling holes
   
3. **Geometric constraints**:
   - Add explicit geometric constraints to prevent obstacle bridging
   - Use geodesic distance (distance around obstacle) instead of Euclidean

4. **Alternative approaches**:
   - Investigate if Hoffman et al. (2016) used different constraints
   - Check if 3D setup (paper's focus) behaves differently
   - Consider if different coactivity measures are needed

### Option 3: Literature Review
- Re-examine Hoffman et al. (2016) paper for details on obstacle handling
- Check if there are follow-up papers with solutions
- Review similar topological mapping approaches

---

## Recommendations

Given that **74+ parameter sweeps** have been run without success, I recommend:

1. **Document current findings** (this report) ✅
2. **Pause extensive parameter exploration** - diminishing returns
3. **Focus on algorithmic solutions** - structural changes likely needed
4. **Re-examine the paper** - ensure we're implementing obstacle handling correctly

---

## Files Generated

All parameter sweeps are logged in:
- `results/sweeps/<timestamp>_<label>/`
  - `run.log`: Full output with parameters and results
  - `figure.png`: Visualization of results
  - Each sweep documents exact parameters used

**Documentation**:
- `docs/PARAMETER_SWEEPS_EXPLAINED.md`: Explanation of parameter sweeps
- `docs/BETTI_GAP_INVESTIGATION_STATUS.md`: This report

---

## Conclusion

The Betti-gap investigation has been **thoroughly explored** through parameter sweeps. The consistent failure to achieve `(b₀ = 1, b₁ = 1)` across all tested parameter combinations suggests this is a **fundamental algorithmic limitation** rather than a parameter-tuning problem.

**Next phase**: Algorithmic modifications to handle obstacles correctly, or re-examination of the underlying approach.

---

**Last Updated**: 2025-01-20  
**Total Sweeps**: 74+  
**Status**: ⚠️ Parameter exploration complete - Algorithmic solutions required (see `BETTI_GAP_FINDINGS_AND_SOLUTIONS.md`)

