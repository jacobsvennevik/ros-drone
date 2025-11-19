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

**Status**: Not started  
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

**Status**: Not started  
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
- ⏳ Unit tests for obstacles
- ⏳ Statistical aggregation

### Should Have (P1)
- ⏳ Multiple obstacles support
- ⏳ Paper parameters preset
- ⏳ Comprehensive test suite
- ⏳ Extended duration tests

### Nice to Have (P2)
- ⏳ 3D support
- ⏳ Theta-precession experiments
- ⏳ Performance optimization
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

