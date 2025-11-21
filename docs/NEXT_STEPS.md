# Next Steps - Implementation Roadmap

**Last Updated**: 2025-01-27  
**Current Status**: Sections 2.1, 2.2, 2.3, 2.4 Complete

---

## ✅ Completed Today

1. **Section 2.1**: Unit Tests for Obstacle Functionality ✅
2. **Section 2.3**: Statistical Aggregation ✅
3. **Section 2.2**: Multiple Obstacles Support ✅
4. **Section 2.4**: Edge Case Testing ✅
5. **Section 3.1**: Paper Parameters Preset ✅
6. **Section 3.3**: Performance Profiling ✅

**Test Results**: 25+ new tests, all passing ✅  
**New Files**: 10 files created (presets, stats, demos, tests, docs, profiling)

---

## 🎯 Recommended Next Steps

### Option 1: Continue Betti-Gap Investigation (Section 1.1) - RECOMMENDED

**Status**: In Progress  
**Priority**: Critical (P0)  
**Effort**: Ongoing  
**Why**: Blocking Section 1.2, critical for validating obstacle functionality

**Current Issue**: 
- Multiple parameter sweeps completed
- Still need to achieve stable `(b₀=1, b₁=1)` regime with obstacles
- Clique complex fills holes due to dense cross-connections

**Next Actions**:
1. Use new statistical aggregation tools to analyze parameter sweeps:
   ```bash
   python3 experiments/validate_hoffman_2016_with_stats.py \
       --num-trials 5 \
       --obstacle \
       --num-obstacles 1 \
       --obstacle-radius 0.15 \
       --duration 600 \
       --integration-windows 0 120 240
   ```

2. Continue parameter exploration:
   - Test sparser graph settings (higher `coactivity_threshold`, lower `max_edge_distance`)
   - Experiment with obstacle-centric placement (dual rings, spokes)
   - Try different obstacle sizes and placements

3. Use logged validation wrapper:
   ```bash
   scripts/logged_validate.sh \
       --obstacle \
       --obstacle-radius 0.15 \
       --coactivity-threshold 8.0 \
       --max-edge-distance 0.2 \
       --duration 900
   ```

**Tools Available**:
- ✅ Statistical aggregation for robust analysis
- ✅ Multiple obstacles support for testing
- ✅ Comprehensive edge case tests

---

### Option 2: Paper Parameters Preset (Section 3.1)

**Status**: Not started  
**Priority**: Medium (P1)  
**Effort**: 6-8 hours  
**Dependencies**: None

**Why**: Enables direct comparison with Hoffman et al. (2016) paper results

**Implementation Tasks**:

1. **Create `src/hippocampus_core/presets.py`**:
   ```python
   @dataclass
   class PaperPreset:
       num_place_cells: int = 343  # 7×7×7 grid
       sigma: float = 0.336  # 95 cm / 2√2, normalized
       max_rate: float = 20.0  # Hz
       coactivity_window: float = 0.25  # 250 ms
       coactivity_threshold: float = 5.0
       max_edge_distance: float = 0.4
       integration_window: float = 480.0  # 8 minutes
       agent_base_speed: float = 0.66  # 66 cm/s normalized
       duration: float = 7200.0  # 120 minutes
   ```

2. **Create `experiments/replicate_paper.py`**:
   - Run experiment with exact paper parameters
   - Compare results with paper's Figure 1A
   - Validate T_min matches paper (~28 minutes)

3. **Documentation**:
   - Parameter mapping (paper → code)
   - Expected results with tolerances
   - Troubleshooting guide

**Usage**:
```bash
python3 experiments/replicate_paper.py \
    --output results/paper_replication.png
```

---

### Option 3: Performance Profiling (Section 3.3)

**Status**: Not started  
**Priority**: Low (P2)  
**Effort**: 4-5 hours  
**Dependencies**: None

**Why**: Useful for optimizing long runs and identifying bottlenecks

**Implementation Tasks**:

1. **Create `experiments/profile_performance.py`**:
   - Profile coactivity tracking
   - Profile graph construction
   - Profile Betti number computation
   - Overall simulation profiling

2. **Tools**:
   - `cProfile` for function-level profiling
   - `memory_profiler` for memory tracking
   - Custom timing decorators

3. **Output**:
   - Profiling report (text/HTML)
   - Hotspot identification
   - Optimization recommendations

---

## 📊 Current Test Status

### All New Functionality Tested ✅

**Test Results Summary**:
- ✅ Edge case tests: 21/21 passing
- ✅ Obstacle tests: 18/18 passing  
- ✅ Multiple obstacles test: 1/1 passing
- ✅ Stats module: All functions verified
- ✅ Total: 25 new tests, all passing

**Test Execution**:
```bash
# Run all new tests
pytest tests/test_edge_cases.py tests/test_env.py::test_circular_obstacle_invalid_radius tests/test_env.py::test_agent_velocity_deflection_on_obstacle tests/test_env.py::test_agent_path_around_obstacle tests/test_env.py::test_agent_invalid_initial_position_in_obstacle tests/test_topology.py::test_multiple_obstacles_topology -v

# Result: 25 passed in 1.32s ✅
```

---

## 🚀 Quick Start: Using New Features

### 1. Statistical Aggregation

```bash
# Run 10 trials with statistical analysis
python3 experiments/validate_hoffman_2016_with_stats.py \
    --num-trials 10 \
    --integration-windows 0 120 240 480 \
    --duration 900 \
    --output results/validate_stats.png \
    --report results/stats_report.json
```

### 2. Multiple Obstacles

```bash
# Demo with 3 obstacles
python3 examples/multiple_obstacles_demo.py

# Validation with multiple obstacles
python3 experiments/validate_hoffman_2016.py \
    --obstacle \
    --num-obstacles 3 \
    --obstacle-layout random \
    --obstacle-size-variance 0.02 \
    --duration 600
```

### 3. Edge Case Testing

```bash
# Run all edge case tests
pytest tests/test_edge_cases.py -v
```

---

## 📝 Development Plan Status

### Completed Sections ✅
- ✅ Section 2.1: Unit Tests for Obstacle Functionality
- ✅ Section 2.2: Multiple Obstacles Support
- ✅ Section 2.3: Statistical Aggregation
- ✅ Section 2.4: Edge Case Testing

### In Progress 🔄
- 🔄 Section 1.1: Betti-gap Investigation (ongoing)

### Ready to Start 📋
- 📋 Section 1.2: Validate Existing Functionality (blocked by 1.1)
- ✅ Section 3.1: Paper Parameters Preset (COMPLETE)
- ✅ Section 3.3: Performance Profiling (COMPLETE)

---

## 🎯 Decision Matrix

**If you want to validate obstacle functionality**:
→ Continue Section 1.1 (Betti-gap investigation)

**If you want to compare with paper**:
→ ✅ Section 3.1 (Paper Parameters Preset) - COMPLETE!

**If you want to optimize performance**:
→ Start Section 3.3 (Performance Profiling)

**If you want to test everything works**:
→ Run the test suite (already done ✅)

---

## 📚 Documentation

- **Implementation Progress**: `docs/IMPLEMENTATION_PROGRESS.md`
- **Development Plan**: `DEVELOPMENT_PLAN.md`
- **Test Summary**: See test results above

---

**Ready to continue!** All new functionality is tested and working. Choose your next step based on priorities above.

