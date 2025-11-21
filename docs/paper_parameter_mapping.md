# Paper Parameter Mapping: Hoffman et al. (2016)

This document maps parameters from Hoffman et al. (2016) to our implementation.

**Paper**: arXiv:1601.04253v1 [q-bio.NC] - "Topological mapping of space in bat hippocampus"

---

## Parameter Mapping

### Place Cell Parameters

| Paper Parameter | Paper Value | Our Implementation | Notes |
|----------------|-------------|-------------------|-------|
| Number of place cells | 343 (7×7×7 grid) | `num_place_cells = 343` | 3D grid in paper, can use 343 in 2D |
| Place field size | 95 cm | `sigma = 0.336` | σ = 95 cm / (2√2) ≈ 33.6 cm, normalized to 1.0 m arena |
| Maximum firing rate | ~20 Hz | `max_rate = 20.0` | Typical value |

### Coactivity Parameters

| Paper Parameter | Paper Value | Our Implementation | Notes |
|----------------|-------------|-------------------|-------|
| Coactivity window (w) | 250 ms | `coactivity_window = 0.25` | Time window for spike coincidence |
| Coactivity threshold | ~5.0 | `coactivity_threshold = 5.0` | Minimum coactivity for edge formation |
| Max edge distance | Place field overlap | `max_edge_distance = 0.4` | Normalized distance |

### Integration Window

| Paper Parameter | Paper Value | Our Implementation | Notes |
|----------------|-------------|-------------------|-------|
| Integration window (ϖ) | 8 minutes | `integration_window = 480.0` | Delay before edge admission |

### Agent Parameters

| Paper Parameter | Paper Value | Our Implementation | Notes |
|----------------|-------------|-------------------|-------|
| Mean speed | 66 cm/s | `agent_base_speed = 0.66` | Normalized to 1.0 m arena |
| Max speed | ~132 cm/s | `agent_max_speed = 1.32` | 2× base speed |

### Simulation Parameters

| Paper Parameter | Paper Value | Our Implementation | Notes |
|----------------|-------------|-------------------|-------|
| Duration | 120 minutes | `duration = 7200.0` | Total simulation time |
| Time step | ~50 ms | `dt = 0.05` | Default in our implementation |

---

## Expected Results

### Learning Time (T_min)

**Paper**: ~28 minutes for clique complex

**Our Implementation**: Should be in range 20-35 minutes with paper parameters

**Validation**:
```bash
python3 experiments/replicate_paper.py --quick  # Quick test
python3 experiments/replicate_paper.py          # Full replication
```

### Final Topology

**Paper**: 
- b₀ = 1 (connected graph)
- b₁ = 0 (no holes, for uniform environment)

**Our Implementation**: Should match these values

---

## Usage Examples

### Full Paper Replication (120 minutes, 343 cells)

```bash
python3 experiments/replicate_paper.py \
    --output results/paper_replication_full.png \
    --seed 42
```

**Note**: This will take a long time (hours). Use `--quick` for testing.

### Quick Test (10 minutes, 100 cells)

```bash
python3 experiments/replicate_paper.py \
    --quick \
    --output results/paper_replication_quick.png
```

### 2D-Optimized (120 minutes, 100 cells)

```bash
python3 experiments/replicate_paper.py \
    --2d \
    --output results/paper_replication_2d.png
```

### Custom Duration

```bash
python3 experiments/replicate_paper.py \
    --duration 1800 \
    --output results/paper_replication_30min.png
```

---

## Parameter Notes

### Normalization

The paper uses physical units (cm, seconds), while our implementation uses normalized units:
- **Arena size**: 1.0 × 1.0 (normalized, equivalent to 1 m × 1 m)
- **Speed**: 0.66 (normalized, equivalent to 66 cm/s in 1 m arena)
- **Sigma**: 0.336 (normalized, equivalent to 33.6 cm in 1 m arena)

### 2D vs 3D

The paper uses 3D (7×7×7 = 343 cells), but our current implementation is 2D. For 2D:
- Use `--2d` flag for 100 cells (maintains similar density)
- Or use full 343 cells in 2D (slower but matches cell count)

### Integration Window

The paper's integration window (ϖ = 8 minutes) is critical for:
- Reducing spurious loops
- Producing stable maps
- Matching paper's T_min values

---

## Troubleshooting

### T_min Too High

**Problem**: T_min > 40 minutes (much higher than paper's ~28 min)

**Solutions**:
1. Check integration window is set correctly (480 seconds)
2. Verify coactivity threshold (should be ~5.0)
3. Ensure sufficient simulation duration (120 minutes)
4. Check place cell density (343 cells for full replication)

### T_min Too Low

**Problem**: T_min < 15 minutes (much lower than paper)

**Possible causes**:
- Integration window too short
- Coactivity threshold too low
- Too many place cells (over-sampling)

### Topology Not Correct

**Problem**: b₀ ≠ 1 or unexpected b₁

**Solutions**:
1. Increase integration window
2. Adjust coactivity threshold
3. Check place cell placement
4. Verify simulation ran long enough

---

## References

- **Paper**: Hoffman et al. (2016) - "Topological mapping of space in bat hippocampus"
- **arXiv**: 1601.04253v1 [q-bio.NC]
- **Key Figure**: Figure 1A (learning curves)
- **Key Result**: T_min ≈ 28 minutes for clique complex with ϖ = 8 min

---

**Last Updated**: 2025-01-27

