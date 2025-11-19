# Obstacle Environment Demonstration

**Example**: Learning topological maps with obstacles that create holes

**Script**: `examples/obstacle_environment_demo.py`

## Overview

This example demonstrates how obstacles create topological holes (b₁ > 0) in the learned map. With a central obstacle, the system should correctly identify b₁ = 1 (one hole encircling the obstacle), matching the physical topology of the environment.

## What It Shows

### Physical Setup

- **Arena**: 1.0 × 1.0 unit square
- **Obstacle**: Circular obstacle at center (0.5, 0.5) with radius 0.15
- **Agent**: Random walk that automatically avoids obstacles
- **Place cells**: 120 cells covering the accessible space

### Expected Topology

**Without obstacle**:
- b₀ = 1 (connected)
- b₁ = 0 (no holes)
- b₂ = 0 (no voids)

**With central obstacle**:
- b₀ = 1 (connected)
- b₁ = 1 (one hole around obstacle) ✓
- b₂ = 0 (no voids)

### Learning Process

1. **Early stage**: Graph fragmented (many components), no clear hole structure
2. **Middle stage**: Edges form, graph connects, may show spurious holes temporarily
3. **Final stage**: Stable topology with correct hole count (b₁ = 1)

## Key Findings

### Topology Correctly Identifies Obstacles

The topological map correctly learns that paths encircling the obstacle are non-contractible, resulting in b₁ = 1.

### Integration Window Matters

With integration window (ϖ), the system takes longer to learn but produces more stable maps with fewer spurious features.

### Visualization Shows Learning

The comparison plots show:
- How b₁ evolves from 0 → peaks → settles at 1 (with obstacle)
- How edges form around the obstacle (final graph visualization)
- Comparison with no obstacle (b₁ stays at 0)

## Usage

```bash
python3 examples/obstacle_environment_demo.py
```

**Output**:
- Comparison table (with vs. without obstacle)
- Comparison plots saved to `results/obstacle_comparison.png`

## Advanced Usage

### Custom Obstacle Parameters

You can create custom obstacle environments:

```python
from hippocampus_core.env import CircularObstacle, Environment

# Multiple obstacles
obstacles = [
    CircularObstacle(0.3, 0.3, 0.1),  # Top-left
    CircularObstacle(0.7, 0.7, 0.1),  # Bottom-right
]

env = Environment(width=1.0, height=1.0, obstacles=obstacles)
```

**Expected**: b₁ should equal number of obstacles (if well-separated and agent explores around all of them).

### With Validation Script

```bash
python3 experiments/validate_hoffman_2016.py \
    --obstacle \
    --obstacle-radius 0.2 \
    --duration 1200 \
    --integration-windows 0 120 240 480
```

This validates that integration windows work correctly with obstacle environments.

## Troubleshooting

**Problem**: Obstacle not detected (b₁ = 0 when should be 1)

**Solutions**:
1. Increase duration (learning takes time)
2. Increase obstacle radius (easier to detect)
3. Ensure agent explores around obstacle (random walk should eventually cover it)
4. Check place cell density (more cells = better coverage)

**Problem**: Too many holes (b₁ >> 1)

**Possible causes**:
- Spurious connections during learning
- Need longer integration window
- Simulation too short (learning incomplete)

**Solutions**:
1. Use integration window (ϖ = 240s or longer)
2. Increase simulation duration
3. Lower coactivity threshold if edges aren't forming

## Interpretation

This example validates that:
1. ✅ Topological learning correctly identifies physical obstacles
2. ✅ Betti numbers match expected topology (b₁ = 1 with one obstacle)
3. ✅ Integration windows produce stable maps
4. ✅ Visualization clearly shows learning progress

This matches the paper's findings that topological maps correctly encode the structure of the physical environment.

