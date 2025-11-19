# Betti Number Evolution (Barcode Timelines)

**Example**: Barcode-style visualization of Betti number persistence over time

**Script**: `examples/topology_learning_visualization.py`

## Overview

This example shows how topological features appear and persist during learning, using barcode-style plots inspired by Figure 1A from Hoffman et al. (2016).

## What Barcodes Show

### Barcode Format

Each horizontal bar represents a topological feature that exists during a time interval:
- **Position (X-axis)**: When the feature exists (time in minutes)
- **Height (Y-axis)**: Betti number value (what type of feature)
- **Length**: How long the feature persists

### b₀ (Components) Barcode

**What it shows**: How the graph fragments/connects over time

**Interpretation**:
- **Long bar at high b₀** (e.g., 120): Many isolated components (fragmented)
- **Bar drops to b₀ = 1**: Graph becomes connected (learning complete)
- **Bar stays at b₀ = 1**: Stable connected map

**Example timeline**:
```
b₀ = 120 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ (0-8 min: fragmented)
b₀ = 1   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ (8-20 min: connected)
```

### b₁ (Holes) Barcode

**What it shows**: How topological holes (loops) appear and persist

**Interpretation**:
- **Bar at b₁ = 0**: No holes (simple connected space) ✓
- **Bar at b₁ > 0**: Holes exist (either correct, e.g., obstacle, or spurious)
- **Short bars that disappear**: Spurious loops during learning (good: they vanish)
- **Long bar at b₁ = 1**: Stable hole (e.g., obstacle correctly detected)

**Example for simple arena**:
```
b₁ = 0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ (entire time: no holes)
```

**Example for arena with obstacle**:
```
b₁ = 0 ━━━━━━━━━━━━━━                                    (early: learning, some spurious)
b₁ = 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ (after learning: correct hole)
```

### b₂ (Voids) Barcode

**What it shows**: 2D holes (usually 0 for 2D simulations)

**Interpretation**:
- Should stay at b₂ = 0 for 2D environments
- b₂ > 0 would indicate 3D structure or over-connection

## Reading the Complete Visualization

The `topology_learning_visualization.py` script shows:

1. **Top row (barcodes)**: Persistence of features
2. **Middle row (line plots)**: Quantitative evolution
3. **Bottom row (snapshots)**: Visual graph structure at key times

**Together they show**:
- When fragmentation ends (b₀ → 1)
- When spurious features disappear (b₁ peaks then settles)
- How graph structure evolves visually

## Key Insights

1. **Integration window delays feature formation**: With ϖ = 480s, b₀ stays high longer, then drops rapidly when edges finally form
2. **Spurious features disappear**: Short bars in b₁ indicate transient holes that vanish (correct behavior)
3. **Learning time visible**: When b₀ drops to 1 and stays there = learning complete
4. **Stable topology**: Long bars at correct values = successful learning

## Usage

```bash
# Standard visualization with integration window
python3 examples/topology_learning_visualization.py \
    --integration-window 480 \
    --duration 1200 \
    --output results/topology_480.png

# Compare with no integration window
python3 examples/topology_learning_visualization.py \
    --integration-window 0 \
    --duration 600 \
    --output results/topology_none.png
```

## Expected Output

For simple arena (no obstacles):
- b₀: Bar at high value (e.g., 120) → drops to 1 → stays at 1
- b₁: Bar at 0 for entire time
- b₂: Bar at 0 for entire time

For arena with obstacle:
- b₀: Same as above
- b₁: May show transient peaks, then stable bar at 1
- b₂: Bar at 0

