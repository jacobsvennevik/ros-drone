# Parameter Sweeps Explained

## What Are Parameter Sweeps?

A **parameter sweep** is a systematic exploration of different parameter combinations to find the optimal settings. In this Betti-gap investigation, we're testing different configurations to achieve the goal: `(b₀ = 1, b₁ = 1)` with obstacles.

## Why Parameter Sweeps?

The topological mapping system has many parameters that affect how the graph is constructed. Finding the right combination is like tuning a radio—you need to find the sweet spot where:
- The graph connects properly (`b₀ = 1`: one connected component)
- The obstacle creates a persistent hole (`b₁ = 1`: one hole around obstacle)
- The graph doesn't fragment (`b₀ > 1` is bad) or fill the hole (`b₁ = 0` is bad)

## Key Parameters We're Testing

### 1. **Coactivity Threshold** (`coactivity_threshold`, `c_min`)
**What it does**: Minimum number of coactivity events needed to form an edge between two place cells.

**Effect**:
- **Too low** (e.g., 3-4): Many edges form quickly, but creates spurious connections → fills the obstacle hole (`b₁ = 0`)
- **Too high** (e.g., 8-10): Fewer edges, graph fragments → multiple components (`b₀ >> 1`)
- **Just right** (e.g., 5-7): Balance between connectivity and sparsity

**Range tested**: 4.0 to 9.5

---

### 2. **Max Edge Distance** (`max_edge_distance`, `d_max`)
**What it does**: Maximum spatial distance between place cell centers to allow an edge. Prevents edges between distant cells.

**Effect**:
- **Too large** (e.g., >0.20): Cells on opposite sides of obstacle can connect → bridges across obstacle → fills hole
- **Too small** (e.g., <0.10): Graph fragments, not enough connectivity → `b₀ >> 1`
- **Just right** (e.g., 0.12-0.16): Cells connect locally but don't bridge obstacle

**Range tested**: 0.09 to 0.19

**Relationship**: Usually set as `d_max = 2.0 * sigma` to ensure place field overlap

---

### 3. **Place Field Size** (`sigma`, σ)
**What it does**: Standard deviation of the Gaussian place field (how wide each place cell's receptive field is).

**Effect**:
- **Smaller σ** (e.g., 0.10-0.11): More precise, needs more cells, tighter connectivity
- **Larger σ** (e.g., 0.15-0.16): Broader coverage, more overlap between cells

**Range tested**: 0.10 to 0.16

---

### 4. **Number of Place Cells** (`num_place_cells`, `N`)
**What it does**: Total number of place cells covering the environment.

**Effect**:
- **Too few** (e.g., <60): Insufficient coverage, gaps in map
- **Too many** (e.g., >120): Dense coverage, more cross-connections → fills holes
- **Just right** (e.g., 80-100): Good coverage without over-connection

**Range tested**: 60 to 110

---

### 5. **Place Cell Placement** (`placement_mode`)
**What it does**: How place cells are distributed in the environment.

**Options**:
- **`uniform`**: Random uniform distribution (default)
- **`obstacle_ring`**: Concentrate cells in a ring around the obstacle
  - `ring_fraction`: Fraction of cells in the ring (e.g., 0.25 = 25%)
  - `ring_offset`: Distance from obstacle edge (e.g., 0.02 = just outside)
  - `ring_jitter`: Random variation in ring positions
- **`ring_spokes`**: Ring + radial spokes extending outward
  - `spoke_fraction`: Fraction of cells in spokes
  - `num_spokes`: Number of radial spokes (e.g., 6)
  - `spoke_extension`: How far spokes extend

**Effect**:
- **Uniform**: General coverage, but cells can bridge across obstacle
- **Ring placement**: Focuses cells around obstacle boundary, helps detect the hole
- **Ring+spokes**: Combines ring (for hole detection) with spokes (for connectivity)

---

### 6. **Integration Window** (`integration_window`, ϖ)
**What it does**: Time window (in seconds) that gates edge admission. An edge is only added if coactivity exceeds threshold for this entire duration.

**Effect**:
- **No window** (0): Edges form immediately from transient coactivity → more spurious edges
- **Short window** (60-120s): Some filtering, but still many false connections
- **Long window** (240-480s): More stable, fewer spurious edges, but slower learning

**Range tested**: 0, 60, 120, 180, 240, 360, 480 seconds

---

### 7. **Simulation Duration**
**What it does**: How long the agent explores the environment.

**Effect**:
- **Too short** (<300s): Not enough time for graph to form
- **Longer** (600-900s): More complete exploration, stable topology

**Range tested**: 180 to 900 seconds

---

### 8. **Trajectory Mode**
**What it does**: How the agent moves through the environment.

**Options**:
- **`random`**: Random walk (default)
- **`orbit_then_random`**: Orbit around obstacle first, then random walk
  - `orbit_duration`: How long to orbit (e.g., 180s)
  - `orbit_radius`: Distance from obstacle center
  - `orbit_speed`: Speed along orbit

**Effect**:
- **Random**: General exploration
- **Orbit**: Ensures agent explores around obstacle boundary, which should help detect the hole

---

## Example Parameter Sweep

Here's what a typical parameter sweep looks like:

```bash
scripts/logged_validate.sh ring_sparse_v1 \
  --obstacle \
  --obstacle-radius 0.15 \
  --coactivity-threshold 6.5 \      # Parameter 1: c_min
  --max-edge-distance 0.13 \        # Parameter 2: d_max
  --num-cells 90 \                  # Parameter 3: N
  --sigma 0.12 \                    # Parameter 4: σ
  --placement-mode obstacle_ring \  # Parameter 5: placement
  --ring-fraction 0.3 \             # Parameter 5a: ring fraction
  --ring-offset 0.02 \              # Parameter 5b: ring offset
  --duration 600 \                  # Parameter 6: duration
  --integration-windows 0 120 240   # Parameter 7: ϖ values
```

This runs experiments with these exact parameters and logs:
- Final `(b₀, b₁)` values
- Number of edges formed
- Learning time `T_min`
- Visualization plots

---

## What We're Looking For

**Success criteria**: `(b₀ = 1, b₁ = 1)`
- `b₀ = 1`: One connected component (graph is connected)
- `b₁ = 1`: One hole (obstacle is correctly detected)

**Common failures**:
- `(b₀ = 1, b₁ = 0)`: Connected but hole filled → **too many edges bridge across obstacle**
- `(b₀ >> 1, b₁ = 0)`: Fragmented graph → **too sparse, not enough connectivity**
- `(b₀ >> 1, b₁ >> 1)`: Many fragments and spurious holes → **very fragmented with spurious loops**

---

## Current Status

**Total sweeps run**: 15+ documented attempts
**Best result so far**: `(b₀ = 2, b₁ = 0)` or `(b₀ = 1, b₁ = 0)` 
**Goal**: `(b₀ = 1, b₁ = 1)` ❌ **Not yet achieved**

**Challenge**: Finding the sweet spot between:
- **Too dense** → bridges across obstacle → `b₁ = 0`
- **Too sparse** → graph fragments → `b₀ >> 1`

This is why we need systematic parameter sweeps—we're searching for that elusive combination that gives us connectivity without filling the hole!

