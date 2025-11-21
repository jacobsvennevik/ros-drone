# Hippocampal Navigation Architecture

## Conceptual Summary

**Concept:** This architecture instantiates a biologically grounded navigation pipeline reproducing bat-like grid and head-direction dynamics in 3-D, without a global theta oscillation.

**Goal:** Test how non-oscillatory continuous attractor dynamics can sustain coherent spatial maps and reward-modulated policy control.

The system implements a hierarchical neural network that transforms raw sensorimotor inputs (velocity, heading) into stable spatial representations (place cells) suitable for topological mapping and navigation. This model is grounded in experimental findings from bat hippocampus (Yartsev et al., 2011) and theoretical work on continuous attractor networks (Burak & Fiete, 2009; Cueva & Wei, 2018).

---

## Overview

This document describes the architecture of the hippocampal navigation system, focusing on the HD → grid → conjunctive place cell pipeline used in `BatNavigationController`.

The bat hippocampal navigation model combines three key neural layers:

1. **Head Direction (HD) Attractor**: Tracks heading from angular velocity
2. **Grid Cell Attractor**: Tracks position via path integration
3. **Conjunctive Place Cells**: Combine HD + grid activity into place fields

These layers work together with periodic calibration to correct drift and maintain stable spatial representations.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Agent/Environment                          │
│  • Position (x, y)                                              │
│  • Heading (θ)                                                  │
│  • Angular velocity (ω)                                         │
│  • Linear velocity (v_x, v_y)                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Observation: [x, y, θ]
                         │
         ┌───────────────▼────────────────────┐
         │   BatNavigationController          │
         │                                    │
         │  ┌──────────────────────────────┐ │
         │  │  Head Direction Attractor    │ │
         │  │  • Angular velocity (ω) ────┤ │
         │  │  • Circular attractor        │ │
         │  │  • HD activity vector        │ │
         │  │  • Estimated heading: θ_est  │ │
         │  └──────────────┬───────────────┘ │
         │                 │                  │
         │  ┌──────────────▼───────────────┐ │
         │  │  Grid Cell Attractor         │ │
         │  │  • Linear velocity (v) ─────┤ │
         │  │  • 2D phase space           │ │
         │  │  • Grid activity matrix     │ │
         │  │  • Estimated position: p_est│ │
         │  └──────────────┬───────────────┘ │
         │                 │                  │
         │  ┌──────────────▼───────────────┐ │
         │  │  Conjunctive Place Cells     │ │
         │  │  • HD activity ─────────────┤ │
         │  │  • Grid activity ───────────┤ │
         │  │  • Weighted combination     │ │
         │  │  • Place cell rates         │ │
         │  └──────────────┬───────────────┘ │
         │                 │                  │
         │  ┌──────────────▼───────────────┐ │
         │  │  Phase Optimizer             │ │
         │  │  • Calibration samples       │ │
         │  │  • Drift estimation          │ │
         │  │  • Correction signals        │ │
         │  └──────────────────────────────┘ │
         └────────────────────────────────────┘
                         │
                         │ Place cell rates
                         │
         ┌───────────────▼────────────────────┐
         │   Topology Learning                │
         │  • Coactivity tracking             │
         │  • Graph construction              │
         │  • Betti number computation        │
         └────────────────────────────────────┘
```

---

## Head Direction (HD) Attractor

The HD attractor maintains a stable heading estimate using a circular attractor network.

### Architecture

```
Angular Velocity (ω)
        │
        ▼
┌───────────────────────┐
│  HD Attractor Network │
│                       │
│  • N neurons (ring)   │
│  • Recurrent weights  │
│  • Global inhibition  │
│  • Attractor dynamics │
└───────────┬───────────┘
            │
            ▼
   HD Activity Vector
   [a₀, a₁, ..., a_{N-1}]
            │
            ▼
     Estimated Heading
         θ_est
```

### Dynamics

- **Input**: Angular velocity `ω` (from heading change)
- **Output**: HD activity vector (firing rates of N HD neurons)
- **Attractor**: Circular topology maintains stable bump of activity
- **Estimation**: Peak activity → heading estimate

### Key Parameters

- `hd_num_neurons`: Number of HD neurons (default: 60)
- `hd_tau`: Time constant for dynamics (default: 0.05s)
- `hd_gamma`: Inhibition strength (default: 1.0)
- `hd_weight_sigma`: Recurrent weight spread (default: 0.4)

---

## Grid Cell Attractor

The grid attractor maintains a position estimate via path integration using a 2D phase space.

### Architecture

```
Linear Velocity (v_x, v_y)
        │
        ▼
┌───────────────────────┐
│  Grid Attractor       │
│                       │
│  • M×M phase space    │
│  • Velocity integration│
│  • Periodic boundary  │
│  • Attractor dynamics │
└───────────┬───────────┘
            │
            ▼
   Grid Activity Matrix
   [a_{0,0}, ..., a_{M-1,M-1}]
            │
            ▼
    Estimated Position
        (x_est, y_est)
```

### Dynamics

- **Input**: Linear velocity `v = (v_x, v_y)` (from position change)
- **Output**: Grid activity matrix (M×M neurons)
- **Attractor**: 2D toroidal topology maintains stable bump
- **Path Integration**: Velocity → phase update → position estimate
- **Drift Metric**: Measures deviation from ground truth (for calibration)

### Key Parameters

- `grid_size`: Grid dimensions (M, M) (default: (15, 15))
- `grid_tau`: Time constant (default: 0.05s)
- `grid_velocity_gain`: Velocity scaling (default: 1.0)

---

## Conjunctive Place Cells

Conjunctive place cells combine HD and grid activity to produce place-specific firing.

### Architecture

```
HD Activity Vector           Grid Activity Matrix
     [a_HD]          +            [a_grid]
        │                         │
        │                         │
        └────────┬────────────────┘
                 │
        ┌────────▼────────┐
        │  Conjunctive    │
        │  Place Cells    │
        │                 │
        │  • Weight matrix│
        │  • Bias terms   │
        │  • ReLU/sigmoid │
        └────────┬────────┘
                 │
                 ▼
        Place Cell Rates
        [r₀, r₁, ..., r_{P-1}]
```

### Computation

For each place cell `i`:
```
r_i = f(W_i^HD · a_HD + W_i^grid · a_grid + b_i)
```

Where:
- `W_i^HD`: Weight vector for HD input
- `W_i^grid`: Weight vector for grid input (flattened)
- `b_i`: Bias term
- `f`: Activation function (typically ReLU or sigmoid)

### Key Parameters

- `num_place_cells`: Number of place cells (default: from parent config)
- `conj_weight_scale`: Weight scaling factor (default: 0.4)
- `conj_bias`: Bias term (default: 0.0)

---

## Calibration Loop

Periodic calibration corrects drift in HD and grid estimates.

### Calibration Architecture

```
Ground Truth              Estimated
Position/Heading    -    Position/Heading
     │                     │
     └──────────┬──────────┘
                │
                ▼
       ┌────────────────┐
       │ Phase Optimizer│
       │                │
       │ • Sample history│
       │ • Estimate drift│
       │ • Compute correction│
       └────────┬───────┘
                │
                ▼
    Correction Signals
    • HD heading delta
    • Grid translation
```

### Calibration Process

1. **Sample Collection**: Every step, store `(position, heading, hd_estimate, grid_estimate)`
2. **Periodic Calibration**: Every `calibration_interval` steps:
   - Compute average drift: `heading_delta`, `grid_translation`
   - Inject HD correction: `hd_attractor.inject_cue(new_heading, gain=2.0)`
   - Shift grid phase: `grid_attractor.shift_phase(shift)`
   - Clear history
3. **Drift Compensation**: Corrections reduce accumulated errors

### Key Parameters

- `calibration_interval`: Steps between calibrations (default: 200)
- `calibration_history`: Number of samples to store (default: 256)

---

## Data Flow

### Step-by-Step Flow

```
1. Agent.step(dt, include_theta=True)
   → Returns: [x, y, θ]

2. BatNavigationController.step(obs, dt)
   ├─ Extract: position = [x, y], theta = θ
   ├─ Compute: omega = (theta - prev_theta) / dt
   ├─ Compute: velocity = (position - prev_position) / dt
   │
   ├─ HD Attractor
   │  └─ hd_attractor.step(omega, dt)
   │     → HD activity vector
   │
   ├─ Grid Attractor
   │  └─ grid_attractor.step(velocity, dt)
   │     → Grid activity matrix
   │
   ├─ Conjunctive Place Cells
   │  └─ compute_rates(grid_activity, hd_activity)
   │     → Place cell rates
   │
   ├─ Calibration
   │  └─ calibration.add_sample(...)
   │     → Periodic correction
   │
   └─ Store prev_position, prev_heading

3. Place cell rates → Topology learning
   └─ Coactivity tracking, graph construction
```

---

## Integration with Topology Learning

The `BatNavigationController` extends `PlaceCellController`, so place cell rates feed into the same topology learning pipeline:

```
Place Cell Rates
       │
       ▼
┌──────────────────┐
│ Coactivity       │
│ Tracking         │
│ • Sliding window │
│ • Coincidence    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Graph            │
│ Construction     │
│ • Edge admission │
│ • Integration    │
│   window (ϖ)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Topology         │
│ Analysis         │
│ • Betti numbers  │
│ • Components     │
└──────────────────┘
```

---

## Comparison: PlaceCellController vs BatNavigationController

### PlaceCellController (Simple)

```
Position [x, y]
    │
    ▼
Gaussian Place Cells
    │
    ▼
Place Cell Rates
    │
    ▼
Topology Learning
```

### BatNavigationController (Full)

```
Position [x, y] + Heading [θ]
    │
    ├─→ HD Attractor ─┐
    │                 │
    └─→ Grid Attractor│
                      │
                      ▼
          Conjunctive Place Cells
                      │
                      ▼
          Place Cell Rates
                      │
                      ▼
          Topology Learning
```

**Key Differences**:
- PlaceCell: Direct position → place cells
- BatNavigation: HD + grid → conjunctive → place cells (biologically realistic)

---

## ROS 2 Integration

### Message Flow

```
/odom (nav_msgs/Odometry)
    │
    ├─ Extract: position (x, y)
    ├─ Extract: orientation → heading (θ)
    │
    ▼
BatNavigationController.step([x, y, θ], dt)
    │
    ├─→ HD/grid activity
    ├─→ Place cell rates
    │
    ▼
Topology learning
    │
    ▼
/cmd_vel (geometry_msgs/Twist)
```

### ROS Node Structure

```
┌──────────────────────┐
│   BrainNode /        │
│   PolicyNode         │
│                      │
│  ┌────────────────┐  │
│  │ ROS Callbacks  │  │
│  │ • /odom        │  │
│  └──────┬─────────┘  │
│         │            │
│  ┌──────▼─────────┐  │
│  │ Controller     │  │
│  │ (Bat/Place)    │  │
│  └──────┬─────────┘  │
│         │            │
│  ┌──────▼─────────┐  │
│  │ Publishers     │  │
│  │ • /cmd_vel     │  │
│  │ • /diagnostics │  │
│  └────────────────┘  │
└──────────────────────┘
```

---

## Key Design Principles

1. **Modularity**: Each layer (HD, grid, conjunctive) is independent
2. **Attractor Dynamics**: Stable representations via recurrent networks
3. **Calibration**: Periodic corrections maintain accuracy
4. **Extensibility**: Easy to add new layers or modify existing ones
5. **Compatibility**: BatNavigationController extends PlaceCellController API

---

## References

- **Hoffman et al. (2016)**: Topological mapping in bat hippocampus
- **Yartsev et al. (2011)**: Grid cells without theta oscillations
- **Rubin et al. (2014)**: Head direction tuning in place cells

For implementation details, see:
- `src/hippocampus_core/controllers/bat_navigation_controller.py`
- `src/hippocampus_core/head_direction.py`
- `src/hippocampus_core/grid_cells.py`
- `src/hippocampus_core/conjunctive_place_cells.py`
- `src/hippocampus_core/calibration/phase_optimizer.py`

