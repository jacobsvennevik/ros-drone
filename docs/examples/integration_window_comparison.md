# Integration Window Comparison

**Example**: Side-by-side comparison of maps with and without integration window

**Script**: `examples/integration_window_demo.py`

## Overview

This example demonstrates the key finding from Hoffman et al. (2016): integration windows (ϖ) prevent spurious connections by requiring sustained pairwise coactivity evidence before admitting edges.

## What It Shows

### Without Integration Window (ϖ = None)

- Edges form **immediately** when coactivity threshold is exceeded
- **Faster** initial learning
- **Risk**: Transient coactivity can create spurious connections
- Useful for: Real-time control where speed matters more than accuracy

### With Integration Window (ϖ = 120s, 240s, 480s)

- Edges only form after **sustained** coactivity evidence
- **Slower** initial learning
- **Benefit**: Fewer spurious connections, more stable maps
- Useful for: Research validation, accurate topological maps

## Expected Results

**Summary Table** (typical values):

| ϖ (s) | Final Edges | Final b₀ | Final b₁ | T_min (min) |
|-------|-------------|----------|----------|-------------|
| None  | ~1073       | 1        | 0        | ~0.5        |
| 120   | ~1073       | 1        | 0        | ~2.0        |
| 240   | ~1073       | 1        | 0        | ~3.5        |
| 480   | ~1073       | 1        | 0        | ~4.5        |

**Key Observations**:
1. Final topology is similar (all reach b₀ = 1, b₁ = 0)
2. Learning time T_min increases with longer integration windows
3. Maps are more stable with integration windows (fewer spurious loops during learning)

## Usage

```bash
python3 examples/integration_window_demo.py
```

Or see the full validation:
```bash
python3 experiments/validate_hoffman_2016.py \
    --integration-windows 0 60 120 240 480 \
    --duration 900
```

## When to Use Integration Window

**Use integration window when**:
- You need **stable, accurate maps**
- Agent moves **fast** (high-speed scenarios)
- You're doing **research validation**
- You want to **match paper findings**

**Skip integration window when**:
- You need **quick map building** for real-time control
- Agent moves **slowly** (transient coactivity less problematic)
- You're doing **exploratory** mapping
- Computation time is critical

