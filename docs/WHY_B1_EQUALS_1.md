# Why Do We Need `b₁ = 1`?

## What Are Betti Numbers?

Betti numbers are topological invariants that measure the "shape" of a space at different dimensions:

- **b₀**: Number of **connected components** (pieces)
  - b₀ = 1 means everything is connected
  - b₀ > 1 means the space is fragmented

- **b₁**: Number of **1-dimensional holes** (loops that can't be contracted)
  - b₁ = 0 means no holes (like a solid disk)
  - b₁ = 1 means one hole (like a donut)
  - b₁ = 2 means two holes (like a figure-8)

- **b₂**: Number of **2-dimensional holes** (voids)
  - Usually 0 in 2D spaces

## Why `b₁ = 1` for One Obstacle?

### Physical Reality

When you have **one circular obstacle** in the middle of an arena:

```
┌─────────────────────┐
│                     │
│      ┌─────┐        │
│      │  ●  │        │  ← Obstacle (hole)
│      └─────┘        │
│                     │
└─────────────────────┘
```

**Topologically**, this is like a **donut** (torus) - there's one hole in the middle. The agent can walk around the obstacle, creating a **non-contractible loop** - a path that goes around the obstacle and can't be shrunk to a point without crossing the obstacle.

### What This Means Mathematically

- **b₀ = 1**: The entire space is connected (agent can reach any point)
- **b₁ = 1**: There is **one hole** (one way to loop around the obstacle)
- **b₂ = 0**: No 2D voids (it's a 2D space)

### Visual Example

**Without obstacle** (empty arena):
```
┌─────────────────────┐
│                     │
│                     │
│                     │
│                     │
└─────────────────────┘

Topology: b₀ = 1, b₁ = 0
(Like a disk - no holes)
```

**With one obstacle**:
```
┌─────────────────────┐
│                     │
│      ┌─────┐        │
│      │  ●  │        │
│      └─────┘        │
│                     │
└─────────────────────┘

Topology: b₀ = 1, b₁ = 1
(Like a donut - one hole)
```

**With two obstacles**:
```
┌─────────────────────┐
│     ┌───┐           │
│     │ ● │           │
│     └───┘           │
│                     │
│           ┌───┐     │
│           │ ● │     │
│           └───┘     │
└─────────────────────┘

Topology: b₀ = 1, b₁ = 2
(Two holes - one per obstacle)
```

## Why This Matters

### Validation of Correct Learning

The Betti numbers tell us if the **learned topological map matches the physical environment**:

- ✅ **Correct**: `(b₀ = 1, b₁ = 1)` → Map correctly identifies one obstacle
- ❌ **Wrong**: `(b₀ = 1, b₁ = 0)` → Map thinks there's no obstacle (hole filled)
- ❌ **Wrong**: `(b₀ >> 1, b₁ = 0)` → Map is fragmented (too sparse)

### Why `b₁ = 0` Is Wrong

If `b₁ = 0` with an obstacle present, it means:

1. **The obstacle hole was filled** by edges in the clique complex
2. **Edges bridge across the obstacle**, creating a path that "crosses through" the obstacle
3. **The topology doesn't match reality** - the map thinks there's no hole when there clearly is one

This is a **sign that the algorithm is not correctly learning the environment structure**.

### The Goal: Match Physical Topology

```
Physical Environment:  Arena + 1 obstacle
                         ↓
Expected Topology:     b₀ = 1, b₁ = 1
                         ↓
Learned Topology:      b₀ = 1, b₁ = 1  ✅ CORRECT
```

If learned topology ≠ physical topology, the mapping is incorrect.

## Current Problem

### What We're Seeing

**All 74+ parameter sweeps show**: `b₁ = 0` even with obstacles present.

**This means**: The learned topology says "no holes" when there clearly is one.

**Why**: Edges bridge across obstacles because we only check Euclidean distance, not whether the path crosses an obstacle.

### What We Want

**Goal**: `b₁ = 1` with one obstacle

**Why**: To match the physical topology of the environment.

**How**: Prevent edges from crossing obstacles (obstacle-aware edge filtering).

---

## Summary

**Why `b₁ = 1`?**

1. **Physical reality**: One obstacle creates one hole in the topology
2. **Mathematical correctness**: Betti numbers should match the environment
3. **Validation**: Correct learning means `b₁ = number of obstacles`
4. **Current problem**: `b₁ = 0` means the algorithm isn't learning correctly

**The algorithmic solution** (obstacle-aware edge filtering) ensures edges don't cross obstacles, preserving the hole topology and achieving the correct `b₁ = 1`.

