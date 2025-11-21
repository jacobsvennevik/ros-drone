# Betti-Gap Investigation: Executive Summary

**Date**: 2025-01-20  
**Section**: 1.1 - Betti-gap Investigation  
**Status**: ✅ **COMPREHENSIVE PARAMETER EXPLORATION COMPLETE**

---

## One-Page Summary

### Goal
Achieve stable `(b₀ = 1, b₁ = 1)` regime with obstacles in topological mapping.

### Investigation Scale
- **74+ parameter sweeps** completed
- **Extensive parameter space coverage**
- **Longest simulation**: 41.7 minutes
- **Best result**: `(b₀ = 6, b₁ = 0)` - Achieved consistently but still not goal

### Key Finding
**No parameter combination achieves `b₁ = 1`**. The consistent failure across all tested parameter combinations indicates a **structural/algorithmic limitation** rather than a parameter-tuning issue.

### Root Cause
Edges are added based on **Euclidean distance** without considering obstacles. When two place cells are within `max_edge_distance`, an edge is added regardless of whether the path crosses an obstacle, causing the clique complex to bridge across obstacles and fill holes.

### Recommended Solution
**Obstacle-aware edge filtering**: Check if edge path intersects obstacles before adding edges. This preserves hole topology and should achieve `(b₀ = 1, b₁ = 1)`.

**Implementation**: 2-4 hours  
**Expected outcome**: Should achieve goal with much wider parameter ranges

---

## Quick Reference

| Metric | Value |
|--------|-------|
| Total sweeps | 74+ |
| Sweeps with `b₁ = 1` | 0 |
| Best `b₀` achieved | 6 |
| Best `b₁` achieved | 0 |
| Parameter space coverage | Extensive |
| Recommendation | Implement algorithmic solution |

---

**See**: `docs/BETTI_GAP_FINDINGS_AND_SOLUTIONS.md` for detailed analysis and implementation proposals

