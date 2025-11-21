# Betti-Number Usage Guide

Betti numbers remain the authoritative test for verifying that the learned coactivity graph matches the environment’s topology (per Hoffman et al. 2016). The newer head-direction/grid diagnostics and notebooks explain *why* learning succeeds or fails, but they do not replace the Betti check.

## When to Run Betti Analysis

1. **Regression / CI**: Every time `PlaceCellController` or `BatNavigationController` logic changes, run `examples/topology_learning_visualization.py` (or the automated `validate_hoffman_2016_with_stats.py`) and confirm:
   - `b₀ = 1` (one connected component)
   - `b₁` equals the expected number of loops (e.g., 1 for a cave with a single column)
   - Higher Betti numbers vanish unless the environment contains voids.
2. **Environment changes**: Any time you change obstacle layouts or edge-gating rules, re-run Betti analysis to ensure the learned graph still matches geometry.
3. **Long-running experiments**: For large-scale sweeps (see `results/sweeps`), capture Betti traces so anomalies can be spotted automatically.

## How to Run

```bash
# Legacy place-cell controller
python examples/topology_learning_visualization.py --integration-window 480 --duration 600

# BatNavigationController (new pipeline)
python examples/topology_learning_visualization.py --controller bat --integration-window 480 --duration 600
```

Both modes save or display Betti timelines; the bat controller adds HD/grid diagnostics for additional insight.

## Relationship to New Diagnostics

| Tool | Purpose | Typical Symptoms |
|------|---------|------------------|
| Betti numbers | Ground-truth topology verification | Fragmented map (`b₀ > 1`), missing loops (`b₁ = 0`), spurious loops (`b₁ 7`) |
| `notebooks/rubin_hd_validation.ipynb` | Validates head-direction tuning inside/outside fields | Reveals whether directional cues are maintained even when spikes are sparse |
| `notebooks/yartsev_grid_without_theta.ipynb` | Confirms grid attractor stability without theta | Detects grid drift or theta-band leakage |
| Controller snapshots (`examples/topology_learning_visualization.py --controller bat`) | Visualizes HD estimates, grid norms, mean conjunctive rates | Diagnoses calibration drift, insufficient HD alignment |

Use the diagnostics to identify causes (e.g., heading integration failure), then confirm the final result by checking Betti numbers.

## Recommended Workflow

1. **Functional run**: Execute the bat controller demo with diagnostics enabled, ensure HD/grid signals stay well-behaved.
2. **Topological verification**: Compute Betti numbers (either through the same demo or `validate_hoffman_2016_with_stats.py`).
3. **Notebook validation** (optional): Use the Rubin/Yartsev notebooks to reproduce specific biological metrics when tuning parameters or preparing publications.

Maintaining this dual view (objective Betti checks + explanatory diagnostics) ensures we meet both the biological fidelity requirements and the robotics/topology guarantees.
