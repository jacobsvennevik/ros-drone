# Troubleshooting Guide

Common issues and solutions for topological mapping experiments.

## Table of Contents

- [Integration Window Issues](#integration-window-issues)
- [Betti Number Computation](#betti-number-computation)
- [Inconsistent Statistics](#inconsistent-statistics)
- [Visualization Problems](#visualization-problems)
- [Performance Issues](#performance-issues)

---

## Integration Window Issues

### Problem: ϖ = 480s shows 0 edges in validation results

**Symptom**: Validation script shows "Final Edges = 0" for long integration windows.

**Cause**: Simulation duration is too short. The integration window gates edge admission - if the simulation ends before edges have been admitted for the required duration, no edges will appear.

**Solution**: 
- Increase simulation duration: `--duration 1200` (20 minutes) for ϖ = 480s
- General rule: Use duration ≥ 2× integration_window
- For ϖ = 480s (8 minutes), use at least 16 minutes (960 seconds), preferably 20+ minutes

**Example**:
```bash
# Wrong: Too short
python3 experiments/validate_hoffman_2016.py --integration-windows 480 --duration 300

# Correct: Long enough
python3 experiments/validate_hoffman_2016.py --integration-windows 480 --duration 1200
```

---

### Problem: T_min shows 0.0 for all integration windows

**Symptom**: Summary table shows "T_min (min) = 0.0" for all rows.

**Cause**: This bug was fixed in recent updates. If you still see this, you may be using an old version or there's an issue with the learning time calculation.

**Solution**:
1. Ensure you have the latest version of `experiments/validate_hoffman_2016.py`
2. Check that Betti numbers are being computed (not showing -1)
3. If the issue persists, the topology may be correct from the start (rare)

**Expected behavior**: T_min should increase with longer integration windows:
- ϖ = None: ~0.5-1.0 minutes
- ϖ = 60s: ~1-2 minutes
- ϖ = 120s: ~2-3 minutes
- ϖ = 240s: ~3-4 minutes
- ϖ = 480s: ~4-6 minutes

---

### Problem: Integration window prevents all edge formation

**Symptom**: Even with long durations, edges don't form with integration window enabled.

**Possible causes**:
1. **Duration still too short**: See solution above
2. **Coactivity threshold too high**: Pairs never exceed threshold
3. **Place cell density too low**: Not enough coactivity events
4. **Agent not exploring enough**: Limited navigation coverage

**Solutions**:
1. Check duration: `--duration 1200` or longer
2. Lower threshold: `--coactivity-threshold 3.0` (default is 5.0)
3. Increase place cells: `--num-cells 150` or more
4. Check agent exploration (should cover arena reasonably well)

---

## Betti Number Computation

### Problem: "No module named 'ripser'" or Betti numbers show -1

**Symptom**: Scripts print warnings or Betti numbers appear as -1.

**Cause**: Persistent homology library not installed.

**Solution**: Install ripser or gudhi:
```bash
pip install ripser
# or
pip install gudhi
```

Or install with project extras:
```bash
pip install -e .[dev,ph]
```

**Note**: Scripts will still run without persistent homology, but Betti number computation will be skipped and marked as -1.

---

### Problem: Betti numbers inconsistent with component count

**Symptom**: `b₀` from `compute_betti_numbers()` doesn't match `num_components()`.

**Cause**: 
- When edges = 0, Betti computation on empty clique complex may return b₀ = 1 (incorrect)
- Betti numbers come from clique complex, which may differ slightly from graph components

**Solution**: 
- When edges = 0, prefer `num_components()` (correctly counts isolated nodes)
- The scripts now handle this automatically
- If edges > 0 and they still disagree, this may be expected (clique complex vs graph topology)

**Example of correct behavior**:
- 100 nodes, 0 edges → `num_components() = 100`, `b₀` should also be 100 (scripts auto-correct)
- 100 nodes, many edges → `num_components() = 1`, `b₀` should be 1 (may differ slightly, that's OK)

---

### Problem: Ripser warnings about distance matrix

**Symptom**: Many warnings like "The input matrix is square, but the distance_matrix flag is off".

**Cause**: Ripser warning about precomputed distance matrix (harmless).

**Solution**: This is now suppressed in the code. If you still see warnings, update to the latest version.

---

## Inconsistent Statistics

### Problem: Final edges = 0 but b₀ = 1

**Symptom**: Summary table shows 0 edges but b₀ = 1 (should equal number of nodes).

**Cause**: Inconsistency in graph computation or Betti number fallback.

**Solution**: 
- This is now automatically detected and corrected in validation script
- If you see this in your own code, use `num_components()` when edges = 0:
```python
if graph.num_edges() == 0:
    b0 = graph.num_components()  # Correct: equals number of isolated nodes
else:
    betti = graph.compute_betti_numbers()
    b0 = betti[0]  # May differ slightly from components (clique complex)
```

---

### Problem: Summary table doesn't match plots

**Symptom**: Final statistics in table don't match what's shown in plots.

**Cause**: Different graph snapshots or timing issues.

**Solution**:
- Ensure summary uses the last sample point: `results["edges"][-1]`
- Verify all stats come from the same graph: `graph = controller.get_graph()` once per sample
- Check integration window isn't causing edge counts to change between sampling and final summary

The validation script now ensures consistency automatically.

---

## Visualization Problems

### Problem: Plots don't appear (blank window or no output)

**Symptom**: Script runs but no plot window appears.

**Possible causes**:
1. Non-GUI environment (SSH, headless server)
2. Matplotlib backend issue
3. Plot window closed too quickly

**Solutions**:
1. Save to file: `--output results/plot.png`
2. Use non-interactive backend:
```python
import matplotlib
matplotlib.use('Agg')  # Before importing pyplot
```

---

### Problem: FileNotFoundError when saving plots

**Symptom**: Error like "No such file or directory: 'results/plot.png'".

**Cause**: Output directory doesn't exist.

**Solution**: Scripts now create directories automatically. If you still see this:
```bash
mkdir -p results
```

Or the scripts will create it automatically (fixed in recent updates).

---

### Problem: Barcode plots show wrong values

**Symptom**: Top row barcodes show b₀ = 1 when there are many isolated components.

**Cause**: Betti number computation issue when edges = 0 (now fixed).

**Solution**: Update to latest version. The scripts now use `num_components()` when edges = 0.

**Expected behavior**:
- Start: b₀ = 120 (or number of place cells if all isolated)
- After learning: b₀ = 1 (connected)
- With obstacle: b₁ should show bars at b₁ = 1 after learning

---

## Performance Issues

### Problem: Scripts run very slowly

**Symptom**: Validation or visualization scripts take a long time.

**Solutions**:
1. **Reduce duration**: Use shorter simulations for testing
2. **Reduce place cells**: `--num-cells 50` instead of 120
3. **Reduce sampling frequency**: Scripts sample automatically, but fewer samples = faster
4. **Skip Betti numbers**: If ripser/gudhi not installed, Betti computation is skipped (faster)

**Example**:
```bash
# Fast test run
python3 experiments/validate_hoffman_2016.py \
    --duration 300 \
    --num-cells 50 \
    --integration-windows 0 120
```

---

### Problem: Out of memory with long simulations

**Symptom**: Memory errors or system slowdown.

**Solutions**:
1. Reduce number of place cells
2. Reduce duration or use fewer integration window values
3. Don't store full graph snapshots (visualization script stores some, but not all)

---

## Obstacle Environment Issues

### Problem: Agent gets stuck near obstacles

**Symptom**: Agent position doesn't change or oscillates near obstacle.

**Cause**: Obstacle avoidance may need tuning.

**Solution**: 
- Agent uses simple bounce-off strategy
- For complex scenarios, you may need to adjust agent speed or obstacle radius
- Agent will automatically push away from obstacles

---

### Problem: Obstacle not detected as hole (b₁ = 0 when should be 1)

**Symptom**: Environment with obstacle shows b₁ = 0 instead of expected b₁ = 1.

**Possible causes**:
1. **Simulation too short**: Not enough time for agent to encircle obstacle
2. **Obstacle too small**: Place cells may not be dense enough to detect hole
3. **Obstacle outside exploration area**: Agent never visits area around obstacle

**Solutions**:
1. Increase duration: `--duration 1200` or longer
2. Increase obstacle radius: `--obstacle-radius 0.2` (default is 0.15)
3. Ensure agent explores around obstacle (random walk should eventually cover it)

**Example**:
```bash
# With obstacle
python3 experiments/validate_hoffman_2016.py \
    --obstacle \
    --obstacle-radius 0.2 \
    --duration 1200 \
    --num-cells 150
```

---

## Getting Help

If you encounter issues not covered here:

1. Check the error message for specific details
2. Verify you're using the latest version of the scripts
3. Try running with simpler parameters first (no integration window, shorter duration)
4. Check that all dependencies are installed correctly
5. Review `docs/topological_mapping_usage.md` for usage examples

---

## Quick Diagnostic Checklist

Run this to check your setup:

```bash
# 1. Check dependencies
python3 -c "import numpy, matplotlib, networkx; print('Core deps OK')"
python3 -c "import ripser; print('Ripser OK')" || echo "Install ripser for Betti numbers"

# 2. Test basic functionality
python3 examples/integration_window_demo.py

# 3. Test validation script (short run)
python3 experiments/validate_hoffman_2016.py \
    --duration 300 \
    --integration-windows 0 60 \
    --num-cells 50

# 4. Test obstacle environment
python3 examples/obstacle_environment_demo.py
```

If all tests pass, your setup is working correctly!

