# Parameter Sweeps for Validation Notebooks

This document describes parameter sweep scripts for the Rubin HD and Yartsev grid validation notebooks.

## Overview

The parameter sweep scripts run validation simulations across multiple parameter sets to understand:
- **Rubin HD validation**: How calibration intervals and HD neuron counts affect head-direction tuning
- **Yartsev grid validation**: How calibration intervals and grid sizes affect grid stability

## Rubin HD Validation Sweep

### Usage

```bash
# Sweep both calibration interval and HD neuron count
python experiments/sweep_rubin_hd_validation.py

# Sweep only calibration interval
python experiments/sweep_rubin_hd_validation.py --sweep calibration

# Sweep only HD neuron count
python experiments/sweep_rubin_hd_validation.py --sweep hd_neurons

# Custom options
python experiments/sweep_rubin_hd_validation.py \
    --duration 600 \
    --trials 10 \
    --output-dir results/my_sweeps
```

### Parameters Swept

**Calibration Interval** (default: `[100, 200, 250, 300, 400, 500]`):
- Controls how often the HD/grid attractors are recalibrated
- Lower values = more frequent calibration = better tracking but more computation
- Higher values = less frequent calibration = may allow more drift

**HD Neuron Count** (default: `[36, 48, 60, 72, 90, 108]`):
- Number of neurons in the head-direction attractor
- More neurons = finer angular resolution = better directional precision
- Fewer neurons = coarser resolution but faster computation

### Output Metrics

- **Rayleigh vector (inside field)**: HD tuning strength within the place field
- **Rayleigh vector (outside field)**: HD tuning strength outside the place field
- Higher values indicate stronger directional tuning
- Rubin et al. (2014) found both in-field and out-of-field tuning remain robust

### Output Files

- `results/sweeps/rubin_sweep_calibration.png` - Rayleigh vector vs calibration interval
- `results/sweeps/rubin_sweep_hd_neurons.png` - Rayleigh vector vs HD neuron count

---

## Yartsev Grid Validation Sweep

### Usage

```bash
# Sweep both calibration interval and grid size
python experiments/sweep_yartsev_grid_validation.py

# Sweep only calibration interval
python experiments/sweep_yartsev_grid_validation.py --sweep calibration

# Sweep only grid size
python experiments/sweep_yartsev_grid_validation.py --sweep grid_size

# Custom options
python experiments/sweep_yartsev_grid_validation.py \
    --duration 1200 \
    --trials 10 \
    --output-dir results/my_sweeps
```

### Parameters Swept

**Calibration Interval** (default: `[200, 300, 400, 500, 600]`):
- Controls how often grid phase is recalibrated
- Affects grid stability and drift over time
- Too infrequent = excessive drift
- Too frequent = may interfere with path integration

**Grid Size** (default: `[(12,12), (16,16), (20,20), (24,24)]`):
- Size of the grid cell attractor network
- Larger grids = more spatial resolution = more computational cost
- Smaller grids = coarser resolution but faster computation

### Output Metrics

- **Grid drift metric**: Measures stability of grid activity (lower = more stable)
- **Theta-band power**: Fraction of power in theta frequency band (4-10 Hz)
- Yartsev et al. (2011) found stable grids without continuous theta oscillations

### Output Files

- `results/sweeps/yartsev_sweep_calibration_drift.png` - Grid drift vs calibration interval
- `results/sweeps/yartsev_sweep_calibration_theta.png` - Theta power vs calibration interval
- `results/sweeps/yartsev_sweep_grid_size_drift.png` - Grid drift vs grid size
- `results/sweeps/yartsev_sweep_grid_size_theta.png` - Theta power vs grid size

---

## Interpretation

### Rubin HD Sweep Results

**Expected findings:**
- HD tuning (Rayleigh vector) should remain robust across calibration intervals
- Higher HD neuron counts should improve tuning precision
- Both in-field and out-of-field tuning should be significant (> 0.3)

**Troubleshooting:**
- If Rayleigh values are very low (< 0.1), check that heading data is accurate
- If in-field and out-of-field differ dramatically, may need longer simulation

### Yartsev Grid Sweep Results

**Expected findings:**
- Grid drift should remain low (< 0.1) across parameter ranges
- Theta-band power should remain negligible (< 0.1)
- Grid stability should improve with more frequent calibration

**Troubleshooting:**
- If grid drift is high (> 0.2), try more frequent calibration
- If theta power is significant (> 0.15), check velocity integration
- Large grid sizes may be slower but should be more stable

---

## Quick Start Example

```bash
# Run quick sweep with default parameters
python experiments/sweep_rubin_hd_validation.py --trials 3 --duration 180

# Run comprehensive sweep
python experiments/sweep_yartsev_grid_validation.py --trials 10 --duration 1200

# View results
ls -lh results/sweeps/
```

---

## Integration with CI

These sweep scripts can be integrated into CI/CD pipelines for automated regression testing:

```yaml
# Example GitHub Actions workflow
- name: Run parameter sweeps
  run: |
    python experiments/sweep_rubin_hd_validation.py --trials 3 --duration 180
    python experiments/sweep_yartsev_grid_validation.py --trials 3 --duration 600
```

For production use, consider:
- Running sweeps nightly with full parameter sets
- Alerting on significant metric changes
- Storing results for trend analysis

---

## Related Documentation

- **Validation notebooks**: `notebooks/rubin_hd_validation.ipynb`, `notebooks/yartsev_grid_without_theta.ipynb`
- **Paper analysis**: `docs/rubin_2014_analysis.md`, `docs/yartsev_2011_analysis.md`
- **Controller comparison**: `docs/CONTROLLER_COMPARISON.md`

