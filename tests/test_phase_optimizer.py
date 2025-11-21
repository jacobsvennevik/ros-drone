import numpy as np

from hippocampus_core.calibration.phase_optimizer import PhaseOptimizer


def test_phase_optimizer_needs_samples():
    optimizer = PhaseOptimizer(max_history=5)
    assert optimizer.estimate_correction() is None


def test_phase_optimizer_returns_mean_offsets():
    optimizer = PhaseOptimizer(max_history=10)
    for _ in range(12):
        position = np.array([1.0, 1.0])
        grid = np.array([0.8, 0.9])
        optimizer.add_sample(position, heading=0.5, hd_estimate=0.3, grid_estimate=grid)

    correction = optimizer.estimate_correction()
    assert correction is not None
    # Grid translation uses linear mean (correct for 2D position)
    assert np.allclose(correction.grid_translation, np.array([0.2, 0.1]))
    # Heading uses circular mean (vector average), which should match linear mean for small angles
    # In this case, all heading errors are 0.2, so circular mean ≈ linear mean
    assert abs(correction.heading_delta - 0.2) < 1e-6


def test_phase_optimizer_circular_mean_handles_wrap():
    """Test that circular mean correctly handles angle wrapping (e.g., [350°, 10°] → 0°)."""
    optimizer = PhaseOptimizer(max_history=10)
    
    # Simulate heading errors near ±π boundary
    # True headings: [350°, 10°] in degrees = [-10°, 10°] in radians = [~-0.175, ~0.175]
    # Estimates: both at 0° = 0.0 radians
    # Errors: [~-0.175, ~0.175] → wrapped: [~-0.175, ~0.175]
    # Circular mean should be ~0.0 (not ~π or linear mean which could be wrong)
    
    import numpy as np
    
    # Add samples with headings near boundary
    for i in range(10):
        # Alternate between -10° and +10° from estimate
        heading = 0.1 if i % 2 == 0 else -0.1
        optimizer.add_sample(
            position=np.array([0.5, 0.5]),
            heading=heading,
            hd_estimate=0.0,  # Estimate is at 0°
            grid_estimate=np.array([0.5, 0.5]),
        )
    
    correction = optimizer.estimate_correction()
    assert correction is not None
    
    # Circular mean of alternating ±0.1 should be ~0.0 (not large positive or negative)
    # With 10 samples alternating, circular mean should be close to 0
    assert abs(correction.heading_delta) < 0.15  # Should be close to zero, not near π

