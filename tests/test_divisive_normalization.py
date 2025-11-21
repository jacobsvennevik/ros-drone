"""Tests for divisive normalization mode in attractors."""

from __future__ import annotations

import numpy as np
import pytest

from hippocampus_core.grid_cells import GridAttractor, GridAttractorConfig
from hippocampus_core.head_direction import HeadDirectionAttractor, HeadDirectionConfig


def test_grid_divisive_normalization_unit_norm():
    """Test that divisive normalization maintains unit L2 norm."""
    config = GridAttractorConfig(normalize_mode="divisive", size=(15, 15), tau=0.05)
    attractor = GridAttractor(config)
    
    # Run steps to reach steady state
    for _ in range(50):
        attractor.step(velocity=np.array([0.1, 0.05]), dt=0.05)
        
        # Check that norm is approximately 1 (or state is all zeros)
        norm = np.linalg.norm(attractor.state)
        if norm > 1e-6:  # Not all zeros
            assert abs(norm - 1.0) < 1e-3, \
                f"Divisive normalization should maintain unit norm, got {norm:.6f}"


def test_grid_subtractive_vs_divisive():
    """Test that both normalization modes maintain stability."""
    config_sub = GridAttractorConfig(normalize_mode="subtractive", size=(15, 15), tau=0.05)
    config_div = GridAttractorConfig(normalize_mode="divisive", size=(15, 15), tau=0.05)
    
    attractor_sub = GridAttractor(config_sub)
    attractor_div = GridAttractor(config_div)
    
    # Run with same velocity
    velocity = np.array([0.2, -0.1])
    
    for _ in range(50):
        attractor_sub.step(velocity=velocity, dt=0.05)
        attractor_div.step(velocity=velocity, dt=0.05)
        
        # Both should be finite
        assert np.all(np.isfinite(attractor_sub.state))
        assert np.all(np.isfinite(attractor_div.state))
        
        # Both should produce non-negative activity
        activity_sub = attractor_sub.activity()
        activity_div = attractor_div.activity()
        assert np.all(activity_sub >= 0.0)
        assert np.all(activity_div >= 0.0)


def test_hd_divisive_normalization():
    """Test that HD attractor divisive normalization works."""
    config = HeadDirectionConfig(normalize_mode="divisive", num_neurons=60, tau=0.05)
    attractor = HeadDirectionAttractor(config)
    
    for _ in range(50):
        attractor.step(omega=0.1, dt=0.05)
        
        norm = np.linalg.norm(attractor.state)
        if norm > 1e-6:
            assert abs(norm - 1.0) < 1e-3, \
                f"HD divisive normalization should maintain unit norm, got {norm:.6f}"


def test_normalization_mode_configuration():
    """Test that normalize_mode config parameter works correctly."""
    for mode in ["subtractive", "divisive"]:
        grid_config = GridAttractorConfig(normalize_mode=mode)
        assert grid_config.normalize_mode == mode
        
        hd_config = HeadDirectionConfig(normalize_mode=mode)
        assert hd_config.normalize_mode == mode

