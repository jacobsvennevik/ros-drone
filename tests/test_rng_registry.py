"""Tests for central RNG registry."""

from __future__ import annotations

import numpy as np

from hippocampus_core.utils.random import RNGRegistry


def test_rng_registry_get():
    """Test that RNG registry returns consistent RNGs."""
    # Reset registry
    RNGRegistry.reset()
    
    # Get RNG with seed
    rng1 = RNGRegistry.get("test_module", seed=42)
    rng2 = RNGRegistry.get("test_module", seed=42)
    
    # Should return same RNG instance
    assert rng1 is rng2
    
    # Should be deterministic - generate values from same RNG instance
    # (calling normal() twice on same RNG advances state, so values differ)
    # Instead, verify that the RNG produces deterministic sequence
    val1 = rng1.normal(size=5)
    val1_continued = rng1.normal(size=5)
    
    # Reset and recreate RNG with same seed should produce same sequence
    RNGRegistry.reset("test_module")
    rng3 = RNGRegistry.get("test_module", seed=42)
    val2 = rng3.normal(size=5)
    val2_continued = rng3.normal(size=5)
    
    # Same seed should produce same sequence
    np.testing.assert_array_equal(val1, val2)
    np.testing.assert_array_equal(val1_continued, val2_continued)


def test_rng_registry_separate_modules():
    """Test that different modules get different RNGs."""
    RNGRegistry.reset()
    
    rng1 = RNGRegistry.get("module1", seed=42)
    rng2 = RNGRegistry.get("module2", seed=42)
    
    # Different modules should get different RNGs
    assert rng1 is not rng2
    
    # But both should be deterministic from their seeds
    val1 = rng1.normal(size=5)
    val2 = rng2.normal(size=5)
    
    # They might produce same values if seeds are same, but instances are different
    # (This is actually expected - same seed gives same sequence)


def test_rng_registry_seed_override():
    """Test that seed only applies on first call."""
    RNGRegistry.reset()
    
    # First call creates RNG with seed 42
    rng1 = RNGRegistry.get("test", seed=42)
    val1 = rng1.normal(size=5)
    
    # Second call with different seed should return existing RNG
    rng2 = RNGRegistry.get("test", seed=99)
    assert rng1 is rng2  # Same instance
    
    # Value should continue sequence from first seed
    val2 = rng2.normal(size=5)
    # This should be the next values in the sequence, not from seed 99


def test_rng_registry_reset():
    """Test that reset clears RNGs."""
    RNGRegistry.reset()
    
    rng1 = RNGRegistry.get("test", seed=42)
    RNGRegistry.reset("test")
    
    rng2 = RNGRegistry.get("test", seed=42)
    
    # After reset, should get new RNG
    assert rng1 is not rng2
    
    # But with same seed, should produce same sequence
    val1 = rng1.normal(size=5)
    val2 = rng2.normal(size=5)
    np.testing.assert_array_equal(val1, val2)


def test_rng_registry_clear():
    """Test that clear removes all RNGs."""
    RNGRegistry.reset()
    
    rng1 = RNGRegistry.get("module1", seed=42)
    rng2 = RNGRegistry.get("module2", seed=43)
    
    RNGRegistry.clear()
    
    # After clear, new RNGs should be created
    rng3 = RNGRegistry.get("module1", seed=42)
    rng4 = RNGRegistry.get("module2", seed=43)
    
    assert rng1 is not rng3
    assert rng2 is not rng4


def test_rng_registry_reproducibility():
    """Test that RNG registry ensures reproducibility."""
    RNGRegistry.reset()
    
    base_seed = 123
    grid_rng = RNGRegistry.get("grid_cells", seed=base_seed + 1)
    hd_rng = RNGRegistry.get("hd_cells", seed=base_seed + 2)
    place_rng = RNGRegistry.get("place_cells", seed=base_seed + 3)
    
    # All should be different instances
    assert grid_rng is not hd_rng
    assert hd_rng is not place_rng
    assert grid_rng is not place_rng
    
    # All should be deterministic
    grid_vals = grid_rng.normal(size=3)
    hd_vals = hd_rng.normal(size=3)
    place_vals = place_rng.normal(size=3)
    
    # Reset and recreate - should get same sequences
    RNGRegistry.clear()
    grid_rng2 = RNGRegistry.get("grid_cells", seed=base_seed + 1)
    hd_rng2 = RNGRegistry.get("hd_cells", seed=base_seed + 2)
    place_rng2 = RNGRegistry.get("place_cells", seed=base_seed + 3)
    
    np.testing.assert_array_equal(grid_rng2.normal(size=3), grid_vals)
    np.testing.assert_array_equal(hd_rng2.normal(size=3), hd_vals)
    np.testing.assert_array_equal(place_rng2.normal(size=3), place_vals)

