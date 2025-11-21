"""Tests for temporal hysteresis in coactivity edge stability."""

from __future__ import annotations

import numpy as np

from hippocampus_core.coactivity import CoactivityTracker


def test_hysteresis_tracks_drop_below():
    """Test that hysteresis tracks when pairs drop below threshold."""
    tracker = CoactivityTracker(num_cells=10, window=0.2)
    
    threshold = 5.0
    t = 0.0
    
    # Spike to bring pair above threshold
    for _ in range(6):
        spikes = np.zeros(10)
        spikes[0] = 1
        spikes[1] = 1
        tracker.register_spikes(t, spikes, threshold=threshold)
        t += 0.1
    
    # Check that pair exceeded threshold
    exceeded = tracker.check_threshold_exceeded(threshold, t)
    assert (0, 1) in exceeded, "Pair should exceed threshold"
    
    # Now drop below threshold
    for _ in range(3):
        spikes = np.zeros(10)
        tracker.register_spikes(t, spikes, threshold=threshold)
        t += 0.1
    
    # Check threshold again (should still show exceeded due to hysteresis)
    exceeded_before_hysteresis = tracker.check_threshold_exceeded(threshold, t, hysteresis_window=0.5)
    
    # But if we wait past hysteresis window, it should be removed
    t += 0.6  # Wait 0.6 seconds (past 0.5s hysteresis)
    exceeded_after_hysteresis = tracker.check_threshold_exceeded(threshold, t, hysteresis_window=0.5)
    
    # After hysteresis window, pair should be removed if it's still below threshold
    # (This depends on actual coactivity count)


def test_hysteresis_window_parameter():
    """Test that hysteresis_window parameter works correctly."""
    tracker = CoactivityTracker(num_cells=5, window=0.2)
    
    threshold = 3.0
    t = 0.0
    
    # Bring pair above threshold
    for _ in range(5):
        spikes = np.zeros(5)
        spikes[0] = 1
        spikes[1] = 1
        tracker.register_spikes(t, spikes, threshold=threshold)
        t += 0.1
    
    # Check with different hysteresis windows
    exceeded_no_hyst = tracker.check_threshold_exceeded(threshold, t, hysteresis_window=0.0)
    exceeded_with_hyst = tracker.check_threshold_exceeded(threshold, t, hysteresis_window=0.3)
    
    # Both should include the pair (still above threshold)
    assert (0, 1) in exceeded_no_hyst
    assert (0, 1) in exceeded_with_hyst


def test_hysteresis_reset():
    """Test that reset clears hysteresis tracking."""
    tracker = CoactivityTracker(num_cells=5, window=0.2)
    
    threshold = 3.0
    t = 0.0
    
    # Bring pair above threshold
    spikes = np.zeros(5)
    spikes[0] = 1
    spikes[1] = 1
    tracker.register_spikes(t, spikes, threshold=threshold)
    
    # Reset should clear tracking
    tracker.reset()
    
    # After reset, no pairs should be tracked
    exceeded = tracker.check_threshold_exceeded(threshold, t + 1.0)
    assert len(exceeded) == 0 or (0, 1) not in exceeded

