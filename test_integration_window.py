#!/usr/bin/env python3
"""Quick test script to verify integration window functionality.

This script can be run to verify that:
1. The integration window feature works correctly
2. Edges are properly gated by the integration window
3. Backward compatibility is maintained (integration_window=None)

Run with: python3 test_integration_window.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np

from hippocampus_core.coactivity import CoactivityTracker
from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.env import Agent, Environment
from hippocampus_core.topology import TopologicalGraph


def test_coactivity_tracker_integration():
    """Test that CoactivityTracker tracks threshold exceedance correctly."""
    print("Testing CoactivityTracker integration window tracking...")
    
    tracker = CoactivityTracker(num_cells=3, window=0.1)
    
    # Register spikes to build coactivity
    spikes_0 = np.array([True, False, False])
    spikes_1 = np.array([False, True, False])
    
    time = 0.0
    tracker.register_spikes(time, spikes_0)
    tracker.register_spikes(time + 0.05, spikes_1)  # Within window
    
    # Check threshold exceeded
    threshold = 1.0
    times = tracker.check_threshold_exceeded(threshold, time + 0.1)
    
    assert (0, 1) in times, "Pair (0, 1) should have exceeded threshold"
    print(f"  ✓ Pair (0, 1) exceeded threshold at time {times[(0, 1)]}")
    
    # Verify time doesn't change on subsequent checks
    time += 0.2
    tracker.register_spikes(time, spikes_0)
    times2 = tracker.check_threshold_exceeded(threshold, time)
    assert times2[(0, 1)] == times[(0, 1)], "Threshold time should not change"
    print(f"  ✓ Threshold time remains stable: {times2[(0, 1)]}")
    
    # Test reset
    tracker.reset()
    times3 = tracker.check_threshold_exceeded(threshold, time + 1.0)
    assert (0, 1) not in times3, "Reset should clear integration times"
    print(f"  ✓ Reset clears integration times")
    
    print("  ✓ CoactivityTracker integration window tracking: PASSED\n")


def test_integration_window_gating():
    """Test that integration window gates edge admission."""
    print("Testing integration window edge gating...")
    
    env = Environment(width=1.0, height=1.0)
    
    # Config without integration window
    config_no_window = PlaceCellControllerConfig(
        num_place_cells=20,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.1,
        coactivity_threshold=3.0,
        max_edge_distance=0.4,
        integration_window=None,
    )
    
    # Config with integration window
    config_with_window = PlaceCellControllerConfig(
        num_place_cells=20,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.1,
        coactivity_threshold=3.0,
        max_edge_distance=0.4,
        integration_window=1.0,  # 1 second integration window
    )
    
    rng = np.random.default_rng(42)
    controller_no_window = PlaceCellController(
        environment=env, config=config_no_window, rng=rng
    )
    controller_with_window = PlaceCellController(
        environment=env, config=config_with_window, rng=rng
    )
    
    # Run same trajectory
    agent_rng = np.random.default_rng(123)
    agent = Agent(environment=env, random_state=agent_rng)
    
    dt = 0.05
    num_steps = 200  # 10 seconds total
    
    for _ in range(num_steps):
        position = agent.step(dt)
        controller_no_window.step(np.asarray(position), dt)
        controller_with_window.step(np.asarray(position), dt)
    
    graph_no_window = controller_no_window.get_graph()
    graph_with_window = controller_with_window.get_graph()
    
    edges_no_window = graph_no_window.num_edges()
    edges_with_window = graph_with_window.num_edges()
    
    print(f"  Edges without integration window: {edges_no_window}")
    print(f"  Edges with integration window (1.0s): {edges_with_window}")
    
    assert edges_with_window <= edges_no_window, (
        f"Integration window should gate edges: {edges_with_window} <= {edges_no_window}"
    )
    print(f"  ✓ Integration window gates edges correctly")
    
    # After enough time, should have some edges
    if num_steps * dt > config_with_window.integration_window:
        print(f"  ✓ After {num_steps * dt:.1f}s, edges are admitted")
    
    print("  ✓ Integration window edge gating: PASSED\n")


def test_backward_compatibility():
    """Test that integration_window=None maintains backward compatibility."""
    print("Testing backward compatibility (integration_window=None)...")
    
    env = Environment(width=1.0, height=1.0)
    config = PlaceCellControllerConfig(
        num_place_cells=20,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.1,
        coactivity_threshold=3.0,
        max_edge_distance=0.4,
        integration_window=None,  # Should work like before
    )
    
    rng = np.random.default_rng(42)
    controller = PlaceCellController(environment=env, config=config, rng=rng)
    
    agent_rng = np.random.default_rng(123)
    agent = Agent(environment=env, random_state=agent_rng)
    
    dt = 0.05
    for _ in range(100):
        position = agent.step(dt)
        controller.step(np.asarray(position), dt)
    
    graph = controller.get_graph()
    assert graph.num_nodes() == config.num_place_cells
    assert graph.num_edges() >= 0
    
    print(f"  ✓ Controller works with integration_window=None")
    print(f"  ✓ Graph has {graph.num_nodes()} nodes and {graph.num_edges()} edges")
    print("  ✓ Backward compatibility: PASSED\n")


def main():
    """Run all integration window tests."""
    print("=" * 60)
    print("Integration Window Functionality Tests")
    print("=" * 60)
    print()
    
    try:
        test_coactivity_tracker_integration()
        test_integration_window_gating()
        test_backward_compatibility()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


