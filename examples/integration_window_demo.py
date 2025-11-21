#!/usr/bin/env python3
"""Demonstration of the integration window feature for edge gating.

This script shows how the integration window (ϖ) gates edge admission in the
topological graph, preventing transient coactivity from creating spurious connections.

Run with: python3 examples/integration_window_demo.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np

from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.env import Agent, Environment


def main():
    """Demonstrate integration window edge gating."""
    print("=" * 70)
    print("Integration Window (ϖ) Demonstration")
    print("=" * 70)
    print()
    
    # Create environment
    env = Environment(width=1.0, height=1.0)
    
    # Configuration without integration window (baseline)
    config_no_window = PlaceCellControllerConfig(
        num_place_cells=50,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.1,      # w: coincidence window (100ms)
        coactivity_threshold=3.0,
        max_edge_distance=0.4,
        integration_window=None,     # No integration window
    )
    
    # Configuration with integration window
    config_with_window = PlaceCellControllerConfig(
        num_place_cells=50,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.1,       # w: coincidence window (100ms)
        coactivity_threshold=3.0,
        max_edge_distance=0.4,
        integration_window=2.0,      # ϖ: integration window (2 seconds)
    )
    
    print("Configuration:")
    print(f"  Coincidence window (w): {config_with_window.coactivity_window} s")
    print(f"  Coactivity threshold: {config_with_window.coactivity_threshold}")
    print(f"  Integration window (ϖ): {config_with_window.integration_window} s")
    print()
    
    # Create controllers
    rng = np.random.default_rng(42)
    controller_no_window = PlaceCellController(
        environment=env, config=config_no_window, rng=rng
    )
    controller_with_window = PlaceCellController(
        environment=env, config=config_with_window, rng=rng
    )
    
    # Run same trajectory for both controllers
    agent_rng = np.random.default_rng(123)
    agent = Agent(environment=env, random_state=agent_rng)
    
    dt = 0.05
    num_steps = 200  # 10 seconds total
    print(f"Running simulation for {num_steps} steps ({num_steps * dt:.1f} seconds)...")
    print()
    
    # Track edge counts over time
    edge_counts_no_window = []
    edge_counts_with_window = []
    time_points = []
    
    for step in range(num_steps):
        position = agent.step(dt)
        controller_no_window.step(np.asarray(position), dt)
        controller_with_window.step(np.asarray(position), dt)
        
        # Check edge counts periodically
        if step % 20 == 0 or step == num_steps - 1:
            current_time = (step + 1) * dt
            time_points.append(current_time)
            
            graph_no_window = controller_no_window.get_graph()
            graph_with_window = controller_with_window.get_graph()
            
            edge_counts_no_window.append(graph_no_window.num_edges())
            edge_counts_with_window.append(graph_with_window.num_edges())
    
    # Final results
    final_graph_no_window = controller_no_window.get_graph()
    final_graph_with_window = controller_with_window.get_graph()
    
    print("Results:")
    print(f"  Without integration window:")
    print(f"    Nodes: {final_graph_no_window.num_nodes()}")
    print(f"    Edges: {final_graph_no_window.num_edges()}")
    print(f"    Components: {final_graph_no_window.num_components()}")
    print()
    print(f"  With integration window ({config_with_window.integration_window} s):")
    print(f"    Nodes: {final_graph_with_window.num_nodes()}")
    print(f"    Edges: {final_graph_with_window.num_edges()}")
    print(f"    Components: {final_graph_with_window.num_components()}")
    print()
    
    # Verify gating effect
    assert final_graph_with_window.num_edges() <= final_graph_no_window.num_edges(), (
        "Integration window should gate edges (fewer or equal edges)"
    )
    
    print("✓ Integration window successfully gates edge admission")
    print()
    
    # Show edge count evolution
    print("Edge count evolution over time:")
    print(f"{'Time (s)':<10} {'No window':<12} {'With window':<12} {'Difference':<12}")
    print("-" * 50)
    for t, edges_no, edges_with in zip(time_points, edge_counts_no_window, edge_counts_with_window):
        diff = edges_no - edges_with
        print(f"{t:<10.2f} {edges_no:<12} {edges_with:<12} {diff:<12}")
    
    print()
    print("=" * 70)
    print("Key observations:")
    print("  • Integration window delays edge admission")
    print("  • Edges only appear after pairs exceed threshold for ϖ seconds")
    print("  • This reduces spurious connections from transient coactivity")
    print("  • More stable topological maps result")
    print("=" * 70)


if __name__ == "__main__":
    main()


