#!/usr/bin/env python3
"""Demonstration of Betti number computation for topological verification.

This script shows how to compute Betti numbers from the learned topological graph
to verify that the topology matches the physical environment.

Run with: python3 examples/betti_numbers_demo.py

Requires: pip install ripser (or gudhi)
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


def check_persistent_homology_available():
    """Check if persistent homology computation is available."""
    try:
        from hippocampus_core.persistent_homology import is_persistent_homology_available
        return is_persistent_homology_available()
    except ImportError:
        return False


def main():
    """Demonstrate Betti number computation."""
    print("=" * 70)
    print("Betti Number Computation Demonstration")
    print("=" * 70)
    print()
    
    # Check if persistent homology is available
    if not check_persistent_homology_available():
        print("ERROR: Persistent homology computation requires ripser or gudhi.")
        print("Install with: pip install ripser")
        print("             or: pip install gudhi")
        sys.exit(1)
    
    from hippocampus_core.persistent_homology import RIPSER_AVAILABLE, GUDHI_AVAILABLE
    
    backend_used = "ripser" if RIPSER_AVAILABLE else "gudhi"
    print(f"Using backend: {backend_used}")
    print()
    
    # Create environment and controller
    env = Environment(width=1.0, height=1.0)
    
    config = PlaceCellControllerConfig(
        num_place_cells=80,
        sigma=0.12,
        max_rate=18.0,
        coactivity_window=0.15,
        coactivity_threshold=4.0,
        max_edge_distance=0.35,
        integration_window=1.5,  # Use integration window for stable maps
    )
    
    rng = np.random.default_rng(42)
    controller = PlaceCellController(environment=env, config=config, rng=rng)
    
    # Run simulation
    agent_rng = np.random.default_rng(123)
    agent = Agent(environment=env, random_state=agent_rng)
    
    dt = 0.05
    num_steps = 300  # 15 seconds
    print(f"Running simulation for {num_steps} steps ({num_steps * dt:.1f} seconds)...")
    print()
    
    for step in range(num_steps):
        position = agent.step(dt)
        controller.step(np.asarray(position), dt)
    
    # Get the learned graph
    graph = controller.get_graph()
    
    print("Graph statistics:")
    print(f"  Nodes (place cells): {graph.num_nodes()}")
    print(f"  Edges: {graph.num_edges()}")
    print(f"  Connected components: {graph.num_components()}")
    print()
    
    # Compute Betti numbers
    print("Computing Betti numbers...")
    try:
        betti = graph.compute_betti_numbers(max_dim=2, backend="auto")
        
        print()
        print("Betti numbers:")
        print(f"  b₀ (connected components): {betti[0]}")
        print(f"  b₁ (1D holes/loops): {betti[1]}")
        print(f"  b₂ (2D holes/voids): {betti.get(2, 0)}")
        print()
        
        # Validate results
        print("Validation:")
        assert betti[0] >= 1, "b_0 should be at least 1"
        assert betti[0] == graph.num_components(), (
            f"b_0 should equal number of components: {betti[0]} == {graph.num_components()}"
        )
        print(f"  ✓ b₀ matches component count: {betti[0]} == {graph.num_components()}")
        print(f"  ✓ b₁ (holes) = {betti[1]}")
        print()
        
        # Interpretation
        print("Topological interpretation:")
        if betti[0] == 1:
            print("  • Single connected component (b₀ = 1)")
        else:
            print(f"  • {betti[0]} disconnected regions (b₀ = {betti[0]})")
        
        if betti[1] == 0:
            print("  • No holes or loops detected (b₁ = 0)")
        elif betti[1] == 1:
            print("  • One hole/loop detected (b₁ = 1)")
        else:
            print(f"  • {betti[1]} holes/loops detected (b₁ = {betti[1]})")
        
        if betti.get(2, 0) > 0:
            print(f"  • {betti[2]} 2D voids detected (b₂ = {betti[2]})")
        
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("Key points:")
    print("  • Betti numbers quantify the topological structure")
    print("  • b₀ = number of connected components")
    print("  • b₁ = number of 1D holes (loops)")
    print("  • Use Betti numbers to verify learned topology matches environment")
    print("=" * 70)


def demonstrate_known_topologies():
    """Demonstrate Betti numbers on known topologies."""
    if not check_persistent_homology_available():
        return
    
    print()
    print("=" * 70)
    print("Known Topology Examples")
    print("=" * 70)
    print()
    
    from hippocampus_core.topology import TopologicalGraph
    
    # Example 1: Cycle (4 nodes in a square)
    print("Example 1: Cycle graph (4 nodes in a square)")
    positions_cycle = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    graph_cycle = TopologicalGraph(positions_cycle)
    
    coactivity_cycle = np.zeros((4, 4))
    coactivity_cycle[0, 1] = coactivity_cycle[1, 0] = 5.0
    coactivity_cycle[1, 2] = coactivity_cycle[2, 1] = 5.0
    coactivity_cycle[2, 3] = coactivity_cycle[3, 2] = 5.0
    coactivity_cycle[3, 0] = coactivity_cycle[0, 3] = 5.0
    
    graph_cycle.build_from_coactivity(coactivity_cycle, c_min=3.0, max_distance=2.0)
    betti_cycle = graph_cycle.compute_betti_numbers(max_dim=1)
    print(f"  Edges: {graph_cycle.num_edges()}")
    print(f"  b₀ = {betti_cycle[0]} (should be 1 - connected)")
    print(f"  b₁ = {betti_cycle[1]} (should be 1 - one hole)")
    print()
    
    # Example 2: Path (3 nodes in a line)
    print("Example 2: Path graph (3 nodes in a line)")
    positions_path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    graph_path = TopologicalGraph(positions_path)
    
    coactivity_path = np.zeros((3, 3))
    coactivity_path[0, 1] = coactivity_path[1, 0] = 5.0
    coactivity_path[1, 2] = coactivity_path[2, 1] = 5.0
    
    graph_path.build_from_coactivity(coactivity_path, c_min=3.0, max_distance=2.0)
    betti_path = graph_path.compute_betti_numbers(max_dim=1)
    print(f"  Edges: {graph_path.num_edges()}")
    print(f"  b₀ = {betti_path[0]} (should be 1 - connected)")
    print(f"  b₁ = {betti_path[1]} (should be 0 - no holes)")
    print()


if __name__ == "__main__":
    main()
    demonstrate_known_topologies()


