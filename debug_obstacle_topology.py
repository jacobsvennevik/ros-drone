#!/usr/bin/env python3
"""Debug script to understand why obstacles aren't detected as holes."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import matplotlib.pyplot as plt

from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.env import Agent, CircularObstacle, Environment

def main():
    # Create environment with central obstacle
    obstacle = CircularObstacle(center_x=0.5, center_y=0.5, radius=0.15)
    env = Environment(width=1.0, height=1.0, obstacles=[obstacle])
    
    config = PlaceCellControllerConfig(
        num_place_cells=120,
        sigma=0.15,
        max_rate=20.0,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        max_edge_distance=0.4,
        integration_window=None,
    )
    
    controller = PlaceCellController(environment=env, config=config, rng=np.random.default_rng(42))
    agent = Agent(environment=env, random_state=np.random.default_rng(142))
    
    # Run short simulation
    duration = 1200.0
    dt = 0.05
    num_steps = int(duration / dt)
    
    print("Running simulation...")
    for step in range(num_steps):
        position = agent.step(dt)
        controller.step(np.asarray(position), dt)
        if (step + 1) % (num_steps // 10) == 0:
            print(f"  {100 * (step + 1) / num_steps:.0f}%")
    
    # Analyze graph
    graph = controller.get_graph()
    positions = controller.place_cell_positions
    
    print(f"\nGraph stats:")
    print(f"  Nodes: {graph.num_nodes()}")
    print(f"  Edges: {graph.num_edges()}")
    print(f"  Components: {graph.num_components()}")
    
    # Check place cells around obstacle
    obstacle_center = np.array([0.5, 0.5])
    distances_to_obstacle = np.linalg.norm(positions - obstacle_center, axis=1)
    obstacle_radius = 0.15
    
    # Place cells near obstacle (within 0.3 of center, but outside obstacle)
    near_obstacle_mask = (distances_to_obstacle > obstacle_radius) & (distances_to_obstacle < 0.3)
    near_obstacle_indices = np.where(near_obstacle_mask)[0]
    
    print(f"\nPlace cells near obstacle (radius 0.15-0.3): {len(near_obstacle_indices)}")
    
    # Check edges between near-obstacle cells
    near_obstacle_edges = 0
    for i in near_obstacle_indices:
        for j in near_obstacle_indices:
            if i < j and graph.graph.has_edge(i, j):
                near_obstacle_edges += 1
    
    print(f"Edges between near-obstacle cells: {near_obstacle_edges}")
    
    # Try to find cycles and check if they encircle the obstacle
    try:
        import networkx as nx
        cycles = list(nx.cycle_basis(graph.graph))
        print(f"\nNumber of cycles: {len(cycles)}")
        if cycles:
            print(f"Largest cycle size: {max(len(c) for c in cycles)}")
            
            # Check if any cycle encircles the obstacle
            # A cycle encircles the obstacle if the obstacle center is inside the polygon formed by the cycle
            def point_in_polygon(point, polygon_vertices):
                """Check if point is inside polygon using ray casting algorithm."""
                x, y = point
                n = len(polygon_vertices)
                inside = False
                p1x, p1y = positions[polygon_vertices[0]]
                for i in range(1, n + 1):
                    p2x, p2y = positions[polygon_vertices[i % n]]
                    if y > min(p1y, p2y):
                        if y <= max(p1y, p2y):
                            if x <= max(p1x, p2x):
                                if p1y != p2y:
                                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xinters:
                                    inside = not inside
                    p1x, p1y = p2x, p2y
                return inside
            
            encircling_cycles = 0
            for cycle in cycles:
                if len(cycle) >= 3:  # Need at least 3 nodes for a cycle
                    cycle_positions = positions[cycle]
                    if point_in_polygon(obstacle_center, cycle):
                        encircling_cycles += 1
            
            print(f"Cycles that encircle obstacle: {encircling_cycles}")
            
            # Check cliques (triangles) that might fill cycles
            cliques = graph.get_maximal_cliques()
            triangles = [c for c in cliques if len(c) == 3]
            print(f"Number of triangles (3-cliques): {len(triangles)}")
            
            # Check if cycles around obstacle are filled by triangles
            if encircling_cycles > 0:
                print(f"\nNote: {encircling_cycles} cycles encircle the obstacle, but they may be filled by triangles.")
                print(f"This would make them contractible (b₁ = 0).")
    except Exception as e:
        print(f"Error analyzing cycles: {e}")
        import traceback
        traceback.print_exc()
    
    # Compute Betti numbers
    try:
        betti = graph.compute_betti_numbers(max_dim=1)
        print(f"\nBetti numbers:")
        print(f"  b₀ (components): {betti.get(0, 0)}")
        print(f"  b₁ (holes): {betti.get(1, 0)}")
    except Exception as e:
        print(f"\nError computing Betti numbers: {e}")
    
    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot obstacle
    circle = plt.Circle((0.5, 0.5), 0.15, color='red', alpha=0.3, label='Obstacle')
    ax.add_patch(circle)
    
    # Plot all place cells
    ax.scatter(positions[:, 0], positions[:, 1], s=20, color='gray', alpha=0.3, label='All place cells')
    
    # Highlight near-obstacle cells
    if len(near_obstacle_indices) > 0:
        ax.scatter(positions[near_obstacle_indices, 0], 
                  positions[near_obstacle_indices, 1], 
                  s=50, color='blue', alpha=0.7, label='Near obstacle')
    
    # Plot edges
    for i, j in graph.graph.edges():
        ax.plot([positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                'k-', alpha=0.1, linewidth=0.5)
    
    # Highlight edges between near-obstacle cells
    for i in near_obstacle_indices:
        for j in near_obstacle_indices:
            if i < j and graph.graph.has_edge(i, j):
                ax.plot([positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        'b-', alpha=0.5, linewidth=2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Graph Structure Around Obstacle')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('results/debug_obstacle.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to results/debug_obstacle.png")

if __name__ == "__main__":
    main()

