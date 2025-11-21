"""Demo script for SNN Policy Service with SNN inference (Milestone B)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import snntorch
    SNN_AVAILABLE = True
except ImportError:
    SNN_AVAILABLE = False
    print("Warning: PyTorch/snnTorch not available. Install with: pip install torch snntorch")
    print("Falling back to heuristic mode.")

from hippocampus_core.env import Environment
from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.policy import (
    TopologyService,
    SpatialFeatureService,
    SpikingPolicyService,
    ActionArbitrationSafety,
    RobotState,
    Mission,
    MissionGoal,
    GoalType,
    PointGoal,
)

if SNN_AVAILABLE:
    from hippocampus_core.policy import PolicySNN


def main():
    """Run SNN policy demo."""
    print("SNN Policy Service Demo (Milestone B: SNN Inference)")
    print("=" * 60)

    # Setup environment
    env = Environment(width=1.0, height=1.0)
    print(f"Environment: {env.width}m × {env.height}m")

    # Setup place cell controller
    config = PlaceCellControllerConfig(
        num_place_cells=80,
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        integration_window=None,
    )
    rng = np.random.default_rng(42)
    place_controller = PlaceCellController(environment=env, config=config, rng=rng)
    print(f"Place cells: {config.num_place_cells}")

    # Setup policy services
    ts = TopologyService()
    sfs = SpatialFeatureService(ts, k_neighbors=8)
    
    # Create SNN model (or use heuristic if not available)
    snn_model = None
    if SNN_AVAILABLE:
        # Get feature dimension (will be computed from first features)
        # For demo, we'll use a default
        feature_dim = 44  # 2D: 3 (goal) + 8*4 (neighbors) + 3 (topo) + 4 (safety) = 42, +2 (dynamics) = 44
        snn_model = PolicySNN(
            feature_dim=feature_dim,
            hidden_dim=64,
            output_dim=2,
            beta=0.9,
        )
        print("SNN model created (randomly initialized)")
    else:
        print("Using heuristic mode (SNN not available)")
    
    sps = SpikingPolicyService(
        sfs,
        config={"feature_dim": 44 if SNN_AVAILABLE else None},
        snn_model=snn_model,
    )
    aas = ActionArbitrationSafety(max_linear=0.3, max_angular=1.0)
    print("Policy services initialized")

    # Setup mission
    goal = PointGoal(position=(0.9, 0.9))
    mission = Mission(
        goal=MissionGoal(
            type=GoalType.POINT,
            value=goal,
        )
    )
    print(f"Mission goal: {goal.position}")

    # Simulation parameters
    dt = 0.05
    num_steps = 200
    position = np.array([0.1, 0.1])
    heading = 0.0

    # Storage for visualization
    trajectory = []
    actions = []
    decisions = []
    confidences = []

    print("\nRunning simulation...")
    for step in range(num_steps):
        # Update place cell controller
        place_controller.step(position, dt)

        # Update topology service periodically
        if step % 10 == 0:
            ts.update_from_controller(place_controller)

        # Build robot state
        robot_state = RobotState(
            pose=(position[0], position[1], heading),
            time=step * dt,
            previous_action=actions[-1] if actions else None,
        )

        # Build features
        features, context = sfs.build_features(robot_state, mission)

        # Make policy decision
        decision = sps.decide(features, context, dt)

        # Filter through safety
        graph_snapshot = ts.get_graph_snapshot(robot_state.time)
        safe_cmd = aas.filter(decision, robot_state, graph_snapshot, mission)

        # Apply action
        v, omega = safe_cmd.cmd
        heading += omega * dt
        position[0] += v * np.cos(heading) * dt
        position[1] += v * np.sin(heading) * dt

        # Clip to environment bounds
        position[0] = np.clip(position[0], 0.0, env.width)
        position[1] = np.clip(position[1], 0.0, env.height)

        # Store
        trajectory.append(position.copy())
        actions.append((v, omega))
        decisions.append(decision)
        confidences.append(decision.confidence)

        # Print progress
        if step % 50 == 0:
            dist_to_goal = np.linalg.norm(np.array(goal.position) - position)
            print(f"Step {step:3d}: pos=({position[0]:.2f}, {position[1]:.2f}), "
                  f"dist={dist_to_goal:.2f}, v={v:.2f}, ω={omega:.2f}, "
                  f"conf={decision.confidence:.2f}, reason={decision.reason}")

    # Final statistics
    trajectory = np.array(trajectory)
    final_dist = np.linalg.norm(np.array(goal.position) - trajectory[-1])
    print(f"\nSimulation complete!")
    print(f"Final distance to goal: {final_dist:.3f}m")
    print(f"Graph nodes: {len(graph_snapshot.V)}")
    print(f"Graph edges: {len(graph_snapshot.E)}")
    print(f"Average confidence: {np.mean(confidences):.3f}")
    print(f"Decision mode: {decisions[0].reason}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Trajectory plot
    ax = axes[0, 0]
    ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", alpha=0.6, label="Trajectory")
    ax.plot(trajectory[0, 0], trajectory[0, 1], "go", markersize=10, label="Start")
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], "ro", markersize=10, label="End")
    ax.plot(goal.position[0], goal.position[1], "r*", markersize=15, label="Goal")
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Robot Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Action plot
    ax = axes[0, 1]
    actions_array = np.array(actions)
    time_steps = np.arange(num_steps) * dt
    ax.plot(time_steps, actions_array[:, 0], "b-", label="v (m/s)")
    ax.plot(time_steps, actions_array[:, 1], "r-", label="ω (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action")
    ax.set_title("Policy Actions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Confidence plot
    ax = axes[1, 0]
    ax.plot(time_steps, confidences, "g-", label="Confidence")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Confidence")
    ax.set_title("Decision Confidence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Graph visualization
    ax = axes[1, 1]
    if len(graph_snapshot.E) > 0:
        for edge in graph_snapshot.E:
            u_pos = next(n.position for n in graph_snapshot.V if n.node_id == edge.u)
            v_pos = next(n.position for n in graph_snapshot.V if n.node_id == edge.v)
            ax.plot([u_pos[0], v_pos[0]], [u_pos[1], v_pos[1]], "k-", alpha=0.3, linewidth=0.5)

        for node in graph_snapshot.V:
            ax.plot(node.position[0], node.position[1], "bo", markersize=3)

    ax.plot(trajectory[:, 0], trajectory[:, 1], "r-", alpha=0.6, linewidth=2, label="Trajectory")
    ax.plot(goal.position[0], goal.position[1], "r*", markersize=15, label="Goal")
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Topological Graph + Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("results/snn_policy_demo.png", dpi=150)
    print("\nPlot saved to results/snn_policy_demo.png")


if __name__ == "__main__":
    main()

