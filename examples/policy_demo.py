"""Demo script for SNN Policy Service showcasing BatNavigationController with HD/grid stats."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from hippocampus_core.env import Environment
from hippocampus_core.controllers.bat_navigation_controller import (
    BatNavigationController,
    BatNavigationControllerConfig,
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


def main():
    """Run policy demo with BatNavigationController and HD/grid stats."""
    print("SNN Policy Service Demo with BatNavigationController")
    print("=" * 60)

    # Setup environment
    env = Environment(width=1.0, height=1.0)
    print(f"Environment: {env.width}m × {env.height}m")

    # Setup bat navigation controller
    config = BatNavigationControllerConfig(
        num_place_cells=80,
        hd_num_neurons=72,
        grid_size=(16, 16),
        coactivity_window=0.2,
        coactivity_threshold=5.0,
        integration_window=None,  # Faster for demo
        calibration_interval=250,
    )
    rng = np.random.default_rng(42)
    controller = BatNavigationController(environment=env, config=config, rng=rng)
    print(f"Place cells: {config.num_place_cells}")
    print(f"HD neurons: {config.hd_num_neurons}")
    print(f"Grid size: {config.grid_size}")

    # Initialize position and heading (policy will control these)
    position = np.array([0.1, 0.1], dtype=float)  # Start position
    heading = 0.0  # Initial heading

    # Setup policy services
    ts = TopologyService()
    sfs = SpatialFeatureService(ts, k_neighbors=8)
    sps = SpikingPolicyService(sfs)
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

    # Storage for visualization
    trajectory = []
    actions = []
    decisions = []
    hd_estimates = []
    hd_errors = []
    grid_estimates = []
    grid_drift = []
    grid_norms = []
    hd_activity_peaks = []
    true_headings = []

    print("\nRunning simulation...")
    for step in range(num_steps):
        # Create observation with current position and heading
        obs = np.array([position[0], position[1], heading], dtype=float)

        # Update bat navigation controller
        controller.step(obs, dt)

        # Extract HD and grid statistics
        hd_estimate = controller.hd_attractor.estimate_heading()
        grid_estimate = controller.grid_attractor.estimate_position()
        grid_drift_val = controller.grid_attractor.drift_metric()
        grid_activity = controller.grid_attractor.activity()
        grid_norm = np.linalg.norm(grid_activity)
        hd_activity = controller.hd_attractor.activity()
        hd_peak_idx = np.argmax(hd_activity)
        hd_peak_activity = hd_activity[hd_peak_idx]

        # Compute HD error (wrapped angle difference)
        hd_error = np.abs(((hd_estimate - heading + np.pi) % (2 * np.pi)) - np.pi)

        # Store statistics
        trajectory.append(position.copy())
        true_headings.append(heading)
        hd_estimates.append(hd_estimate)
        hd_errors.append(hd_error)
        grid_estimates.append(grid_estimate.copy())
        grid_drift.append(grid_drift_val)
        grid_norms.append(grid_norm)
        hd_activity_peaks.append(hd_peak_activity)

        # Update topology service periodically
        if step % 10 == 0:
            ts.update_from_controller(controller)

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

        # Apply action (simple kinematic model)
        v, omega = safe_cmd.cmd
        heading += omega * dt
        position[0] += v * np.cos(heading) * dt
        position[1] += v * np.sin(heading) * dt

        # Clip to environment bounds
        position[0] = np.clip(position[0], 0.0, env.width)
        position[1] = np.clip(position[1], 0.0, env.height)

        actions.append((v, omega))
        decisions.append(decision)

        # Print progress with HD/grid stats
        if step % 50 == 0:
            dist_to_goal = np.linalg.norm(np.array(goal.position) - position)
            print(
                f"Step {step:3d}: pos=({position[0]:.2f}, {position[1]:.2f}), "
                f"dist={dist_to_goal:.2f}m, "
                f"HD_err={np.degrees(hd_error):.1f}°, "
                f"grid_drift={grid_drift_val:.4f}, "
                f"grid_norm={grid_norm:.2f}"
            )

    # Final statistics
    trajectory = np.array(trajectory)
    hd_estimates = np.array(hd_estimates)
    hd_errors = np.array(hd_errors)
    grid_estimates = np.array(grid_estimates)
    grid_drift = np.array(grid_drift)
    grid_norms = np.array(grid_norms)
    true_headings = np.array(true_headings)

    final_dist = np.linalg.norm(np.array(goal.position) - trajectory[-1])
    mean_hd_error = np.mean(hd_errors)
    mean_grid_drift = np.mean(grid_drift)
    mean_grid_norm = np.mean(grid_norms)

    print(f"\nSimulation complete!")
    print(f"Final distance to goal: {final_dist:.3f}m")
    print(f"Graph nodes: {graph_snapshot.meta.epoch_id}")
    print(f"Graph edges: {len(graph_snapshot.E)}")
    print(f"\nHD Statistics:")
    print(f"  Mean HD error: {np.degrees(mean_hd_error):.2f}°")
    print(f"  Max HD error: {np.degrees(np.max(hd_errors)):.2f}°")
    print(f"\nGrid Statistics:")
    print(f"  Mean grid drift: {mean_grid_drift:.4f}")
    print(f"  Mean grid activity norm: {mean_grid_norm:.2f}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

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

    # HD error plot
    ax = axes[0, 2]
    ax.plot(time_steps, np.degrees(hd_errors), "g-", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HD Error (degrees)")
    ax.set_title(f"HD Estimation Error (mean: {np.degrees(mean_hd_error):.1f}°)")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=np.degrees(mean_hd_error), color="r", linestyle="--", alpha=0.5, label="Mean")
    ax.legend()

    # HD estimate vs true heading
    ax = axes[1, 0]
    ax.plot(time_steps, np.degrees(true_headings), "b-", alpha=0.6, label="True heading")
    ax.plot(time_steps, np.degrees(hd_estimates), "r--", alpha=0.6, label="HD estimate")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading (degrees)")
    ax.set_title("Head Direction Tracking")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Grid drift metric
    ax = axes[1, 1]
    ax.plot(time_steps, grid_drift, "purple", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Drift Metric")
    ax.set_title(f"Grid Activity Drift (mean: {mean_grid_drift:.4f})")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=mean_grid_drift, color="r", linestyle="--", alpha=0.5, label="Mean")
    ax.legend()

    # Grid activity norm
    ax = axes[1, 2]
    ax.plot(time_steps, grid_norms, "orange", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity Norm")
    ax.set_title(f"Grid Activity Norm (mean: {mean_grid_norm:.2f})")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=mean_grid_norm, color="r", linestyle="--", alpha=0.5, label="Mean")
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/policy_demo.png", dpi=150)
    print("\nPlot saved to results/policy_demo.png")

    # Show graph if available
    if len(graph_snapshot.E) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # Plot graph
        for edge in graph_snapshot.E:
            u_pos = next(n.position for n in graph_snapshot.V if n.node_id == edge.u)
            v_pos = next(n.position for n in graph_snapshot.V if n.node_id == edge.v)
            ax.plot([u_pos[0], v_pos[0]], [u_pos[1], v_pos[1]], "k-", alpha=0.3, linewidth=0.5)

        # Plot nodes
        for node in graph_snapshot.V:
            ax.plot(node.position[0], node.position[1], "bo", markersize=3)

        # Plot trajectory
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
        plt.savefig("results/policy_demo_graph.png", dpi=150)
        print("Graph plot saved to results/policy_demo_graph.png")


if __name__ == "__main__":
    main()

