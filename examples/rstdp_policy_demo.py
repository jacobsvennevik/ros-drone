#!/usr/bin/env python3
"""Demo of R-STDP policy learning (biologically plausible, no backpropagation).

This demonstrates:
- R-STDP network initialization
- Online learning during navigation
- Reward-based weight updates
- Biologically plausible learning rules
"""
from __future__ import annotations

import numpy as np
from hippocampus_core.policy import (
    TopologyService,
    SpatialFeatureService,
    SpikingPolicyService,
    RSTDPPolicySNN,
    RSTDPConfig,
    NavigationRewardFunction,
    RewardConfig,
    RobotState,
    Mission,
    MissionGoal,
    GoalType,
    PointGoal,
    Pose,
)

def main():
    """Run R-STDP policy demo."""
    print("R-STDP Policy Demo")
    print("=" * 60)
    print("Biologically plausible learning (no backpropagation)")
    print()

    # Initialize services
    ts = TopologyService()
    sfs = SpatialFeatureService(ts, k_neighbors=8, is_3d=False)

    # Create R-STDP network
    # Get feature dimension from feature service (will be computed on first use)
    # For now, use a typical 2D feature dimension
    feature_dim = 44  # Typical 2D feature dimension
    rstdp_config = RSTDPConfig(
        feature_dim=feature_dim,
        hidden_size=64,
        output_size=2,  # 2D: [v, omega]
        learning_rate=5e-3,
    )
    rstdp_model = RSTDPPolicySNN(rstdp_config)

    # Create reward function
    reward_config = RewardConfig(
        goal_reward_gain=2.0,
        goal_reached_reward=10.0,
        smoothness_reward=0.1,
    )
    reward_function = NavigationRewardFunction(reward_config)

    # Create policy service with R-STDP
    policy = SpikingPolicyService(
        feature_service=sfs,
        config={"max_linear": 0.3, "max_angular": 1.0},
        rstdp_model=rstdp_model,
        reward_function=reward_function,
    )

    # Create mission
    mission = Mission(
        goal=MissionGoal(
            type=GoalType.POINT,
            value=PointGoal(position=(0.9, 0.9)),
        )
    )

    # Simulate navigation
    print("Simulating navigation with online R-STDP learning...")
    print()

    # Initial robot state
    robot_state = RobotState(
        pose=Pose(x=0.1, y=0.1, heading=0.0),
        velocity=(0.0, 0.0),
    )

    total_reward = 0.0
    num_steps = 100

    for step in range(num_steps):
        # Build features
        features, local_context = sfs.build_features(
            robot_state=robot_state,
            mission=mission,
        )

        # Make decision (R-STDP forward pass)
        decision = policy.decide(features, local_context, dt=0.05, mission=mission)

        # Compute reward (for learning)
        reward = reward_function.compute(
            robot_state=robot_state,
            action=decision,
            mission=mission,
            features=features,
            dt=0.05,
        )
        total_reward += reward

        # Update weights (R-STDP learning)
        rstdp_model.update_weights(reward)

        # Update robot state (simple simulation)
        action = decision.action_proposal
        v = action.v
        omega = action.omega

        # Simple dynamics
        robot_state.pose.heading += omega * 0.05
        robot_state.pose.x += v * np.cos(robot_state.pose.heading) * 0.05
        robot_state.pose.y += v * np.sin(robot_state.pose.heading) * 0.05

        # Print progress
        if step % 20 == 0:
            goal_pos = mission.goal.value.position
            distance = np.sqrt(
                (robot_state.pose.x - goal_pos.x) ** 2
                + (robot_state.pose.y - goal_pos.y) ** 2
            )
            print(
                f"Step {step:3d} | "
                f"Position: ({robot_state.pose.x:.2f}, {robot_state.pose.y:.2f}) | "
                f"Distance to goal: {distance:.3f} | "
                f"Reward: {reward:.3f}"
            )

    print()
    print("=" * 60)
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward / num_steps:.3f}")
    print()
    print("R-STDP learning complete!")
    print("Weights have been updated using biologically plausible local rules.")
    print("No backpropagation was used.")


if __name__ == "__main__":
    main()

