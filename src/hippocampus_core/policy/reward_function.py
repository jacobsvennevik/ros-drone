"""Reward functions for R-STDP policy learning.

Reward functions compute scalar rewards from robot state, actions, and mission goals.
These rewards are used to modulate STDP learning in the R-STDP network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .data_structures import (
    RobotState,
    Mission,
    PolicyDecision,
    FeatureVector,
    GoalType,
    LocalContext,
    GraphSnapshot,
)


@dataclass
class RewardConfig:
    """Configuration for reward function.

    Attributes
    ----------
    goal_reward_gain:
        Reward per meter of progress toward goal.
    goal_reached_reward:
        Large reward when goal is reached.
    goal_reached_tolerance:
        Distance threshold for considering goal reached (meters).
    obstacle_penalty:
        Penalty when too close to obstacles.
    obstacle_safety_margin:
        Safety margin around obstacles (meters).
    collision_penalty:
        Large penalty for collisions.
    smoothness_reward:
        Reward for smooth actions (penalizes large angular velocities).
    angular_penalty_gain:
        Penalty coefficient for angular velocity.
    forward_progress_gain:
        Reward for forward progress (independent of goal direction).
    reward_clip:
        Maximum absolute reward value (for stability).
    reward_scale:
        Overall scaling factor for rewards.
    """

    # Goal rewards
    goal_reward_gain: float = 2.0
    goal_reached_reward: float = 10.0
    goal_reached_tolerance: float = 0.1

    # Obstacle avoidance
    obstacle_penalty: float = -1.0
    obstacle_safety_margin: float = 0.2
    collision_penalty: float = -5.0

    # Action smoothness
    smoothness_reward: float = 0.1
    angular_penalty_gain: float = 0.2

    # General progress
    forward_progress_gain: float = 0.5

    # Reward shaping
    reward_clip: float = 10.0
    reward_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.goal_reached_tolerance <= 0:
            raise ValueError("goal_reached_tolerance must be positive")
        if self.obstacle_safety_margin < 0:
            raise ValueError("obstacle_safety_margin must be non-negative")
        if self.reward_clip <= 0:
            raise ValueError("reward_clip must be positive")


class NavigationRewardFunction:
    """Reward function for navigation tasks.

    Computes rewards based on:
    - Goal progress (distance reduction)
    - Obstacle avoidance (distance to obstacles)
    - Action smoothness (penalize large angular velocities)
    - Goal reaching (large reward when goal reached)
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        """Initialize reward function.

        Parameters
        ----------
        config:
            Reward configuration. Uses defaults if None.
        """
        self.config = config or RewardConfig()
        self._last_position: Optional[np.ndarray] = None
        self._last_goal_distance: Optional[float] = None
        self._last_node_id: Optional[int] = None
        self._explored_nodes: set[int] = set()

    def reset(self) -> None:
        """Reset reward function state (for new episode)."""
        self._last_position = None
        self._last_goal_distance = None
        self._last_node_id = None
        self._explored_nodes.clear()

    def compute(
        self,
        robot_state: RobotState,
        action: PolicyDecision,
        mission: Mission,
        features: Optional[FeatureVector] = None,
        local_context: Optional[LocalContext] = None,
        dt: float = 0.05,
    ) -> float:
        """Compute reward for current state and action.

        Parameters
        ----------
        robot_state:
            Current robot state (position, heading, etc.).
        action:
            Policy decision (action taken).
        mission:
            Mission with goal information.
        features:
            Optional feature vector (for obstacle information).
        local_context:
            Optional local context (for graph snapshot access).
        dt:
            Time step duration.

        Returns
        -------
        float
            Scalar reward (positive = good, negative = bad).
        """
        reward = 0.0
        # Extract position from pose tuple: (x, y, yaw) or (x, y, z, yaw, pitch)
        # Handle both tuple access and object attribute access for compatibility
        if isinstance(robot_state.pose, tuple):
            position = np.array([robot_state.pose[0], robot_state.pose[1]], dtype=np.float32)
        else:
            # Handle object with x, y attributes (for compatibility with test code)
            position = np.array([robot_state.pose.x, robot_state.pose.y], dtype=np.float32)

        # 1. Goal progress reward
        goal_reward = self._compute_goal_reward(
            position, mission, robot_state, local_context, dt
        )
        reward += goal_reward

        # 2. Obstacle avoidance
        obstacle_reward = self._compute_obstacle_reward(position, features, dt)
        reward += obstacle_reward

        # 3. Action smoothness
        smoothness_reward = self._compute_smoothness_reward(action)
        reward += smoothness_reward

        # 4. Forward progress (general movement)
        progress_reward = self._compute_progress_reward(position, dt)
        reward += progress_reward

        # Update state
        self._last_position = position.copy()
        if mission.goal.type == GoalType.POINT:
            goal_pos = mission.goal.value.position
            goal_distance = np.linalg.norm(position - np.array(goal_pos[:2]))
            self._last_goal_distance = goal_distance
        elif mission.goal.type == GoalType.NODE:
            # Track current node for node goal progress
            self._last_node_id = robot_state.current_node

        # Clip and scale
        reward = np.clip(reward, -self.config.reward_clip, self.config.reward_clip)
        return float(reward * self.config.reward_scale)

    def _compute_goal_reward(
        self,
        position: np.ndarray,
        mission: Mission,
        robot_state: RobotState,
        local_context: Optional[LocalContext],
        dt: float,
    ) -> float:
        """Compute reward based on goal progress.

        Parameters
        ----------
        position:
            Current robot position (2D or 3D).
        mission:
            Mission with goal.
        robot_state:
            Current robot state (for node ID access).
        local_context:
            Optional local context (for graph snapshot access).
        dt:
            Time step.

        Returns
        -------
        float
            Goal-related reward.
        """
        if mission.goal.type == GoalType.POINT:
            goal_pos = mission.goal.value.position
            goal_position = np.array(goal_pos[:2], dtype=np.float32)
            
            # Handle 3D if present
            if len(goal_pos) >= 3:
                goal_position = np.array(goal_pos[:3], dtype=np.float32)
                current_pos = np.array([position[0], position[1], 0.0], dtype=np.float32)
            else:
                current_pos = position[:2]

            distance = float(np.linalg.norm(current_pos - goal_position))

            # Check if goal reached
            if distance < self.config.goal_reached_tolerance:
                return self.config.goal_reached_reward

            # Progress reward: reward for reducing distance to goal
            if self._last_goal_distance is not None:
                progress = self._last_goal_distance - distance
                return self.config.goal_reward_gain * progress
            else:
                # First step: small reward for being closer than initial distance
                return 0.0

        elif mission.goal.type == GoalType.NODE:
            # Node goal: reward for getting closer to target node
            node_goal = mission.goal.value
            target_node_id = node_goal.node_id

            # Check if goal reached (at target node)
            if robot_state.current_node == target_node_id:
                return self.config.goal_reached_reward

            # Get graph snapshot to find node position
            graph_snapshot = None
            if local_context and local_context.graph_snapshot:
                graph_snapshot = local_context.graph_snapshot
            elif local_context and hasattr(local_context, "graph_snapshot"):
                graph_snapshot = local_context.graph_snapshot

            if graph_snapshot and graph_snapshot.V:
                # Find target node position
                target_node = next(
                    (n for n in graph_snapshot.V if n.node_id == target_node_id), None
                )
                if target_node:
                    target_position = np.array(target_node.position[:2], dtype=np.float32)
                    distance = float(np.linalg.norm(position - target_position))

                    # Check if goal reached (within tolerance)
                    if distance < node_goal.tolerance:
                        return self.config.goal_reached_reward

                    # Progress reward: reward for reducing distance to node
                    if self._last_goal_distance is not None:
                        progress = self._last_goal_distance - distance
                        return self.config.goal_reward_gain * progress
                    else:
                        self._last_goal_distance = distance
                        return 0.0

            # If we can't find the node, return small negative reward
            return -0.1

        elif mission.goal.type == GoalType.REGION:
            # Region goal: reward for entering target region
            # TODO: Implement when RegionGoal is fully defined in data_structures
            # For now, return 0 (stub)
            return 0.0

        elif mission.goal.type == GoalType.SEQUENTIAL:
            # Sequential goal: reward for progress through waypoint sequence
            # TODO: Implement when SequentialGoal is fully defined in data_structures
            # For now, return 0 (stub)
            return 0.0

        elif mission.goal.type == GoalType.EXPLORE:
            # Exploration goal: reward for visiting new nodes
            if robot_state.current_node is not None:
                if robot_state.current_node not in self._explored_nodes:
                    # New node discovered - positive reward
                    self._explored_nodes.add(robot_state.current_node)
                    return 1.0  # Reward for exploration
                else:
                    # Already explored - small negative reward to encourage new exploration
                    return -0.05
            return 0.0

        return 0.0

    def _compute_obstacle_reward(
        self,
        position: np.ndarray,
        features: Optional[FeatureVector],
        dt: float,
    ) -> float:
        """Compute reward based on obstacle avoidance.

        Parameters
        ----------
        position:
            Current robot position.
        features:
            Feature vector (contains safety features with obstacle information).
        dt:
            Time step.

        Returns
        -------
        float
            Obstacle-related reward (negative = too close).
        """
        if features is None or not features.safety:
            return 0.0

        # Safety features are normalized distances (0-1):
        # - 1.0 = no obstacle (safe)
        # - 0.0 = obstacle very close (dangerous)
        # Safety features: [front, left, right, back] for 2D
        #                  [front, left, right, up, down] for 3D
        safety_features = np.array(features.safety, dtype=np.float32)
        
        # Threshold for "too close" - normalized value below which we apply penalty
        # Assuming max_range=10.0 from feature service, safety_margin=0.2m means normalized ~0.02
        # Use a normalized threshold of 0.1 (roughly 1.0m at max_range=10.0)
        danger_threshold = 0.1
        collision_threshold = 0.05  # Very close (roughly 0.5m at max_range=10.0)

        reward = 0.0

        # Check each direction for obstacles
        min_safety = float(np.min(safety_features))

        # Collision penalty (very close to obstacle)
        if min_safety < collision_threshold:
            reward += self.config.collision_penalty
        # Danger penalty (too close to obstacle)
        elif min_safety < danger_threshold:
            # Penalty proportional to how close we are
            # More penalty as safety value approaches 0
            danger_factor = (danger_threshold - min_safety) / danger_threshold
            reward += self.config.obstacle_penalty * danger_factor

        return reward

    def _compute_smoothness_reward(self, action: PolicyDecision) -> float:
        """Compute reward for smooth actions.

        Parameters
        ----------
        action:
            Policy decision with action.

        Returns
        -------
        float
            Smoothness reward (negative for large angular velocities).
        """
        # Penalize large angular velocities (encourages smooth turns)
        angular_penalty = self.config.angular_penalty_gain * abs(action.action.angular_z)
        return -angular_penalty

    def _compute_progress_reward(self, position: np.ndarray, dt: float) -> float:
        """Compute reward for general forward progress.

        Parameters
        ----------
        position:
            Current robot position.
        dt:
            Time step.

        Returns
        -------
        float
            Progress reward.
        """
        if self._last_position is None:
            return 0.0

        # Reward for any movement (encourages exploration)
        delta = position[:2] - self._last_position[:2]
        distance = float(np.linalg.norm(delta))
        return self.config.forward_progress_gain * distance

