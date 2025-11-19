"""Environment and agent abstractions for the 2D continuous arena."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CircularObstacle:
    """Represents a circular obstacle in the environment."""

    center_x: float
    center_y: float
    radius: float

    def contains(self, position: Tuple[float, float]) -> bool:
        """Return True if the position is inside this obstacle."""
        x, y = position
        dx = x - self.center_x
        dy = y - self.center_y
        return (dx * dx + dy * dy) <= (self.radius * self.radius)

    def distance_to_edge(self, position: Tuple[float, float]) -> float:
        """Return signed distance to obstacle edge (negative if inside, positive if outside)."""
        x, y = position
        dx = x - self.center_x
        dy = y - self.center_y
        distance_to_center = np.sqrt(dx * dx + dy * dy)
        return distance_to_center - self.radius


@dataclass(frozen=True)
class Bounds:
    """Represents rectangular bounds in 2D space."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


class Environment:
    """Continuous 2D environment with rectangular bounds and optional obstacles."""

    def __init__(
        self,
        width: float = 1.0,
        height: float = 1.0,
        obstacles: Optional[List[CircularObstacle]] = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Environment dimensions must be positive.")
        self._bounds = Bounds(0.0, width, 0.0, height)
        self._obstacles = obstacles or []

        # Validate obstacles are within bounds
        for obs in self._obstacles:
            if obs.radius <= 0:
                raise ValueError("Obstacle radius must be positive.")
            # Check if obstacle center is within bounds
            if not self._bounds.min_x <= obs.center_x <= self._bounds.max_x:
                raise ValueError(f"Obstacle center_x {obs.center_x} is outside bounds.")
            if not self._bounds.min_y <= obs.center_y <= self._bounds.max_y:
                raise ValueError(f"Obstacle center_y {obs.center_y} is outside bounds.")
            # Check if obstacle fits within bounds
            if (
                obs.center_x - obs.radius < self._bounds.min_x
                or obs.center_x + obs.radius > self._bounds.max_x
                or obs.center_y - obs.radius < self._bounds.min_y
                or obs.center_y + obs.radius > self._bounds.max_y
            ):
                raise ValueError(f"Obstacle at ({obs.center_x}, {obs.center_y}) with radius {obs.radius} extends outside bounds.")

    @property
    def obstacles(self) -> List[CircularObstacle]:
        """Return the list of obstacles in the environment."""
        return self._obstacles.copy()

    @property
    def bounds(self) -> Bounds:
        """Return the environment bounds."""

        return self._bounds

    def contains(self, position: Tuple[float, float]) -> bool:
        """Return True if the given position lies inside the environment (excluding obstacles)."""

        x, y = position
        # Check bounds
        if not (
            self._bounds.min_x <= x <= self._bounds.max_x
            and self._bounds.min_y <= y <= self._bounds.max_y
        ):
            return False

        # Check obstacles
        for obstacle in self._obstacles:
            if obstacle.contains(position):
                return False

        return True

    def contains_with_obstacles(self, position: Tuple[float, float]) -> bool:
        """Return True if position is valid (inside bounds and not in obstacles)."""
        return self.contains(position)

    def clip(self, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clip the position to remain within bounds.

        Returns
        -------
        clipped : np.ndarray
            The clipped position.
        mask : np.ndarray
            Boolean mask indicating which axes were clipped.
        """

        clipped = np.empty_like(position)
        mask = np.zeros_like(position, dtype=bool)

        for i, (value, min_val, max_val) in enumerate(
            zip(
                position,
                (self._bounds.min_x, self._bounds.min_y),
                (self._bounds.max_x, self._bounds.max_y),
            )
        ):
            if value < min_val:
                clipped[i] = min_val
                mask[i] = True
            elif value > max_val:
                clipped[i] = max_val
                mask[i] = True
            else:
                clipped[i] = value
        return clipped, mask


class Agent:
    """Point-mass agent performing a random walk inside an environment."""

    def __init__(
        self,
        environment: Environment,
        position: Optional[Tuple[float, float]] = None,
        base_speed: float = 0.2,
        max_speed: float = 0.4,
        velocity_noise: float = 0.2,
        random_state: Optional[np.random.Generator] = None,
    ) -> None:
        if base_speed <= 0:
            raise ValueError("base_speed must be positive.")
        if max_speed <= 0:
            raise ValueError("max_speed must be positive.")
        if velocity_noise < 0:
            raise ValueError("velocity_noise must be non-negative.")

        self.environment = environment
        self.base_speed = base_speed
        self.max_speed = max_speed
        self.velocity_noise = velocity_noise
        self.random = random_state or np.random.default_rng()

        bounds = self.environment.bounds
        if position is None:
            start_position = np.array(
                [
                    0.5 * (bounds.min_x + bounds.max_x),
                    0.5 * (bounds.min_y + bounds.max_y),
                ],
                dtype=float,
            )
        else:
            start_position = np.asarray(position, dtype=float)
            if not self.environment.contains(tuple(start_position)):
                raise ValueError("Initial position must lie within the environment bounds.")

        self.position = start_position
        self.velocity = self._initialize_velocity()

    def _initialize_velocity(self) -> np.ndarray:
        angle = self.random.uniform(0.0, 2.0 * np.pi)
        vx = self.base_speed * np.cos(angle)
        vy = self.base_speed * np.sin(angle)
        return np.array([vx, vy], dtype=float)

    def step(self, dt: float) -> np.ndarray:
        """Advance the agent state by dt seconds and return the new position."""

        if dt <= 0:
            raise ValueError("dt must be positive.")

        # Randomly perturb the velocity.
        noise = self.random.normal(scale=self.velocity_noise, size=2)
        self.velocity += noise

        speed = np.linalg.norm(self.velocity)
        if speed == 0:
            # Reinitialize to avoid getting stuck.
            self.velocity = self._initialize_velocity()
        else:
            # Keep velocity within desired speed range.
            capped_speed = np.clip(speed, self.base_speed, self.max_speed)
            self.velocity = self.velocity * (capped_speed / speed)

        proposed_position = self.position + self.velocity * dt
        clipped_position, clipped_axes = self.environment.clip(proposed_position)

        # Check for obstacle collisions
        if not self.environment.contains(tuple(clipped_position)):
            # Position is in an obstacle, need to avoid it
            # Find closest obstacle and push away from it
            min_distance = float("inf")
            closest_obstacle = None

            for obstacle in self.environment.obstacles:
                distance = obstacle.distance_to_edge(tuple(clipped_position))
                if distance < min_distance:
                    min_distance = distance
                    closest_obstacle = obstacle

            if closest_obstacle is not None:
                # Push away from obstacle center
                dx = clipped_position[0] - closest_obstacle.center_x
                dy = clipped_position[1] - closest_obstacle.center_y
                dist_to_center = np.sqrt(dx * dx + dy * dy)

                if dist_to_center > 0:
                    # Normalize and push to edge plus small margin
                    push_distance = closest_obstacle.radius + 0.01
                    clipped_position[0] = closest_obstacle.center_x + (dx / dist_to_center) * push_distance
                    clipped_position[1] = closest_obstacle.center_y + (dy / dist_to_center) * push_distance

                # Deflect velocity away from obstacle
                normal = np.array([dx, dy], dtype=float)
                if np.linalg.norm(normal) > 0:
                    normal = normal / np.linalg.norm(normal)
                    # Reflect velocity component along normal
                    vel_normal = np.dot(self.velocity, normal)
                    if vel_normal < 0:  # Moving towards obstacle
                        self.velocity = self.velocity - 2 * vel_normal * normal

        # Reflect velocity components for any boundary hits.
        for axis, clipped in enumerate(clipped_axes):
            if clipped:
                self.velocity[axis] *= -1.0

        # Ensure final position is valid
        if not self.environment.contains(tuple(clipped_position)):
            # Fallback: stay at current position if can't find valid position
            clipped_position = self.position.copy()

        self.position = clipped_position
        return self.position.copy()
