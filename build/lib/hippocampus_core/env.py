"""Environment and agent abstractions for the 2D continuous arena."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


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
    """Continuous 2D environment with rectangular bounds."""

    def __init__(self, width: float = 1.0, height: float = 1.0) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Environment dimensions must be positive.")
        self._bounds = Bounds(0.0, width, 0.0, height)

    @property
    def bounds(self) -> Bounds:
        """Return the environment bounds."""

        return self._bounds

    def contains(self, position: Tuple[float, float]) -> bool:
        """Return True if the given position lies inside the environment."""

        x, y = position
        return (
            self._bounds.min_x <= x <= self._bounds.max_x
            and self._bounds.min_y <= y <= self._bounds.max_y
        )

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

        # Reflect velocity components for any boundary hits.
        for axis, clipped in enumerate(clipped_axes):
            if clipped:
                self.velocity[axis] *= -1.0

        self.position = clipped_position
        return self.position.copy()
