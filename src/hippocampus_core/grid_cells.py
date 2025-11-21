"""Grid-cell continuous attractor utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np


@dataclass
class GridAttractorConfig:
    """Configuration for :class:`GridAttractor`."""

    size: Tuple[int, int] = (20, 20)
    tau: float = 0.05
    velocity_gain: float = 1.0
    activation: Optional[Callable[[np.ndarray], np.ndarray]] = None
    periodic: bool = True
    normalize_mode: Literal["subtractive", "divisive"] = "subtractive"
    """Global inhibition mode: 'subtractive' (mean subtraction) or 'divisive' (L2 normalization)."""


class GridAttractor:
    """Implements a 2D toroidal continuous attractor for grid coding."""

    def __init__(
        self,
        config: Optional[GridAttractorConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config or GridAttractorConfig()
        if self.config.tau <= 0:
            raise ValueError("tau must be positive.")
        if self.config.size[0] <= 0 or self.config.size[1] <= 0:
            raise ValueError("Grid size must be positive.")

        self.activation: Callable[[np.ndarray], np.ndarray] = (
            self.config.activation or np.tanh
        )
        self.rng = rng or np.random.default_rng()
        self.state = self.rng.normal(
            scale=1e-3, size=(self.config.size[0], self.config.size[1])
        )
        # Track previous position estimate for phase-space drift metric
        self._prev_position_estimate: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset attractor state."""
        self.state.fill(0.0)
        self._prev_position_estimate = None

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """Discrete Laplacian with optional periodic boundaries."""
        if self.config.periodic:
            shifts = [
                np.roll(field, 1, axis=0),
                np.roll(field, -1, axis=0),
                np.roll(field, 1, axis=1),
                np.roll(field, -1, axis=1),
            ]
            return sum(shifts) - 4 * field
        padded = np.pad(field, 1, mode="edge")
        lap = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4 * field
        )
        return lap

    def step(
        self,
        velocity: np.ndarray,
        dt: float,
        external_drive: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Advance the attractor with the provided planar velocity."""
        if dt <= 0:
            raise ValueError("dt must be positive.")
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (2,):
            raise ValueError("velocity must be a 2-vector.")

        activation = self.activation(self.state)
        recurrent = self._laplacian(activation)

        grad_y, grad_x = np.gradient(self.state)
        flow = velocity[0] * grad_x + velocity[1] * grad_y
        drive = recurrent + self.config.velocity_gain * flow

        if external_drive is not None:
            drive += external_drive

        delta = (-self.state + drive) / self.config.tau
        self.state += dt * delta
        
        # Global inhibition: prevent amplitude drift and maintain stability
        if self.config.normalize_mode == "divisive":
            # Divisive normalization: L2 norm normalization
            # This provides gain control and noise robustness (Burak & Fiete, 2009; Cueva & Wei, 2018)
            norm = np.linalg.norm(self.state)
            if norm > 1e-6:
                self.state /= norm
        else:
            # Subtractive normalization: mean subtraction (default)
            # This maintains attractor stability under floating-point arithmetic
            self.state -= self.state.mean()
        
        return self.state.copy()

    def shift_phase(self, delta: np.ndarray) -> None:
        """Roll the attractor to reflect an external phase correction."""
        delta = np.asarray(delta, dtype=int)
        if delta.shape != (2,):
            raise ValueError("delta must be shape (2,).")
        self.state = np.roll(self.state, shift=tuple(delta), axis=(0, 1))

    def activity(self) -> np.ndarray:
        """Return the current firing activity (non-negative)."""
        return np.maximum(self.activation(self.state), 0.0)

    def estimate_position(self) -> np.ndarray:
        """Decode a coarse spatial position from the attractor activity."""
        activity = self.activity()
        if np.allclose(activity.sum(), 0.0):
            pos = np.array([0.0, 0.0])
        else:
            total = activity.sum()
            grid_y = np.arange(self.config.size[0])
            grid_x = np.arange(self.config.size[1])
            y_expectation = (activity.sum(axis=1) * grid_y).sum() / total
            x_expectation = (activity.sum(axis=0) * grid_x).sum() / total
            pos = np.array([x_expectation, y_expectation])
        return pos

    def drift_metric(self) -> float:
        """Compute phase-space drift metric (distance-based, not amplitude-based).
        
        Returns the phase-space distance (in grid cells) since last position estimate.
        This should be called after estimate_position() to get the drift between
        consecutive calls. If no previous position is available, returns 0.0.
        
        This is more accurate than amplitude-based metrics for detecting coherent
        translations of the activity bump during path integration.
        """
        current_pos = self.estimate_position()
        
        if self._prev_position_estimate is None:
            # First call - store position and return 0 drift
            self._prev_position_estimate = current_pos.copy()
            return 0.0
        
        delta = current_pos - self._prev_position_estimate
        # Phase-space distance in grid cells
        distance = float(np.linalg.norm(delta))
        
        # Update previous position for next call
        self._prev_position_estimate = current_pos.copy()
        return distance

