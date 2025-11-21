"""Head-direction continuous attractor network utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to the range [-π, π)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class HeadDirectionConfig:
    """Configuration for :class:`HeadDirectionAttractor`."""

    num_neurons: int = 60
    tau: float = 0.05
    gamma: float = 1.0
    weight_sigma: float = 0.4
    activation: Optional[Callable[[np.ndarray], np.ndarray]] = None


class HeadDirectionAttractor:
    """Implements a 1D ring-attractor for head-direction coding."""

    def __init__(
        self,
        config: Optional[HeadDirectionConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config or HeadDirectionConfig()
        if self.config.num_neurons <= 0:
            raise ValueError("num_neurons must be positive.")
        if self.config.tau <= 0:
            raise ValueError("tau must be positive.")
        if self.config.weight_sigma <= 0:
            raise ValueError("weight_sigma must be positive.")

        self.activation: Callable[[np.ndarray], np.ndarray] = (
            self.config.activation or np.tanh
        )
        self.rng = rng or np.random.default_rng()
        self.preferred_angles = np.linspace(
            0.0, 2.0 * np.pi, self.config.num_neurons, endpoint=False
        )
        self.weights = self._build_ring_weights()
        self.state = np.zeros(self.config.num_neurons, dtype=float)
        self._estimated_heading = 0.0

    def _build_ring_weights(self) -> np.ndarray:
        """Create distance-dependent recurrent weights on a ring."""
        diffs = self.preferred_angles[:, None] - self.preferred_angles[None, :]
        wrapped = _wrap_angle(diffs)
        excitatory = np.exp(-(wrapped ** 2) / (2.0 * self.config.weight_sigma**2))
        inhibitory = np.eye(self.config.num_neurons)
        return excitatory - inhibitory

    def reset(self) -> None:
        """Reset network state."""
        self.state.fill(0.0)
        self._estimated_heading = 0.0

    def step(
        self,
        omega: float,
        dt: float,
        external_drive: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Advance the attractor dynamics by a single time step."""
        if dt <= 0:
            raise ValueError("dt must be positive.")

        activation = self.activation(self.state)
        recurrent = self.weights @ activation

        heading = self.estimate_heading()
        directional_drive = self.config.gamma * omega * np.sin(
            _wrap_angle(self.preferred_angles - heading)
        )

        drive = recurrent + directional_drive
        if external_drive is not None:
            drive += external_drive

        delta = (-self.state + drive) / self.config.tau
        self.state += dt * delta
        return self.state.copy()

    def estimate_heading(self) -> float:
        """Return the decoded heading using population vector decoding."""
        firing = np.maximum(self.activation(self.state), 1e-6)
        complex_sum = np.sum(firing * np.exp(1j * self.preferred_angles))
        if np.abs(complex_sum) < 1e-12:
            return self._estimated_heading
        self._estimated_heading = float(np.angle(complex_sum))
        return self._estimated_heading

    def inject_cue(self, theta: float, gain: float = 1.0) -> None:
        """Inject a sensory cue to realign the activity bump."""
        wrapped = _wrap_angle(theta - self.preferred_angles)
        bump = np.exp(-(wrapped ** 2) / (2.0 * self.config.weight_sigma**2))
        self.state += gain * bump

    def activity(self) -> np.ndarray:
        """Return the current firing activity (non-negative)."""
        return np.maximum(self.activation(self.state), 0.0)

