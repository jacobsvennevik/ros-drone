"""Controller that combines HD/grid attractors with conjunctive place cells."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from ..calibration.phase_optimizer import PhaseOptimizer
from ..conjunctive_place_cells import (
    ConjunctivePlaceCellConfig,
    ConjunctivePlaceCellPopulation,
)
from ..grid_cells import GridAttractor, GridAttractorConfig
from ..head_direction import HeadDirectionAttractor, HeadDirectionConfig
from .place_cell_controller import PlaceCellController, PlaceCellControllerConfig


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class BatNavigationControllerConfig(PlaceCellControllerConfig):
    """Extends :class:`PlaceCellControllerConfig` with HD/grid parameters."""

    hd_num_neurons: int = 60
    hd_tau: float = 0.05
    hd_gamma: float = 1.0
    hd_weight_sigma: float = 0.4
    grid_size: Tuple[int, int] = (15, 15)
    grid_tau: float = 0.05
    grid_velocity_gain: float = 1.0
    conj_weight_scale: float = 0.4
    conj_bias: float = 0.0
    calibration_history: int = 256
    calibration_interval: int = 200
    adaptive_calibration: bool = False
    calibration_drift_threshold: float = 0.1  # Trigger calibration if drift exceeds this
    normalize_mode: Literal["subtractive", "divisive"] = "subtractive"
    """Global inhibition mode for both grid and HD attractors."""


class BatNavigationController(PlaceCellController):
    """Controller that fuses HD, grid, and place layers before topology learning."""

    def __init__(
        self,
        environment,
        config: Optional[BatNavigationControllerConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config or BatNavigationControllerConfig()
        super().__init__(environment=environment, config=self.config, rng=rng)

        hd_config = HeadDirectionConfig(
            num_neurons=self.config.hd_num_neurons,
            tau=self.config.hd_tau,
            gamma=self.config.hd_gamma,
            weight_sigma=self.config.hd_weight_sigma,
            normalize_mode=self.config.normalize_mode,
        )
        grid_config = GridAttractorConfig(
            size=self.config.grid_size,
            tau=self.config.grid_tau,
            velocity_gain=self.config.grid_velocity_gain,
            normalize_mode=self.config.normalize_mode,
        )
        conj_config = ConjunctivePlaceCellConfig(
            num_place_cells=self.config.num_place_cells,
            grid_dim=self.config.grid_size[0] * self.config.grid_size[1],
            head_direction_dim=self.config.hd_num_neurons,
            weight_scale=self.config.conj_weight_scale,
            bias=self.config.conj_bias,
        )

        self.hd_attractor = HeadDirectionAttractor(config=hd_config, rng=self.rng)
        self.grid_attractor = GridAttractor(config=grid_config, rng=self.rng)
        self.conjunctive_cells = ConjunctivePlaceCellPopulation(
            conj_config, rng=self.rng
        )
        self.calibration = PhaseOptimizer(max_history=self.config.calibration_history)
        self._prev_position: Optional[np.ndarray] = None
        self._prev_heading: Optional[float] = None
        self._steps_since_calibration = 0

    def reset(self) -> None:
        super().reset()
        self.hd_attractor.reset()
        self.grid_attractor.reset()
        self.calibration.clear()
        self._prev_position = None
        self._prev_heading = None
        self._steps_since_calibration = 0

    def _compute_rates(self, observation: np.ndarray, dt: float) -> np.ndarray:
        if observation.shape[0] < 3:
            raise ValueError(
                "BatNavigationController requires observations containing (x, y, θ)."
            )

        position = observation[:2]
        theta_raw = observation[2]
        
        # Validate heading: check for NaN/Inf
        if not np.isfinite(theta_raw):
            if self._prev_heading is not None:
                # Fallback to last valid heading
                theta = self._prev_heading
            else:
                # Fallback to zero if no previous heading
                theta = 0.0
        else:
            theta = float(theta_raw)

        omega = 0.0
        if self._prev_heading is not None:
            omega = _wrap_angle(theta - self._prev_heading) / dt
        self.hd_attractor.step(omega=omega, dt=dt)

        velocity = np.zeros(2, dtype=float)
        if self._prev_position is not None:
            velocity = (position - self._prev_position) / dt
        self.grid_attractor.step(velocity=velocity, dt=dt)

        grid_activity = self.grid_attractor.activity().ravel()
        hd_activity = self.hd_attractor.activity()
        rates = self.conjunctive_cells.compute_rates(grid_activity, hd_activity)

        self._prev_position = position.copy()
        self._prev_heading = theta
        self._maybe_calibrate(position, theta)

        return rates

    def _maybe_calibrate(self, position: np.ndarray, theta: float) -> None:
        self.calibration.add_sample(
            position=position,
            heading=theta,
            hd_estimate=self.hd_attractor.estimate_heading(),
            grid_estimate=self.grid_attractor.estimate_position(),
        )

        self._steps_since_calibration += 1
        
        # Adaptive calibration: trigger if drift exceeds threshold
        if self.config.adaptive_calibration:
            grid_drift = self.grid_attractor.drift_metric()
            hd_activity = self.hd_attractor.activity()
            hd_norm = np.linalg.norm(hd_activity)
            # Trigger if grid drift or HD activity is unstable
            should_calibrate = (grid_drift > self.config.calibration_drift_threshold or
                              hd_norm > self.config.calibration_drift_threshold * 10)
        else:
            # Fixed interval calibration
            should_calibrate = self._steps_since_calibration >= self.config.calibration_interval
        
        if not should_calibrate:
            return

        correction = self.calibration.estimate_correction()
        if correction is None:
            return

        new_heading = self.hd_attractor.estimate_heading() + correction.heading_delta
        self.hd_attractor.inject_cue(new_heading, gain=2.0)

        shift = np.round(correction.grid_translation).astype(int)
        min_shift = np.array(
            [-self.config.grid_size[0], -self.config.grid_size[1]], dtype=int
        )
        max_shift = np.array(
            [self.config.grid_size[0], self.config.grid_size[1]], dtype=int
        )
        shift = np.clip(shift, min_shift, max_shift)
        self.grid_attractor.shift_phase(shift)

        self.calibration.clear()
        self._steps_since_calibration = 0

