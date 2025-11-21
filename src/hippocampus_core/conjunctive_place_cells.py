"""Conjunctive place-cell population combining grid and head-direction inputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ConjunctivePlaceCellConfig:
    """Configuration for :class:`ConjunctivePlaceCellPopulation`."""

    num_place_cells: int
    grid_dim: int
    head_direction_dim: int
    weight_scale: float = 0.5
    bias: float = 0.0


class ConjunctivePlaceCellPopulation:
    """Linear population that combines grid and head-direction activities."""

    def __init__(
        self,
        config: ConjunctivePlaceCellConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if config.num_place_cells <= 0:
            raise ValueError("num_place_cells must be positive.")
        if config.grid_dim <= 0 or config.head_direction_dim <= 0:
            raise ValueError("grid_dim and head_direction_dim must be positive.")

        self.config = config
        self.rng = rng or np.random.default_rng()
        scale = self.config.weight_scale
        self.grid_weights = self.rng.normal(
            scale=scale, size=(self.config.num_place_cells, self.config.grid_dim)
        )
        self.hd_weights = self.rng.normal(
            scale=scale,
            size=(self.config.num_place_cells, self.config.head_direction_dim),
        )
        self.bias = np.full(self.config.num_place_cells, self.config.bias, dtype=float)

    def compute_rates(
        self, grid_activity: np.ndarray, hd_activity: np.ndarray
    ) -> np.ndarray:
        """Compute conjunctive place-cell rates given upstream activities."""
        grid_activity = np.asarray(grid_activity, dtype=float).reshape(-1)
        hd_activity = np.asarray(hd_activity, dtype=float).reshape(-1)

        if grid_activity.shape[0] != self.config.grid_dim:
            raise ValueError(
                f"grid_activity must have shape ({self.config.grid_dim},); "
                f"got {grid_activity.shape}"
            )
        if hd_activity.shape[0] != self.config.head_direction_dim:
            raise ValueError(
                f"hd_activity must have shape ({self.config.head_direction_dim},); "
                f"got {hd_activity.shape}"
            )

        # Multiplicative modulation: grid × HD interaction
        # This matches biological conjunctive place cells (e.g., in bat MEC)
        # where grid and HD signals interact multiplicatively rather than additively
        grid_contribution = self.grid_weights @ grid_activity
        hd_contribution = self.hd_weights @ hd_activity
        
        # Normalize contributions before multiplication to prevent runaway amplitude
        # This preserves tuning-shape stability and prevents single-cell saturation
        grid_norm = np.linalg.norm(grid_contribution)
        hd_norm = np.linalg.norm(hd_contribution)
        if grid_norm > 1e-6:
            grid_contribution = grid_contribution / grid_norm
        if hd_norm > 1e-6:
            hd_contribution = hd_contribution / hd_norm
        
        # Bilinear interaction: element-wise product plus bias
        # This captures the multiplicative modulation while maintaining tractability
        multiplicative_term = grid_contribution * hd_contribution
        
        # Combine multiplicative and additive terms
        # The multiplicative term captures grid×HD interaction
        # Additive terms provide baseline responses
        combined = (
            0.7 * multiplicative_term  # Multiplicative interaction (primary)
            + 0.3 * grid_contribution  # Grid baseline
            + 0.3 * hd_contribution    # HD baseline
            + self.bias
        )
        return np.maximum(combined, 0.0)

