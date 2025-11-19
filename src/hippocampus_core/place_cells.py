"""Place-cell population with Gaussian firing-rate tuning curves and spike sampling."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .env import Environment


class PlaceCellPopulation:
    """Gaussian place-cell population defined over a 2D environment.

    Parameters
    ----------
    environment:
        The environment providing spatial bounds for sampling place-field centers.
    num_cells:
        Number of place cells in the population.
    sigma:
        Standard deviation of the Gaussian tuning curve in arena units.
    max_rate:
        Peak firing rate in Hertz (Hz) at the center of a place field.
    rng:
        Optional NumPy random generator for reproducible sampling of centers
        and spikes.
    """

    def __init__(
        self,
        environment: Environment,
        num_cells: int = 100,
        sigma: float = 0.1,
        max_rate: float = 15.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if num_cells <= 0:
            raise ValueError("num_cells must be positive.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        if max_rate < 0:
            raise ValueError("max_rate must be non-negative.")

        self.environment = environment
        self.num_cells = num_cells
        self.sigma = sigma
        self.max_rate = max_rate
        self.rng = rng or np.random.default_rng()

        bounds = self.environment.bounds
        # Sample place cell centers only in valid (non-obstacle) positions
        self.centers = np.zeros((num_cells, 2))
        valid_count = 0
        max_attempts = num_cells * 100   # Prevent infinite loops
        attempts = 0
        
        while valid_count < num_cells and attempts < max_attempts:
            candidate = self.rng.uniform(
                low=(bounds.min_x, bounds.min_y),
                high=(bounds.max_x, bounds.max_y),
                size=(2,),
            )
            if self.environment.contains(tuple(candidate)):
                self.centers[valid_count] = candidate
                valid_count += 1
            attempts += 1
        
        # If we couldn't place all cells (unlikely), fill remaining with valid positions
        if valid_count < num_cells:
            # Use rejection sampling with a grid-based approach as fallback
            for i in range(valid_count, num_cells):
                placed = False
                for _ in range(1000):
                    candidate = self.rng.uniform(
                        low=(bounds.min_x, bounds.min_y),
                        high=(bounds.max_x, bounds.max_y),
                        size=(2,),
                    )
                    if self.environment.contains(tuple(candidate)):
                        self.centers[i] = candidate
                        placed = True
                        break
                if not placed:
                    # Last resort: place at a safe corner position
                    self.centers[i] = np.array([bounds.min_x + 0.01, bounds.min_y + 0.01])
        
        self._inv_two_sigma_sq = 1.0 / (2.0 * self.sigma**2)

    def get_rates(self, x: float, y: float) -> np.ndarray:
        """Return firing rates (Hz) for the population at position (x, y)."""

        position = np.array([x, y], dtype=float)
        deltas = self.centers - position
        squared_distances = np.einsum("ij,ij->i", deltas, deltas)
        rates = self.max_rate * np.exp(-squared_distances * self._inv_two_sigma_sq)
        return rates

    def sample_spikes(self, rates: np.ndarray, dt: float) -> np.ndarray:
        """Sample Poisson spikes for one simulation step.

        Parameters
        ----------
        rates:
            Firing rates in Hertz (Hz) for each place cell, shape (num_cells,).
        dt:
            Duration of the simulation step in seconds.

        Returns
        -------
        np.ndarray
            Boolean array of shape (num_cells,) where True indicates a spike.
        """

        if dt <= 0:
            raise ValueError("dt must be positive.")
        if rates.shape != (self.num_cells,):
            raise ValueError("rates must have shape (num_cells,)")
        if np.any(rates < 0):
            raise ValueError("rates must be non-negative.")

        spike_probs = rates * dt
        spike_probs = np.clip(spike_probs, 0.0, 1.0)
        spikes = self.rng.binomial(n=1, p=spike_probs, size=self.num_cells).astype(bool)
        return spikes

    def get_positions(self) -> np.ndarray:
        """Return the place-field centers as an array of shape (num_cells, 2)."""

        return self.centers.copy()

    def get_centers(self) -> np.ndarray:
        """Alias for ``get_positions`` for backwards compatibility."""

        return self.get_positions()

    def __len__(self) -> int:
        return self.num_cells
