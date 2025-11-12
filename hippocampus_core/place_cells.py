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
        self.centers = self.rng.uniform(
            low=(bounds.min_x, bounds.min_y),
            high=(bounds.max_x, bounds.max_y),
            size=(self.num_cells, 2),
        )
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
