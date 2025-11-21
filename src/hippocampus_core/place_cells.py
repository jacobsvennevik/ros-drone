"""Place-cell population with Gaussian firing-rate tuning curves and spike sampling."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .env import Environment


class PlaceCellPopulation:
    """Gaussian place-cell population with optional orientation tuning.

    Parameters
    ----------
    environment:
        Environment used for sampling spatial place-field centers.
    num_cells:
        Number of place cells in the population.
    sigma:
        Standard deviation of the Gaussian tuning curve in arena units.
    max_rate:
        Peak firing rate in Hertz (Hz) at the center of a place field.
    track_altitude:
        When True, sample/store an additional altitude coordinate (z) per cell.
        Callers must then pass a ``z`` value when evaluating the population.
    altitude_bounds:
        Min/max range for sampling altitude coordinates when ``track_altitude`` is
        enabled and explicit centers were not provided.
    orientation_kappa:
        Concentration parameter for an optional von Mises tuning curve over
        head direction θ. ``None`` disables orientation modulation.
    orientation_preferences:
        Optional per-cell preferred heading (radians). If omitted and
        ``orientation_kappa`` is set, preferences are sampled uniformly from
        ``[-π, π)``.
    rng:
        Optional NumPy random generator for reproducible sampling of centers,
        orientations, and spikes.
    centers:
        Optional explicit array of place-field centers. The expected shape is
        ``(num_cells, 2)`` for planar-only fields or ``(num_cells, 3)`` when
        ``track_altitude`` is enabled.
    """

    def __init__(
        self,
        environment: Environment,
        num_cells: int = 100,
        sigma: float = 0.1,
        max_rate: float = 15.0,
        rng: Optional[np.random.Generator] = None,
        centers: Optional[np.ndarray] = None,
        track_altitude: bool = False,
        altitude_bounds: Tuple[float, float] = (0.0, 1.0),
        orientation_kappa: Optional[float] = None,
        orientation_preferences: Optional[np.ndarray] = None,
    ) -> None:
        if num_cells <= 0:
            raise ValueError("num_cells must be positive.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        if max_rate < 0:
            raise ValueError("max_rate must be non-negative.")
        if track_altitude:
            if altitude_bounds[0] > altitude_bounds[1]:
                raise ValueError("altitude_bounds must be provided as (min, max)")
        if orientation_kappa is not None and orientation_kappa <= 0:
            raise ValueError("orientation_kappa must be positive if provided.")

        self.environment = environment
        self.num_cells = num_cells
        self.sigma = sigma
        self.max_rate = max_rate
        self.rng = rng or np.random.default_rng()
        self.track_altitude = track_altitude
        self.altitude_bounds = altitude_bounds
        self.orientation_kappa = orientation_kappa

        bounds = self.environment.bounds
        feature_dim = 3 if self.track_altitude else 2
        if centers is not None:
            centers = np.asarray(centers, dtype=float)
            expected_shape = (num_cells, feature_dim)
            if centers.shape != expected_shape:
                raise ValueError(
                    f"centers must have shape {expected_shape}; got {centers.shape}"
                )
            for idx, candidate in enumerate(centers):
                if not self.environment.contains(tuple(candidate[:2])):
                    raise ValueError(
                        f"Provided place-cell center at index {idx} lies outside the environment or inside an obstacle: {candidate}"
                    )
            self.centers = centers.copy()
        else:
            # Sample place cell centers only in valid (non-obstacle) positions
            self.centers = np.zeros((num_cells, feature_dim))
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
                    self.centers[valid_count, :2] = candidate
                    if self.track_altitude:
                        self.centers[valid_count, 2] = self.rng.uniform(
                            self.altitude_bounds[0], self.altitude_bounds[1]
                        )
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
                            self.centers[i, :2] = candidate
                            if self.track_altitude:
                                self.centers[i, 2] = self.rng.uniform(
                                    self.altitude_bounds[0], self.altitude_bounds[1]
                                )
                            placed = True
                            break
                    if not placed:
                        # Last resort: place at a safe corner position
                        fallback = np.array([bounds.min_x + 0.01, bounds.min_y + 0.01])
                        self.centers[i, :2] = fallback
                        if self.track_altitude:
                            self.centers[i, 2] = self.altitude_bounds[0]
        
        self._inv_two_sigma_sq = 1.0 / (2.0 * self.sigma**2)
        if self.orientation_kappa is not None:
            if orientation_preferences is not None:
                preferences = np.asarray(orientation_preferences, dtype=float)
                if preferences.shape != (num_cells,):
                    raise ValueError(
                        f"orientation_preferences must have shape ({num_cells},); got {preferences.shape}"
                    )
                self._orientation_preferences = preferences
            else:
                self._orientation_preferences = self.rng.uniform(
                    -np.pi, np.pi, size=num_cells
                )
        else:
            self._orientation_preferences = None

    def get_rates(
        self,
        x: float | np.ndarray,
        y: Optional[float] = None,
        theta: Optional[float] = None,
        z: Optional[float] = None,
    ) -> np.ndarray:
        """Return firing rates (Hz) for the population at the requested location.

        Parameters
        ----------
        x, y:
            Planar coordinates. Alternatively, ``x`` can be an array-like object
            whose first two entries correspond to `(x, y)` and any remaining entries
            encode ``z`` and/or ``theta``.
        theta:
            Optional head-direction input. Required when ``orientation_kappa`` is set.
        z:
            Optional altitude coordinate. Required when ``track_altitude`` is True and
            not provided implicitly through the ``x`` argument.
        """

        if np.isscalar(x):
            if y is None:
                raise ValueError("y must be provided when x is a scalar.")
            xy = np.array([float(x), float(y)], dtype=float)
            z_value = z
            theta_value = theta
        else:
            obs = np.asarray(x, dtype=float).ravel()
            if obs.size < 2:
                raise ValueError("Observation vector must contain at least x and y.")
            xy = obs[:2]
            remainder = obs[2:]
            z_value = z
            theta_value = theta
            if self.track_altitude:
                if remainder.size > 0:
                    z_value = remainder[0]
                    remainder = remainder[1:]
                elif z_value is None:
                    raise ValueError("z must be provided when altitude tracking is enabled.")
            if self.orientation_kappa is not None:
                if remainder.size > 0 and theta_value is None:
                    theta_value = remainder[0]

        if self.track_altitude:
            if z_value is None:
                raise ValueError("z must be provided when altitude tracking is enabled.")
            position = np.array([xy[0], xy[1], float(z_value)], dtype=float)
        else:
            position = xy

        deltas = self.centers - position
        squared_distances = np.einsum("ij,ij->i", deltas, deltas)
        rates = self.max_rate * np.exp(-squared_distances * self._inv_two_sigma_sq)

        if self.orientation_kappa is not None:
            if theta_value is None:
                raise ValueError("theta must be provided when orientation_kappa is set.")
            orientation_term = np.exp(
                self.orientation_kappa * np.cos(theta_value - self._orientation_preferences)
            )
            rates *= orientation_term

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
        """Return the place-field centers (shape ``(num_cells, 2 or 3)``)."""

        return self.centers.copy()

    def get_centers(self) -> np.ndarray:
        """Alias for ``get_positions`` for backwards compatibility."""

        return self.get_positions()

    @property
    def orientation_preferences(self) -> Optional[np.ndarray]:
        """Return the per-cell preferred orientation when von-Mises tuning is enabled."""

        if self._orientation_preferences is None:
            return None
        return self._orientation_preferences.copy()

    def __len__(self) -> int:
        return self.num_cells
