"""Experience-dependent calibration utilities for aligning neural maps."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class PhaseCorrection:
    """Represents a calibration recommendation for HD and grid systems."""

    heading_delta: float
    grid_translation: np.ndarray


class PhaseOptimizer:
    """Tracks loop-closure observations and proposes phase corrections."""

    def __init__(self, max_history: int = 256) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be positive.")
        self.max_history = max_history
        self._positions: Deque[np.ndarray] = deque(maxlen=max_history)
        self._headings: Deque[float] = deque(maxlen=max_history)
        self._hd_estimates: Deque[float] = deque(maxlen=max_history)
        self._grid_estimates: Deque[np.ndarray] = deque(maxlen=max_history)

    def add_sample(
        self,
        position: np.ndarray,
        heading: float,
        hd_estimate: float,
        grid_estimate: np.ndarray,
    ) -> None:
        """Record a new experience sample."""
        self._positions.append(np.asarray(position, dtype=float))
        self._headings.append(float(heading))
        self._hd_estimates.append(float(hd_estimate))
        self._grid_estimates.append(np.asarray(grid_estimate, dtype=float))

    def ready(self, min_samples: int = 10) -> bool:
        """Return True if enough samples exist to compute a correction."""
        return len(self._positions) >= min_samples

    def effective_sample_size(self) -> float:
        """Compute effective sample size (ESS) for current history window.
        
        ESS = (sum(w_i))^2 / sum(w_i^2) where w_i are sample weights.
        For uniform weights (current implementation), ESS equals number of samples.
        This can be extended with temporal weighting in the future.
        
        Returns
        -------
        float
            Effective sample size. If ESS < 20, window may be too noisy.
        """
        n = len(self._positions)
        if n == 0:
            return 0.0
        # For uniform weights, ESS = n (all samples are independent)
        # Future: could add temporal decay weights for more sophisticated ESS
        return float(n)

    def estimate_correction(self, min_ess: float = 20.0) -> Optional[PhaseCorrection]:
        """Return a suggested correction if sufficient data exist.
        
        Parameters
        ----------
        min_ess:
            Minimum effective sample size required. If ESS < min_ess,
            returns None (window is too noisy for reliable correction).
            Default: 20.0 (recommended for statistical rigor).
        """
        if not self.ready():
            return None
        
        # Check effective sample size to ensure statistical reliability
        ess = self.effective_sample_size()
        if ess < min_ess:
            # Window is too noisy - don't compute correction yet
            return None

        positions = np.stack(self._positions)
        grid_estimates = np.stack(self._grid_estimates)
        pos_error = positions - grid_estimates
        translation = pos_error.mean(axis=0)

        headings = np.array(self._headings)
        estimates = np.array(self._hd_estimates)
        heading_error = _wrap_angle(headings - estimates)
        
        # Circular mean: use vector average instead of linear mean
        # This correctly handles angles near ±π boundary (e.g., [350°, 10°] → 0°)
        complex_sum = np.sum(np.exp(1j * heading_error))
        if np.abs(complex_sum) < 1e-12:
            heading_delta = 0.0
        else:
            heading_delta = float(np.angle(complex_sum))

        return PhaseCorrection(heading_delta=heading_delta, grid_translation=translation)

    def clear(self) -> None:
        """Clear accumulated samples after applying a correction."""
        self._positions.clear()
        self._headings.clear()
        self._hd_estimates.clear()
        self._grid_estimates.clear()

