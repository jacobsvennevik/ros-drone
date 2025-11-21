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

    def estimate_correction(self) -> Optional[PhaseCorrection]:
        """Return a suggested correction if sufficient data exist."""
        if not self.ready():
            return None

        positions = np.stack(self._positions)
        grid_estimates = np.stack(self._grid_estimates)
        pos_error = positions - grid_estimates
        translation = pos_error.mean(axis=0)

        headings = np.array(self._headings)
        estimates = np.array(self._hd_estimates)
        heading_error = _wrap_angle(headings - estimates)
        heading_delta = float(np.mean(heading_error))

        return PhaseCorrection(heading_delta=heading_delta, grid_translation=translation)

    def clear(self) -> None:
        """Clear accumulated samples after applying a correction."""
        self._positions.clear()
        self._headings.clear()
        self._hd_estimates.clear()
        self._grid_estimates.clear()

