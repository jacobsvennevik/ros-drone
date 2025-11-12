"""Windowed coactivity tracker for place-cell spike trains."""
from __future__ import annotations

from collections import deque
from typing import Deque, List

import numpy as np


class CoactivityTracker:
    """Track pairwise coactivity counts within a sliding time window.

    Coactivity is defined as two cells emitting at least one spike within a
    temporal window of length ``window`` seconds. When a new spike arrives,
    the tracker increments the count for all cells that have spiked within the
    past ``window`` seconds, ensuring a symmetric coactivity matrix.
    """

    def __init__(self, num_cells: int, window: float = 0.2) -> None:
        if num_cells <= 0:
            raise ValueError("num_cells must be positive.")
        if window <= 0:
            raise ValueError("window must be positive.")

        self.num_cells = num_cells
        self.window = window
        self._coactivity = np.zeros((num_cells, num_cells), dtype=float)
        self._histories: List[Deque[float]] = [deque() for _ in range(num_cells)]

    def register_spikes(self, t: float, spikes: np.ndarray) -> None:
        """Record spikes at time ``t`` and update coactivity counts.

        Parameters
        ----------
        t:
            Simulation time in seconds at which the spikes occurred.
        spikes:
            Boolean or {0,1} array of shape (num_cells,) indicating which cells
            spiked at time ``t``.
        """

        if spikes.shape != (self.num_cells,):
            raise ValueError("spikes must have shape (num_cells,)")

        spikes_bool = spikes.astype(bool, copy=False)
        window_start = t - self.window

        # Prune outdated spike times for all cells.
        for history in self._histories:
            while history and history[0] < window_start:
                history.popleft()

        active_indices = np.flatnonzero(spikes_bool)
        if active_indices.size == 0:
            return

        for idx in active_indices:
            self._histories[idx].append(t)

        counted_pairs = set()
        for i in active_indices:
            for j, history in enumerate(self._histories):
                if not history:
                    continue
                pair = (i, j) if i <= j else (j, i)
                if pair in counted_pairs:
                    continue
                counted_pairs.add(pair)
                self._coactivity[pair[0], pair[1]] += 1.0
                if pair[0] != pair[1]:
                    self._coactivity[pair[1], pair[0]] += 1.0

    def get_coactivity_matrix(self) -> np.ndarray:
        """Return a copy of the symmetric coactivity count matrix."""

        return self._coactivity.copy()

    def reset(self) -> None:
        """Clear spike histories and coactivity counts."""

        self._coactivity.fill(0.0)
        for history in self._histories:
            history.clear()
