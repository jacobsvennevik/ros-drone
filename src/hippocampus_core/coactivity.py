"""Windowed coactivity tracker for place-cell spike trains."""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np


class CoactivityTracker:
    """Track pairwise coactivity counts within a sliding time window.

    Coactivity is defined as two cells emitting at least one spike within a
    temporal window of length ``window`` seconds. When a new spike arrives,
    the tracker increments the count for all cells that have spiked within the
    past ``window`` seconds, ensuring a symmetric coactivity matrix.

    For integration window functionality, tracks when each pair first exceeds
    a given threshold, enabling temporal gating of edge admission.
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
        # Track when each pair first exceeded threshold (for integration window)
        # Key: (i, j) tuple where i <= j, Value: time when threshold was first exceeded
        self._threshold_exceeded_time: dict[tuple[int, int], float] = {}
        # Track when pairs dropped below threshold (for temporal hysteresis)
        # Key: (i, j) tuple where i <= j, Value: time when threshold was last dropped below
        self._threshold_dropped_below_time: dict[tuple[int, int], float] = {}

    def register_spikes(
        self, t: float, spikes: np.ndarray, threshold: float | None = None
    ) -> None:
        """Record spikes at time ``t`` and update coactivity counts.

        Parameters
        ----------
        t:
            Simulation time in seconds at which the spikes occurred.
        spikes:
            Boolean or {0,1} array of shape (num_cells,) indicating which cells
            spiked at time ``t``.
        threshold:
            Optional threshold value. If provided, tracks when pairs first exceed
            this threshold in real-time during spike registration.
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
                old_count = self._coactivity[pair[0], pair[1]]
                self._coactivity[pair[0], pair[1]] += 1.0
                if pair[0] != pair[1]:
                    self._coactivity[pair[1], pair[0]] += 1.0
                
                # Track when pair first exceeds threshold (for integration window)
                # Also track when pair drops below threshold (for temporal hysteresis)
                if threshold is not None:
                    new_count = self._coactivity[pair[0], pair[1]]
                    if old_count < threshold <= new_count:
                        # This increment caused the pair to cross the threshold
                        if pair not in self._threshold_exceeded_time:
                            self._threshold_exceeded_time[pair] = t
                        # Remove from dropped-below tracking if it's back above threshold
                        self._threshold_dropped_below_time.pop(pair, None)
                    elif old_count >= threshold > new_count:
                        # This increment caused the pair to drop below threshold
                        self._threshold_dropped_below_time[pair] = t

    def check_threshold_exceeded(
        self, threshold: float, current_time: float, hysteresis_window: float = 0.0
    ) -> dict[tuple[int, int], float]:
        """Check which pairs exceed threshold and return when they first did so.
        
        With temporal hysteresis, a pair that dropped below threshold must stay
        below for at least `hysteresis_window` seconds before being removed.

        Parameters
        ----------
        threshold:
            Coactivity count threshold to check against.
        current_time:
            Current simulation time in seconds (used as fallback if pair wasn't
            tracked during registration).
        hysteresis_window:
            Optional hysteresis window in seconds. If a pair drops below threshold,
            it must remain below for at least this duration before being removed
            from the active set. Default 0.0 (no hysteresis).

        Returns
        -------
        dict[tuple[int, int], float]
            Dictionary mapping (i, j) pairs to the time when they first exceeded
            the threshold. Only includes pairs that currently exceed the threshold
            (or are within hysteresis window). If a pair wasn't tracked during
            registration (threshold not passed), uses current_time as fallback.
        """
        result: dict[tuple[int, int], float] = {}
        for i in range(self.num_cells):
            for j in range(i + 1, self.num_cells):
                pair = (i, j)
                current_count = self._coactivity[i, j]
                
                # Check if pair currently exceeds threshold
                if current_count >= threshold:
                    # Pair exceeds threshold - include if tracked
                    if pair in self._threshold_exceeded_time:
                        result[pair] = self._threshold_exceeded_time[pair]
                    else:
                        # Fallback: pair exceeds threshold but wasn't tracked
                        self._threshold_exceeded_time[pair] = current_time
                        result[pair] = current_time
                elif hysteresis_window > 0.0 and pair in self._threshold_exceeded_time:
                    # Pair is below threshold, but check if within hysteresis window
                    if pair in self._threshold_dropped_below_time:
                        drop_time = self._threshold_dropped_below_time[pair]
                        time_below = current_time - drop_time
                        if time_below < hysteresis_window:
                            # Still within hysteresis window - keep in active set
                            result[pair] = self._threshold_exceeded_time[pair]
                        else:
                            # Hysteresis expired - remove from tracking
                            self._threshold_exceeded_time.pop(pair, None)
        return result

    def get_coactivity_matrix(self) -> np.ndarray:
        """Return a copy of the symmetric coactivity count matrix."""

        return self._coactivity.copy()

    def reset(self) -> None:
        """Clear spike histories and coactivity counts."""

        self._coactivity.fill(0.0)
        for history in self._histories:
            history.clear()
        self._threshold_exceeded_time.clear()
        self._threshold_dropped_below_time.clear()
