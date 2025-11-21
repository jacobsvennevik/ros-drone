"""Temporal context for policy decisions."""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .data_structures import FeatureVector, PolicyDecision


class TemporalContext:
    """Maintains temporal context for policy decisions."""

    def __init__(self, history_length: int = 10):
        """Initialize temporal context.

        Parameters
        ----------
        history_length:
            Maximum history length.
        """
        self.history_length = history_length
        self.feature_history: Deque[FeatureVector] = deque(maxlen=history_length)
        self.decision_history: Deque[PolicyDecision] = deque(maxlen=history_length)
        self.membrane_history: Deque[torch.Tensor] = deque(maxlen=history_length) if torch else deque(maxlen=history_length)

    def update(
        self,
        features: FeatureVector,
        decision: PolicyDecision,
        membrane: Optional[torch.Tensor] = None,
    ) -> None:
        """Update history buffers.

        Parameters
        ----------
        features:
            Current features.
        decision:
            Current decision.
        membrane:
            Current membrane potential (if using SNN).
        """
        self.feature_history.append(features)
        self.decision_history.append(decision)
        if membrane is not None and torch is not None:
            self.membrane_history.append(membrane.clone())

    def get_temporal_features(self) -> Optional[torch.Tensor]:
        """Get concatenated temporal features for multi-step encoding.

        Returns
        -------
        Optional[torch.Tensor]
            Temporal features (time_steps, feature_dim) or None if no history.
        """
        if not self.feature_history or torch is None:
            return None

        # Convert features to tensors
        feature_tensors = []
        for fv in self.feature_history:
            feature_array = fv.to_array()
            feature_tensors.append(torch.from_numpy(feature_array))

        if not feature_tensors:
            return None

        return torch.stack(feature_tensors, dim=0)  # (time_steps, feature_dim)

    def reset(self) -> None:
        """Reset history."""
        self.feature_history.clear()
        self.decision_history.clear()
        self.membrane_history.clear()

