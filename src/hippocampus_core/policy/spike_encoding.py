"""Spike encoding for converting features to spike trains."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import torch
    import snntorch.spikegen as spikegen
except ImportError:
    torch = None
    spikegen = None

from .data_structures import FeatureVector


@dataclass
class EncodingConfig:
    """Configuration for spike encoding."""

    encoding_scheme: str = "rate"  # "rate", "latency", "delta"
    num_steps: int = 1  # Time steps for encoding
    gain: float = 1.0
    offset: float = 0.0
    threshold: float = 0.01  # For latency/delta encoding


def encode_rate(
    features: torch.Tensor,
    num_steps: int = 1,
    gain: float = 1.0,
    offset: float = 0.0,
) -> torch.Tensor:
    """Encode features using rate coding.

    Parameters
    ----------
    features:
        Tensor of shape (batch, feature_dim) with values in [0, 1] or normalized.
    num_steps:
        Number of time steps to encode (default: 1 for single-step inference).
    gain:
        Gain factor to scale firing rates.
    offset:
        Offset to shift firing rates.

    Returns
    -------
    torch.Tensor
        Spike train of shape (num_steps, batch, feature_dim).
    """
    if torch is None:
        raise ImportError("PyTorch and snnTorch required for spike encoding")

    # Normalize features to [0, 1] if needed
    normalized = torch.clamp(features * gain + offset, 0.0, 1.0)

    # Generate Poisson spike trains
    spike_train = spikegen.rate(
        normalized.unsqueeze(0).repeat(num_steps, 1, 1),
        num_steps=num_steps,
    )
    return spike_train


def encode_latency(
    features: torch.Tensor,
    num_steps: int = 10,
    threshold: float = 0.01,
    gain: float = 1.0,
) -> torch.Tensor:
    """Encode features using latency coding.

    Higher values spike earlier in the sequence.

    Parameters
    ----------
    features:
        Tensor of shape (batch, feature_dim).
    num_steps:
        Number of time steps.
    threshold:
        Minimum threshold for spiking.
    gain:
        Gain factor.

    Returns
    -------
    torch.Tensor
        Spike train of shape (num_steps, batch, feature_dim).
    """
    if torch is None:
        raise ImportError("PyTorch and snnTorch required for spike encoding")

    normalized = torch.clamp(features * gain, threshold, 1.0)
    spike_train = spikegen.latency(
        normalized.unsqueeze(0).repeat(num_steps, 1, 1),
        num_steps=num_steps,
        threshold=threshold,
    )
    return spike_train


def encode_delta(
    features: torch.Tensor,
    features_prev: torch.Tensor,
    num_steps: int = 1,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Encode feature changes using delta modulation.

    Parameters
    ----------
    features:
        Current features tensor.
    features_prev:
        Previous features tensor.
    num_steps:
        Number of time steps.
    threshold:
        Change threshold.

    Returns
    -------
    torch.Tensor
        Spike train of shape (num_steps, batch, feature_dim).
    """
    if torch is None:
        raise ImportError("PyTorch and snnTorch required for spike encoding")

    delta = features - features_prev
    spike_train = spikegen.delta(
        delta.unsqueeze(0).repeat(num_steps, 1, 1),
        threshold=threshold,
    )
    return spike_train


class FeatureEncoder:
    """Encodes FeatureVector to spike trains."""

    def __init__(self, config: EncodingConfig):
        """Initialize encoder.

        Parameters
        ----------
        config:
            Encoding configuration.
        """
        if torch is None:
            raise ImportError("PyTorch and snnTorch required for FeatureEncoder")
        self.config = config
        self._prev_features: Optional[torch.Tensor] = None

    def encode(
        self,
        features: FeatureVector,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Convert FeatureVector to spike train.

        Parameters
        ----------
        features:
            Feature vector to encode.
        device:
            Device to place tensors on.

        Returns
        -------
        torch.Tensor
            Spike train of shape (num_steps, batch=1, feature_dim).
        """
        # Convert FeatureVector to tensor
        feature_tensor = self._vector_to_tensor(features, device)

        # Apply encoding scheme
        if self.config.encoding_scheme == "rate":
            spike_train = encode_rate(
                feature_tensor,
                num_steps=self.config.num_steps,
                gain=self.config.gain,
                offset=self.config.offset,
            )
        elif self.config.encoding_scheme == "latency":
            spike_train = encode_latency(
                feature_tensor,
                num_steps=self.config.num_steps,
                threshold=self.config.threshold,
                gain=self.config.gain,
            )
        elif self.config.encoding_scheme == "delta":
            if self._prev_features is None:
                # Fallback to rate on first step
                spike_train = encode_rate(feature_tensor, num_steps=self.config.num_steps)
            else:
                spike_train = encode_delta(
                    feature_tensor,
                    self._prev_features,
                    num_steps=self.config.num_steps,
                    threshold=self.config.threshold,
                )
            self._prev_features = feature_tensor.clone()
        else:
            raise ValueError(f"Unknown encoding scheme: {self.config.encoding_scheme}")

        return spike_train

    def _vector_to_tensor(self, features: FeatureVector, device: torch.device) -> torch.Tensor:
        """Convert FeatureVector to flat tensor.

        Parameters
        ----------
        features:
            Feature vector.
        device:
            Device to place tensor on.

        Returns
        -------
        torch.Tensor
            Tensor of shape (1, feature_dim).
        """
        # Concatenate all feature components
        feature_list = []
        feature_list.extend(features.goal_ego)
        for neighbor in features.neighbors_k:
            feature_list.extend(neighbor)
        feature_list.extend(features.topo_ctx)
        feature_list.extend(features.safety)
        if features.dynamics:
            feature_list.extend(features.dynamics)

        return torch.tensor(feature_list, dtype=torch.float32, device=device).unsqueeze(0)  # (1, dim)

    def reset(self) -> None:
        """Reset encoder state (e.g., for new episode)."""
        self._prev_features = None

