"""R-STDP (Reward-modulated Spike-Timing-Dependent Plasticity) network for policy decisions.

This module implements a biologically plausible SNN that learns using local learning rules
without backpropagation. The network uses R-STDP: a three-factor learning rule that combines
pre-synaptic traces, post-synaptic traces, and a reward signal.

Key features:
- Biologically plausible: Only local information at synapses
- Online learning: Learns during execution
- No backpropagation: Uses eligibility traces instead
- Reward-modulated: Learning gated by reward signal
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .data_structures import FeatureVector


@dataclass
class RSTDPConfig:
    """Configuration for R-STDP network.

    Attributes
    ----------
    feature_dim:
        Input feature dimensionality.
    hidden_size:
        Number of hidden spiking neurons.
    output_size:
        Number of output neurons (2 for 2D, 3 for 3D).
    hidden_decay:
        Membrane decay factor for hidden neurons (0 < decay <= 1).
    output_decay:
        Membrane decay factor for output neurons.
    hidden_threshold:
        Spike threshold for hidden neurons.
    output_threshold:
        Spike threshold for output neurons.
    reset_voltage:
        Voltage reset after spike.
    pre_trace_decay:
        Decay for pre-synaptic eligibility traces.
    post_trace_decay:
        Decay for post-synaptic eligibility traces.
    eligibility_decay:
        Decay for eligibility traces (pre × post).
    learning_rate:
        Learning rate for weight updates.
    weight_min, weight_max:
        Bounds on synaptic weights.
    input_scale:
        Gain applied to input features.
    weight_init_scale:
        Standard deviation for random weight initialization.
    """

    feature_dim: int
    hidden_size: int = 64
    output_size: int = 2  # 2 for 2D, 3 for 3D

    # Membrane dynamics
    hidden_decay: float = 0.9
    output_decay: float = 0.9
    hidden_threshold: float = 1.0
    output_threshold: float = 0.6
    reset_voltage: float = 1.0

    # Eligibility traces
    pre_trace_decay: float = 0.8
    post_trace_decay: float = 0.8
    eligibility_decay: float = 0.85

    # Learning
    learning_rate: float = 5e-3
    weight_min: float = -1.5
    weight_max: float = 1.5

    # Input/output scaling
    input_scale: float = 1.0
    weight_init_scale: float = 0.2

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not (0.0 < self.hidden_decay <= 1.0):
            raise ValueError("hidden_decay must be in (0, 1]")
        if not (0.0 < self.output_decay <= 1.0):
            raise ValueError("output_decay must be in (0, 1]")
        if self.weight_min >= self.weight_max:
            raise ValueError("weight_min must be < weight_max")


class RSTDPPolicySNN:
    """R-STDP-based SNN for policy decisions.

    This network learns using biologically plausible local learning rules:
    - Pre-synaptic traces track recent input spikes
    - Post-synaptic traces track recent output spikes
    - Eligibility traces = pre × post (temporal correlation)
    - Weight update: Δw = learning_rate × reward × eligibility

    Architecture:
    - Input: Feature vector (from SpatialFeatureService)
    - Hidden: Spiking neurons with LIF dynamics
    - Output: Spiking neurons with continuous readout (membrane potential)
    """

    def __init__(self, config: RSTDPConfig, rng: Optional[np.random.Generator] = None):
        """Initialize R-STDP network.

        Parameters
        ----------
        config:
            Network configuration.
        rng:
            Optional random number generator for weight initialization.
        """
        self.config = config
        self._rng = rng or np.random.default_rng()

        # Synaptic weights
        # W_in: feature_dim -> hidden_size (fixed random projection)
        self._w_in = np.zeros((config.hidden_size, config.feature_dim), dtype=np.float32)
        # W_out: hidden_size -> output_size (learnable via R-STDP)
        self._w_out = np.zeros((config.output_size, config.hidden_size), dtype=np.float32)

        # Membrane potentials
        self._mem_hidden = np.zeros(config.hidden_size, dtype=np.float32)
        self._mem_out = np.zeros(config.output_size, dtype=np.float32)

        # Eligibility traces (for R-STDP learning)
        self._pre_trace = np.zeros(config.hidden_size, dtype=np.float32)  # Pre-synaptic
        self._post_trace = np.zeros(config.output_size, dtype=np.float32)  # Post-synaptic
        self._eligibility = np.zeros(
            (config.output_size, config.hidden_size), dtype=np.float32
        )  # Pre × Post

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize synaptic weights randomly."""
        scale = self.config.weight_init_scale
        # Input weights: fixed random projection (not learned)
        self._w_in = self._rng.normal(scale=scale, size=self._w_in.shape).astype(np.float32)
        # Output weights: random initialization (will be learned via R-STDP)
        self._w_out = self._rng.normal(scale=scale, size=self._w_out.shape).astype(np.float32)
        np.clip(
            self._w_out, self.config.weight_min, self.config.weight_max, out=self._w_out
        )

    def reset(self) -> None:
        """Reset network state (membrane potentials and traces).

        Note: Weights are preserved (learning accumulates across episodes).
        """
        self._mem_hidden.fill(0.0)
        self._mem_out.fill(0.0)
        self._pre_trace.fill(0.0)
        self._post_trace.fill(0.0)
        self._eligibility.fill(0.0)

    def forward(self, features: FeatureVector) -> np.ndarray:
        """Forward pass: compute action from features.

        Parameters
        ----------
        features:
            Feature vector from SpatialFeatureService.

        Returns
        -------
        np.ndarray
            Action vector (output_size,) with values in [-1, 1].
        """
        # Convert FeatureVector to numpy array
        feature_array = features.to_array() * self.config.input_scale

        # Update hidden layer
        hidden_spikes = self._update_hidden(feature_array)

        # Update output layer
        action = self._update_output(hidden_spikes)

        return action

    def _update_hidden(self, features: np.ndarray) -> np.ndarray:
        """Update hidden layer neurons.

        Parameters
        ----------
        features:
            Input feature vector (feature_dim,).

        Returns
        -------
        np.ndarray
            Binary spike vector (hidden_size,).
        """
        # LIF dynamics: decay membrane, add input current
        self._mem_hidden *= self.config.hidden_decay
        self._mem_hidden += self._w_in @ features

        # Check for spikes
        spikes = self._mem_hidden >= self.config.hidden_threshold
        if np.any(spikes):
            self._mem_hidden[spikes] -= self.config.reset_voltage

        # Update pre-synaptic trace (for eligibility)
        spike_vector = spikes.astype(np.float32)
        self._pre_trace *= self.config.pre_trace_decay
        self._pre_trace += spike_vector

        return spike_vector

    def _update_output(self, hidden_spikes: np.ndarray) -> np.ndarray:
        """Update output layer neurons.

        Parameters
        ----------
        hidden_spikes:
            Hidden layer spikes (hidden_size,).

        Returns
        -------
        np.ndarray
            Continuous action vector (output_size,) in [-1, 1].
        """
        # LIF dynamics: decay membrane, add input current
        self._mem_out *= self.config.output_decay
        self._mem_out += self._w_out @ hidden_spikes

        # Check for spikes
        post_spikes = self._mem_out >= self.config.output_threshold
        if np.any(post_spikes):
            self._mem_out[post_spikes] -= self.config.reset_voltage

        # Update post-synaptic trace
        post_vector = post_spikes.astype(np.float32)
        self._post_trace *= self.config.post_trace_decay
        self._post_trace += post_vector

        # Update eligibility trace: outer product of post × pre
        # This captures temporal correlation between pre and post spikes
        self._eligibility *= self.config.eligibility_decay
        self._eligibility += np.outer(post_vector, self._pre_trace)

        # Continuous readout: tanh of membrane potential
        # This provides smooth actions even when neurons don't spike
        action = np.tanh(self._mem_out).astype(np.float32)

        return action

    def update_weights(self, reward: float) -> None:
        """Update weights using R-STDP learning rule.

        R-STDP: Δw = learning_rate × reward × eligibility_trace

        This is a three-factor learning rule:
        1. Pre-synaptic trace (from hidden spikes)
        2. Post-synaptic trace (from output spikes)
        3. Reward signal (external, task-dependent)

        Parameters
        ----------
        reward:
            Scalar reward signal (positive = good, negative = bad).
        """
        if reward == 0.0:
            # Still decay eligibility to keep traces bounded
            self._eligibility *= self.config.eligibility_decay
            return

        # R-STDP weight update: reward-modulated eligibility
        delta_w = self.config.learning_rate * reward * self._eligibility
        self._w_out += delta_w

        # Clip weights to bounds
        np.clip(
            self._w_out, self.config.weight_min, self.config.weight_max, out=self._w_out
        )

    def get_weights(self) -> dict[str, np.ndarray]:
        """Get current synaptic weights (for checkpointing).

        Returns
        -------
        dict
            Dictionary with 'w_in' and 'w_out' keys.
        """
        return {
            "w_in": self._w_in.copy(),
            "w_out": self._w_out.copy(),
        }

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Set synaptic weights (for loading checkpoints).

        Parameters
        ----------
        weights:
            Dictionary with 'w_in' and 'w_out' keys.
        """
        w_in = np.asarray(weights.get("w_in"), dtype=np.float32)
        w_out = np.asarray(weights.get("w_out"), dtype=np.float32)

        if w_in.shape != self._w_in.shape:
            raise ValueError(f"w_in shape mismatch: {w_in.shape} vs {self._w_in.shape}")
        if w_out.shape != self._w_out.shape:
            raise ValueError(f"w_out shape mismatch: {w_out.shape} vs {self._w_out.shape}")

        self._w_in[...] = w_in
        self._w_out[...] = w_out

    @property
    def feature_dim(self) -> int:
        """Input feature dimensionality."""
        return self.config.feature_dim

    @property
    def output_dim(self) -> int:
        """Output action dimensionality."""
        return self.config.output_size

