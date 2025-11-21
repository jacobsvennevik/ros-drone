"""SNN network architecture for policy decisions."""
from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    import snntorch as snn
    from snntorch import surrogate
except ImportError:
    torch = None
    nn = None
    Tensor = None
    snn = None
    surrogate = None

from ..controllers.snntorch_controller import resolve_surrogate


@dataclass
class SNNConfig:
    """Configuration for SNN network."""

    feature_dim: int
    hidden_dim: int = 64
    output_dim: int = 2  # [v, ω] for 2D, [v, ω, vz] for 3D
    beta: float = 0.9  # Membrane decay
    spike_grad: str = "atan"  # Surrogate gradient function
    threshold: float = 1.0  # Spike threshold
    reset_mechanism: str = "subtract"  # "subtract" or "zero"
    num_steps: int = 1  # Time steps for encoding/inference


class PolicySNN(nn.Module):
    """SNN network for policy decisions.

    Architecture:
    - Input layer: Linear(feature_dim, hidden_dim)
    - Hidden layer: LIF neurons
    - Output layer: Linear(hidden_dim, output_dim)
    - Readout: tanh activation for continuous actions
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 2,
        beta: float = 0.9,
        spike_grad: str = "atan",
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
    ):
        """Initialize SNN network.

        Parameters
        ----------
        feature_dim:
            Input feature dimensionality.
        hidden_dim:
            Hidden layer size.
        output_dim:
            Output action dimensionality.
        beta:
            Membrane decay factor (0 < β < 1).
        spike_grad:
            Surrogate gradient function name.
        threshold:
            Spike threshold.
        reset_mechanism:
            Reset mechanism ("subtract" or "zero").
        """
        if nn is None:
            raise ImportError("PyTorch and snnTorch required for PolicySNN")

        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.fc_in = nn.Linear(feature_dim, hidden_dim)

        # LIF neuron layer
        surrogate_fn = resolve_surrogate(spike_grad)
        self.lif = snn.Leaky(
            beta=beta,
            spike_grad=surrogate_fn,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
        )

        # Output readout
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward_step(
        self,
        spike_input: Tensor,  # (batch, feature_dim) - already encoded
        membrane: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Single-step forward pass.

        Parameters
        ----------
        spike_input:
            Input spikes (batch, feature_dim).
        membrane:
            Previous membrane potential (batch, hidden_dim).

        Returns
        -------
        tuple[Tensor, Tensor]
            (action, next_membrane) where action is (batch, output_dim).
        """
        batch_size = spike_input.size(0)
        if membrane is None:
            membrane = torch.zeros(batch_size, self.hidden_dim, device=spike_input.device)

        # Project input spikes to currents
        currents = self.fc_in(spike_input)

        # LIF neuron dynamics
        spikes, membrane = self.lif(currents, membrane)

        # Readout from membrane potential (not spikes)
        # This provides smooth continuous output
        action_logits = self.readout(membrane)
        actions = torch.tanh(action_logits)  # Bound to [-1, 1]

        return actions, membrane

    def forward_sequence(
        self,
        spike_train: Tensor,  # (time_steps, batch, feature_dim)
        membrane: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Multi-step forward pass for temporal integration.

        Parameters
        ----------
        spike_train:
            Spike train (time_steps, batch, feature_dim).
        membrane:
            Initial membrane potential.

        Returns
        -------
        tuple[Tensor, Tensor]
            (actions, final_membrane) where actions is (time_steps, batch, output_dim).
        """
        time_steps = spike_train.size(0)
        outputs = []
        next_membrane = membrane

        for t in range(time_steps):
            action, next_membrane = self.forward_step(spike_train[t], next_membrane)
            outputs.append(action)

        return torch.stack(outputs, dim=0), next_membrane

    def init_state(self, batch_size: int, device: torch.device) -> Tensor:
        """Initialize membrane potential to zero.

        Parameters
        ----------
        batch_size:
            Batch size.
        device:
            Device to create tensor on.

        Returns
        -------
        Tensor
            Zero-initialized membrane potential (batch_size, hidden_dim).
        """
        return torch.zeros(batch_size, self.hidden_dim, device=device)

