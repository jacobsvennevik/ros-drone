"""Tiny snnTorch-based controller network used for offline training."""
from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
    from torch import Tensor
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "TrainableSnnController requires PyTorch. Install with "
        "'python3 -m pip install torch snntorch'."
    ) from exc

from hippocampus_core.controllers.snntorch_controller import (
    SnnControllerNet,
    SnnControllerState,
)


class TrainableSnnController(nn.Module):
    """Wrapper around :class:`SnnControllerNet` exposing sequence forward passes."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        beta: float = 0.9,
        spike_grad: str = "atan",
    ) -> None:
        super().__init__()
        self.model = SnnControllerNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            beta=beta,
            spike_grad=spike_grad,
        )

    def init_state(self, batch_size: int, device: torch.device) -> SnnControllerState:
        return self.model.init_state(batch_size=batch_size, device=device)

    def forward(
        self,
        inputs: Tensor,
        state: Optional[SnnControllerState] = None,
    ) -> Tuple[Tensor, SnnControllerState]:
        """Run the network over a sequence of inputs.

        Parameters
        ----------
        inputs:
            Tensor with shape ``(batch, time, features)`` containing normalised
            observation windows.
        state:
            Optional recurrent state initialiser. When omitted the state is reset
            to zeros at the start of the sequence, matching offline training.

        Returns
        -------
        Tuple[Tensor, SnnControllerState]
            The per-timestep action predictions (scaled to ``[-1, 1]``) and the
            final recurrent state.
        """

        return self.model.forward_sequence(inputs, state)



