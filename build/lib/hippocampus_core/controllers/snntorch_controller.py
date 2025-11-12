"""Controller implementation backed by a simple snnTorch spiking network."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import snntorch as snn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "SnnTorchController requires PyTorch and snnTorch. Install with "
        "'python3 -m pip install torch snntorch'."
    ) from exc

from .base import SNNController


@dataclass
class SnnTorchControllerConfig:
    """Configuration for :class:`SnnTorchController`."""

    obs_dim: int
    action_dim: int
    hidden_dim: int = 32
    beta: float = 0.9  # membrane decay for Leaky neurons
    device: str = "cpu"


class _SimpleSNN(nn.Module):
    """Minimal feedforward SNN with a single hidden layer of LIF neurons."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, beta: float, device: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta)
        self.to(device)

    def forward(self, spikes: torch.Tensor, mem1: torch.Tensor, mem2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cur1 = self.fc1(spikes)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        return spk2, mem1, mem2


class SnnTorchController(SNNController):
    """Controller mapping observations to actions using a simple snnTorch network.

    The network weights are randomly initialised and untrained; actions are
    therefore not meaningful yet but serve to demonstrate integration.
    """

    def __init__(
        self,
        config: SnnTorchControllerConfig,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)

        self.net = _SimpleSNN(
            input_dim=config.obs_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.action_dim,
            beta=config.beta,
            device=config.device,
        )

        self.mem1 = torch.zeros(config.hidden_dim, device=self.device)
        self.mem2 = torch.zeros(config.action_dim, device=self.device)

    def reset(self) -> None:
        self.mem1.zero_()
        self.mem2.zero_()

    def step(self, obs: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0:
            raise ValueError("dt must be positive")

        observation = np.asarray(obs, dtype=np.float32)
        if observation.ndim != 1 or observation.shape[0] != self.config.obs_dim:
            raise ValueError(
                f"Expected observation of shape ({self.config.obs_dim},), got {observation.shape}."
            )

        x = torch.from_numpy(observation).to(self.device)
        spk_out, self.mem1, self.mem2 = self.net(x, self.mem1, self.mem2)
        action = torch.tanh(self.mem2).detach().cpu().numpy()
        if action.shape[0] != self.config.action_dim:
            raise RuntimeError("Unexpected action dimensionality from SNN network")
        return action
