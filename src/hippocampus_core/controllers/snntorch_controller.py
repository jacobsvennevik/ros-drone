"""Controller backed by a frozen snnTorch policy checkpoint."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Literal
import warnings

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    import snntorch as snn
    from snntorch import surrogate
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "SnnTorchController requires PyTorch and snnTorch. Install with "
        "'python3 -m pip install torch snntorch'."
    ) from exc

from .base import SNNController

CHECKPOINT_VERSION = "1.0"

SURROGATE_FNS = {
    "atan": surrogate.atan,
    "fast_sigmoid": surrogate.fast_sigmoid,
    "sigmoid": surrogate.sigmoid,
    "erf": surrogate.erf,
    "logistic": surrogate.logistic,
    "linear": surrogate.linear,
}


def resolve_surrogate(name: str) -> Any:
    """Resolve a surrogate gradient constructor by canonical name."""

    key = name.lower()
    try:
        return SURROGATE_FNS[key]()
    except KeyError as exc:
        valid = ", ".join(sorted(SURROGATE_FNS))
        raise ValueError(f"Unsupported surrogate gradient '{name}'. Available: {valid}.") from exc


@dataclass
class SnnControllerState:
    """Hidden state for the recurrent membrane potentials."""

    hidden: Tensor

    def detach_(self) -> None:
        self.hidden = self.hidden.detach()

    def to(self, device: torch.device) -> "SnnControllerState":
        self.hidden = self.hidden.to(device)
        return self


class SnnControllerNet(nn.Module):
    """Minimal LIF network with tanh readout for continuous control."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        beta: float,
        spike_grad: str = "atan",
    ) -> None:
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.spike_grad_name = spike_grad.lower()
        surrogate_fn = resolve_surrogate(self.spike_grad_name)
        self.lif = snn.Leaky(beta=beta, spike_grad=surrogate_fn)
        self.readout = nn.Linear(hidden_dim, output_dim)

    @property
    def hidden_dim(self) -> int:
        return self.fc_in.out_features

    def init_state(self, batch_size: int, device: torch.device) -> SnnControllerState:
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        return SnnControllerState(hidden=hidden)

    def forward_step(
        self,
        inputs: Tensor,
        state: Optional[SnnControllerState],
    ) -> tuple[Tensor, SnnControllerState]:
        if inputs.dim() != 2:
            raise ValueError(f"Expected inputs with shape (batch, features); got {inputs.shape}.")

        if state is None:
            state = self.init_state(inputs.size(0), inputs.device)

        currents = self.fc_in(inputs)
        spikes, hidden = self.lif(currents, state.hidden)
        readout = self.readout(hidden)
        actions = torch.tanh(readout)
        return actions, SnnControllerState(hidden=hidden)

    def forward_sequence(
        self,
        inputs: Tensor,
        state: Optional[SnnControllerState] = None,
    ) -> tuple[Tensor, SnnControllerState]:
        if inputs.dim() != 3:
            raise ValueError(f"Expected inputs with shape (batch, time, features); got {inputs.shape}.")

        outputs = []
        next_state = state
        for t in range(inputs.size(1)):
            out, next_state = self.forward_step(inputs[:, t, :], next_state)
            outputs.append(out)
        stacked = torch.stack(outputs, dim=1)
        return stacked, next_state


@dataclass
class SnnTorchCheckpoint:
    """Structured representation of a saved snnTorch controller checkpoint."""

    state_dict: Dict[str, Tensor]
    model_hparams: Dict[str, Any]
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_scale: np.ndarray
    time_steps: int
    metadata: Dict[str, Any]


def _ensure_numpy(array: Any, *, name: str) -> np.ndarray:
    np_array = np.asarray(array, dtype=np.float32)
    if np_array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got {np_array.shape}.")
    return np_array


def load_snntorch_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> SnnTorchCheckpoint:
    """Load a snnTorch controller checkpoint saved by the offline trainer."""

    resolved = Path(checkpoint_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")

    data = torch.load(resolved, map_location=map_location)
    if not isinstance(data, dict):
        raise ValueError("Checkpoint file must contain a dictionary.")

    required = {
        "version",
        "model_state",
        "model_hparams",
        "obs_mean",
        "obs_std",
        "action_scale",
        "time_steps",
    }
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"Checkpoint missing required fields: {sorted(missing)}")

    version = str(data["version"])
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version '{version}'. Expected '{CHECKPOINT_VERSION}'."
        )

    obs_mean = _ensure_numpy(data["obs_mean"], name="obs_mean")
    obs_std = _ensure_numpy(data["obs_std"], name="obs_std")
    action_scale = _ensure_numpy(data["action_scale"], name="action_scale")

    metadata = dict(data.get("metadata", {}))

    return SnnTorchCheckpoint(
        state_dict=data["model_state"],
        model_hparams=dict(data["model_hparams"]),
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_scale=action_scale,
        time_steps=int(data["time_steps"]),
        metadata=metadata,
    )


@dataclass
class SnnTorchControllerConfig:
    """Configuration inputs for :class:`SnnTorchController`."""

    obs_dim: int
    action_dim: int
    hidden_dim: int = 32
    beta: float = 0.9
    time_steps: int = 1
    device: str = "cpu"
    action_scale: tuple[float, float] | np.ndarray = (0.3, 1.0)
    obs_mean: Optional[np.ndarray] = None
    obs_std: Optional[np.ndarray] = None
    spike_grad: str = "atan"
    checkpoint_path: Optional[str | Path] = None
    torchscript_path: Optional[str | Path] = None
    model_kind: Literal["state_dict", "torchscript"] = "state_dict"
    metadata: Dict[str, Any] = field(default_factory=dict)


class SnnTorchController(SNNController):
    """Controller that maps observations to actions using a trained snnTorch model."""

    def __init__(
        self,
        config: SnnTorchControllerConfig,
        checkpoint: Optional[SnnTorchCheckpoint] = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._metadata: Dict[str, Any] = dict(config.metadata)
        self._model_kind = str(config.model_kind).lower()
        if self._model_kind not in {"state_dict", "torchscript"}:
            raise ValueError("model_kind must be 'state_dict' or 'torchscript'.")

        checkpoint_obj = checkpoint
        if checkpoint_obj is None and config.checkpoint_path is not None:
            checkpoint_obj = load_snntorch_checkpoint(
                config.checkpoint_path,
                map_location=self.device,
            )

        if checkpoint_obj is None:
            if self._model_kind == "torchscript":
                if config.obs_mean is None or config.obs_std is None:
                    raise ValueError(
                        "TorchScript controller requires either a checkpoint with metadata or "
                        "explicit obs_mean/obs_std in the configuration."
                    )
            else:
                warnings.warn(
                    "SnnTorchController initialised without a checkpoint; "
                    "actions are produced by randomly initialised weights.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            hparams = checkpoint_obj.model_hparams
            config.obs_dim = int(hparams.get("input_dim", config.obs_dim))
            config.hidden_dim = int(hparams.get("hidden_dim", config.hidden_dim))
            config.action_dim = int(hparams.get("output_dim", config.action_dim))
            config.beta = float(hparams.get("beta", config.beta))
            config.spike_grad = str(hparams.get("spike_grad", config.spike_grad))
            config.time_steps = int(checkpoint_obj.time_steps)
            if config.obs_mean is None:
                config.obs_mean = checkpoint_obj.obs_mean
            if config.obs_std is None:
                config.obs_std = checkpoint_obj.obs_std
            config.action_scale = checkpoint_obj.action_scale
            self._metadata.update(checkpoint_obj.metadata)

        if self._model_kind == "torchscript":
            if config.torchscript_path is None and config.checkpoint_path is not None:
                config.torchscript_path = Path(config.checkpoint_path).with_suffix(".ts")
            if config.torchscript_path is None:
                raise ValueError("TorchScript controller requires 'torchscript_path'.")

        action_scale = np.asarray(config.action_scale, dtype=np.float32)
        if action_scale.shape != (config.action_dim,):
            raise ValueError(
                f"action_scale must have shape ({config.action_dim},); got {action_scale.shape}."
            )
        obs_mean = (
            np.zeros(config.obs_dim, dtype=np.float32)
            if config.obs_mean is None
            else np.asarray(config.obs_mean, dtype=np.float32)
        )
        obs_std = (
            np.ones(config.obs_dim, dtype=np.float32)
            if config.obs_std is None
            else np.asarray(config.obs_std, dtype=np.float32)
        )
        if obs_mean.shape != (config.obs_dim,) or obs_std.shape != (config.obs_dim,):
            raise ValueError("obs_mean and obs_std must match obs_dim.")

        self.obs_mean = torch.from_numpy(obs_mean).to(self.device)
        self.obs_std = torch.from_numpy(np.maximum(obs_std, 1e-6)).to(self.device)
        self.action_scale = torch.from_numpy(action_scale).to(self.device)

        self.net: Optional[SnnControllerNet] = None
        self.script_module: Optional[torch.jit.ScriptModule] = None

        if self._model_kind == "state_dict":
            self.net = SnnControllerNet(
                input_dim=config.obs_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.action_dim,
                beta=config.beta,
                spike_grad=config.spike_grad,
            )
            self.net.to(self.device)
            self.net.eval()
            if checkpoint_obj is not None:
                self.net.load_state_dict(checkpoint_obj.state_dict)
            self.state = self.net.init_state(batch_size=1, device=self.device)
        else:
            script_path = Path(config.torchscript_path).expanduser().resolve()
            if not script_path.exists():
                raise FileNotFoundError(f"TorchScript module not found: {script_path}")
            self.script_module = torch.jit.load(str(script_path), map_location=self.device)
            self.script_module.eval()
            hidden = torch.zeros(1, config.hidden_dim, device=self.device)
            self.state = SnnControllerState(hidden=hidden)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Arbitrary metadata stored alongside the checkpoint."""

        return dict(self._metadata)

    def reset(self) -> None:
        if self._model_kind == "state_dict" and self.net is not None:
            self.state = self.net.init_state(batch_size=1, device=self.device)
        else:
            hidden = torch.zeros(1, self.config.hidden_dim, device=self.device)
            self.state = SnnControllerState(hidden=hidden)

    def step(self, obs: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0:
            raise ValueError("dt must be positive")

        observation = np.asarray(obs, dtype=np.float32)
        if observation.ndim != 1 or observation.shape[0] != self.config.obs_dim:
            raise ValueError(
                f"Expected observation of shape ({self.config.obs_dim},), got {observation.shape}."
            )

        obs_tensor = torch.from_numpy(observation).to(self.device)
        normalized = (obs_tensor - self.obs_mean) / self.obs_std
        normalized = normalized.unsqueeze(0)  # (1, obs_dim)

        with torch.no_grad():
            if self._model_kind == "state_dict":
                if self.net is None:
                    raise RuntimeError("State_dict controller initialised without network weights.")
                output, self.state = self.net.forward_step(normalized, self.state)
                tensor_output = output
            else:
                if self.script_module is None:
                    raise RuntimeError("TorchScript controller initialised without a scripted module.")
                tensor_output, hidden = self.script_module(normalized, self.state.hidden)
                if hidden.shape != self.state.hidden.shape:
                    raise RuntimeError(
                        f"Unexpected hidden state shape from TorchScript module: {hidden.shape}"
                    )
                self.state.hidden = hidden
            self.state.detach_()
            action = (tensor_output.squeeze(0) * self.action_scale).cpu().numpy()

        if action.shape[0] != self.config.action_dim:
            raise RuntimeError("Unexpected action dimensionality from SNN network.")
        return action

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        model_kind: Literal["state_dict", "torchscript"] = "state_dict",
        torchscript_path: str | Path | None = None,
    ) -> "SnnTorchController":
        """Instantiate a controller using the metadata packaged in a checkpoint file."""

        checkpoint = load_snntorch_checkpoint(checkpoint_path, map_location=device)
        hparams = checkpoint.model_hparams
        config = SnnTorchControllerConfig(
            obs_dim=int(hparams["input_dim"]),
            action_dim=int(hparams["output_dim"]),
            hidden_dim=int(hparams.get("hidden_dim", 32)),
            beta=float(hparams.get("beta", 0.9)),
            time_steps=int(checkpoint.time_steps),
            device=str(device),
            action_scale=checkpoint.action_scale,
            obs_mean=checkpoint.obs_mean,
            obs_std=checkpoint.obs_std,
            spike_grad=str(hparams.get("spike_grad", "atan")),
            metadata=checkpoint.metadata,
            checkpoint_path=checkpoint_path,
            torchscript_path=torchscript_path,
            model_kind=model_kind,
        )
        return cls(config=config, checkpoint=checkpoint)

