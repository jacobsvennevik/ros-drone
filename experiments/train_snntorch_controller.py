#!/usr/bin/env python3
"""Offline training pipeline for the snnTorch controller."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from experiments.models.snn_controller import TrainableSnnController
from hippocampus_core.controllers.snntorch_controller import CHECKPOINT_VERSION


@dataclass(slots=True)
class SequenceDataset(Dataset):
    """Simple dataset wrapping observation windows and action targets."""

    observations: Tensor
    targets: Tensor

    def __post_init__(self) -> None:
        if self.observations.dim() != 3:
            raise ValueError("observations must have shape (N, time, features)")
        if self.targets.dim() != 2:
            raise ValueError("targets must have shape (N, features)")
        if self.observations.size(0) != self.targets.size(0):
            raise ValueError("observations and targets must have the same batch dimension")

    def __len__(self) -> int:
        return self.observations.size(0)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.observations[index], self.targets[index]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small snnTorch controller offline using surrogate gradients.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to an .npz file containing 'observations' and 'actions' arrays.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/snn_controller.pt"),
        help="Destination for the saved checkpoint (default: checkpoints/snn_controller.pt).",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=5,
        help="Number of unrolled time steps used during BPTT (default: 5).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension of the SNN (default: 64).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9,
        help="Membrane decay parameter for LIF neurons (default: 0.9).",
    )
    parser.add_argument(
        "--spike-grad",
        choices=["atan"],
        default="atan",
        help="Surrogate gradient to use for the spiking nonlinearity (default: atan).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size (default: 128).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW (default: 1e-3).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay coefficient (default: 1e-4).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping max-norm (default: 1.0, disable with <= 0).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for validation (default: 0.2).",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        nargs="+",
        default=[0.3, 1.0],
        help=(
            "Per-dimension scaling applied to tanh outputs to recover physical actions. "
            "Provide one value per action dimension (default: 0.3 1.0 for [linear, angular])."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Computation device (auto, cpu, cuda, mps, ...). Default selects GPU if available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for numpy/torch (default: 17).",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling of training sequences (useful for debugging).",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def _load_dataset(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(str(npz_path), allow_pickle=False) as data:
        if "observations" not in data or "actions" not in data:
            raise KeyError("Dataset must contain 'observations' and 'actions' arrays.")
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
    if observations.ndim != 2:
        raise ValueError(f"'observations' must have shape (N, obs_dim); got {observations.shape}.")
    if actions.ndim != 2:
        raise ValueError(f"'actions' must have shape (N, action_dim); got {actions.shape}.")
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("observations and actions must share the same leading dimension.")
    return observations, actions


def _create_sequences(
    observations: np.ndarray,
    actions: np.ndarray,
    time_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if time_steps <= 0:
        raise ValueError("time_steps must be >= 1.")
    if time_steps > observations.shape[0]:
        raise ValueError(
            f"time_steps ({time_steps}) exceeds available samples ({observations.shape[0]})."
        )

    seq_obs = []
    seq_actions = []
    for idx in range(time_steps - 1, observations.shape[0]):
        start = idx - time_steps + 1
        seq_obs.append(observations[start : idx + 1])
        seq_actions.append(actions[idx])

    return np.stack(seq_obs, axis=0), np.stack(seq_actions, axis=0)


def _standardize_observations(
    observations: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = observations.mean(axis=0)
    std = observations.std(axis=0)
    std = np.maximum(std, 1e-6)
    normalized = (observations - mean) / std
    return normalized, mean.astype(np.float32), std.astype(np.float32)


def _prepare_datasets(
    args: argparse.Namespace,
    *,
    rng: np.random.Generator,
) -> tuple[
    SequenceDataset,
    SequenceDataset,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    observations, actions = _load_dataset(args.dataset)
    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]

    action_scale = np.asarray(args.action_scale, dtype=np.float32)
    if action_scale.shape != (action_dim,):
        raise ValueError(
            f"action_scale requires {action_dim} values; received {len(action_scale)}."
        )
    if np.any(action_scale <= 0.0):
        raise ValueError("action_scale values must be strictly positive.")

    standardized_obs, obs_mean, obs_std = _standardize_observations(observations)
    seq_obs, seq_actions = _create_sequences(standardized_obs, actions, args.time_steps)

    targets = seq_actions / action_scale  # broadcast over dimensions
    targets = np.clip(targets, -1.0, 1.0)

    num_sequences = seq_obs.shape[0]
    if num_sequences < 2:
        raise ValueError("Dataset needs at least 2 sequences to create a validation split.")

    indices = np.arange(num_sequences)
    if not args.no_shuffle:
        rng.shuffle(indices)

    val_count = max(int(num_sequences * args.val_split), 1)
    train_count = num_sequences - val_count
    if train_count <= 0:
        raise ValueError("Validation split is too large; no data left for training.")

    train_idx = indices[:train_count]
    val_idx = indices[train_count:]

    train_dataset = SequenceDataset(
        observations=torch.from_numpy(seq_obs[train_idx]),
        targets=torch.from_numpy(targets[train_idx]),
    )
    val_dataset = SequenceDataset(
        observations=torch.from_numpy(seq_obs[val_idx]),
        targets=torch.from_numpy(targets[val_idx]),
    )

    return train_dataset, val_dataset, obs_mean, obs_std, action_scale


def _train_epoch(
    model: TrainableSnnController,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_inputs, batch_targets in loader:
        batch_size = batch_inputs.size(0)
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        state = model.init_state(batch_size, device)
        outputs, _ = model(batch_inputs, state)
        predictions = outputs[:, -1, :]

        loss = criterion(predictions, batch_targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def _evaluate(
    model: TrainableSnnController,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: torch.nn.Module,
    action_scale: Tensor,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_actual_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_size = batch_inputs.size(0)
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            state = model.init_state(batch_size, device)
            outputs, _ = model(batch_inputs, state)
            predictions = outputs[:, -1, :]

            loss = criterion(predictions, batch_targets)

            predicted_actions = predictions * action_scale
            actual_targets = batch_targets * action_scale
            mse_actual = torch.mean((predicted_actions - actual_targets) ** 2)

            total_loss += loss.item() * batch_size
            total_actual_mse += mse_actual.item() * batch_size
            total_samples += batch_size

    mean_loss = total_loss / max(total_samples, 1)
    mean_actual = total_actual_mse / max(total_samples, 1)
    return mean_loss, mean_actual


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    args.dataset = args.dataset.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rng = np.random.default_rng(args.seed)

    train_dataset, val_dataset, obs_mean, obs_std, action_scale = _prepare_datasets(
        args,
        rng=rng,
    )

    obs_dim = train_dataset.observations.size(-1)
    action_dim = train_dataset.targets.size(-1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = TrainableSnnController(
        input_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        output_dim=action_dim,
        beta=args.beta,
        spike_grad=args.spike_grad,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.MSELoss()
    action_scale_tensor = torch.from_numpy(action_scale).to(device)

    best_val_loss = float("inf")
    best_actual_mse = float("inf")
    best_epoch = 0
    best_state = None

    print(
        f"Starting training on {device} | obs_dim={obs_dim} | action_dim={action_dim} | "
        f"time_steps={args.time_steps} | train_samples={len(train_dataset)} | "
        f"val_samples={len(val_dataset)}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            grad_clip=args.grad_clip,
        )
        val_loss, val_actual = _evaluate(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            action_scale=action_scale_tensor,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"val_mse_actual={val_actual:.6f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_actual_mse = val_actual
            best_epoch = epoch
            best_state = {
                "model_state": model.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

    if best_state is None:
        best_state = {
            "model_state": model.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "model_state": best_state["model_state"],
        "model_hparams": {
            "input_dim": obs_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": action_dim,
            "beta": args.beta,
            "spike_grad": args.spike_grad,
        },
        "obs_mean": obs_mean.tolist(),
        "obs_std": obs_std.tolist(),
        "action_scale": action_scale.tolist(),
        "time_steps": args.time_steps,
        "metadata": {
            "train_loss": best_state["train_loss"],
            "val_loss": best_state["val_loss"],
            "val_mse_actual": best_actual_mse,
            "best_epoch": best_epoch,
            "epochs_trained": args.epochs,
            "dataset_path": str(args.dataset),
            "seed": args.seed,
        },
    }
    torch.save(checkpoint, args.output)

    print(f"Checkpoint saved to {args.output} (best epoch {best_epoch}).", flush=True)


if __name__ == "__main__":
    main()

