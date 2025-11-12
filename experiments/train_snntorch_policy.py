"""Offline training pipeline for a small snnTorch navigation policy.

This script synthesises expert demonstrations inside the toy arena,
trains a compact spiking neural network policy with surrogate gradients,
and saves the resulting checkpoint (weights plus normalisation metadata).

References:
    - snnTorch surrogate gradient tutorial (Tutorial 5/6):
      https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5_bptt.html
    - PyTorch state_dict checkpoint best practices:
      https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import random

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split
    import snntorch as snn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "The offline snnTorch trainer requires 'torch' and 'snntorch'. "
        "Install them with 'python3 -m pip install torch snntorch'."
    ) from exc
from hippocampus_core.env import Environment
from hippocampus_core.controllers.snntorch_controller import (
    CHECKPOINT_VERSION,
    SnnControllerNet,
    SURROGATE_FNS,
    resolve_surrogate,
)


@dataclass
class ArenaConfig:
    """Arena geometry and obstacle parameters."""

    width: float = 1.0
    height: float = 1.0
    obstacle_center: Tuple[float, float] = (0.5, 0.5)
    obstacle_radius: float = 0.12
    obstacle_influence: float = 0.25
    goal: Tuple[float, float] = (0.85, 0.8)


@dataclass
class TrajectoryConfig:
    """Simulation parameters for generating expert demonstrations."""

    dt: float = 0.05
    steps: int = 256
    episodes: int = 128
    linear_limit: float = 0.25
    angular_limit: float = 1.2
    seed: int = 1


@dataclass
class TrainingConfig:
    """Hyper-parameters for the snnTorch training loop."""

    hidden_dim: int = 48
    beta: float = 0.9
    time_steps: int = 6
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    patience: int = 10
    device: str = "cpu"
    grad_clip: float = 1.0
    surrogate: str = "atan"
    seed: int = 1

    @property
    def sequence_length(self) -> int:
        """Backward-compatible alias for scripts expecting the old argument name."""

        return self.time_steps


class OfflineTrajectoryDataset(Dataset):
    """Dataset of sliding-window observation/action sequences."""

    def __init__(
        self,
        obs_sequences: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> None:
        if obs_sequences.ndim != 3:
            raise ValueError("obs_sequences must have shape (N, T, obs_dim)")
        if action_sequences.ndim != 3:
            raise ValueError("action_sequences must have shape (N, T, action_dim)")
        if obs_sequences.shape[:2] != action_sequences.shape[:2]:
            raise ValueError("Sequence dimensions of observations and actions must match.")

        self.obs_sequences = obs_sequences
        self.action_sequences = action_sequences

    def __len__(self) -> int:
        return int(self.obs_sequences.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.obs_sequences[idx], self.action_sequences[idx]


class ExpertPolicy:
    """Rule-based policy that drives towards the goal while avoiding the obstacle."""

    def __init__(self, arena: ArenaConfig, traj_cfg: TrajectoryConfig):
        self.arena = arena
        self.traj_cfg = traj_cfg

    def __call__(self, position: np.ndarray, heading: float) -> np.ndarray:
        goal_vec = np.array(self.arena.goal, dtype=float) - position
        goal_dist = np.linalg.norm(goal_vec) + 1e-6
        goal_dir = goal_vec / goal_dist

        obstacle_vec = position - np.array(self.arena.obstacle_center, dtype=float)
        obstacle_dist = np.linalg.norm(obstacle_vec) + 1e-6
        obstacle_dir = obstacle_vec / obstacle_dist
        obstacle_strength = 0.0
        if obstacle_dist < self.arena.obstacle_influence:
            obstacle_strength = (self.arena.obstacle_influence - obstacle_dist) / self.arena.obstacle_influence
        repulsive = obstacle_strength * obstacle_dir

        desired_dir = goal_dir + 1.5 * repulsive
        desired_dir_norm = np.linalg.norm(desired_dir)
        if desired_dir_norm < 1e-6:
            desired_dir = goal_dir
        else:
            desired_dir = desired_dir / desired_dir_norm

        desired_heading = float(np.arctan2(desired_dir[1], desired_dir[0]))
        heading_error = wrap_to_pi(desired_heading - heading)
        angular = np.clip(2.5 * heading_error, -self.traj_cfg.angular_limit, self.traj_cfg.angular_limit)

        linear = np.clip(0.6 * goal_dist, 0.0, self.traj_cfg.linear_limit)
        if obstacle_dist < (self.arena.obstacle_radius + 0.05):
            linear *= -0.2
        return np.array([linear, angular], dtype=np.float32)


class PolicySNN(nn.Module):
    """Feedforward spiking policy with a single hidden LIF layer."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        beta: float,
        spike_grad: str = "atan",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.spike_grad_name = spike_grad
        self.fc_in = nn.Linear(obs_dim, hidden_dim)
        surrogate_fn = resolve_surrogate(spike_grad)
        self.lif = snn.Leaky(beta=beta, spike_grad=surrogate_fn, threshold=1.0)
        self.readout = nn.Linear(hidden_dim, action_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process a sequence of observations.

        Parameters
        ----------
        inputs:
            Tensor shaped (time, batch, obs_dim).
        """

        time_steps, batch_size, _ = inputs.shape
        membrane = torch.zeros(batch_size, self.hidden_dim, device=inputs.device, dtype=inputs.dtype)
        outputs: List[torch.Tensor] = []
        for t in range(time_steps):
            current = self.fc_in(inputs[t])
            spk, membrane = self.lif(current, membrane)
            outputs.append(self.readout(membrane))
        return torch.stack(tuple(outputs), dim=0)


class ScriptableSnnController(nn.Module):
    """Wrapper that exposes a stateless forward for TorchScript export."""

    def __init__(self, net: SnnControllerNet) -> None:
        super().__init__()
        # Share parameters with the provided controller network.
        self.fc_in = net.fc_in
        self.lif = net.lif
        self.readout = net.readout

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() != 2 or hidden.dim() != 2:
            raise RuntimeError("Expected obs and hidden tensors with shape (batch, features).")
        currents = self.fc_in(obs)
        spikes, next_hidden = self.lif(currents, hidden)
        readout = self.readout(next_hidden)
        actions = torch.tanh(readout)
        return actions, next_hidden


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""

    wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped)


def simulate_expert_trajectories(
    arena_cfg: ArenaConfig,
    traj_cfg: TrajectoryConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate expert trajectories and return observations/actions arrays.

    Returns
    -------
    obs : np.ndarray
        Array of shape (episodes, steps, obs_dim).
    actions : np.ndarray
        Array of shape (episodes, steps, action_dim).
    """

    rng = np.random.default_rng(traj_cfg.seed)
    environment = Environment(width=arena_cfg.width, height=arena_cfg.height)
    policy = ExpertPolicy(arena_cfg, traj_cfg)

    obs_dim = 4
    action_dim = 2
    obs = np.zeros((traj_cfg.episodes, traj_cfg.steps, obs_dim), dtype=np.float32)
    actions = np.zeros((traj_cfg.episodes, traj_cfg.steps, action_dim), dtype=np.float32)

    for episode in range(traj_cfg.episodes):
        position = rng.uniform(
            [0.1 * arena_cfg.width, 0.1 * arena_cfg.height],
            [0.9 * arena_cfg.width, 0.9 * arena_cfg.height],
        )
        heading = rng.uniform(-np.pi, np.pi)

        for step in range(traj_cfg.steps):
            observation = np.array(
                [position[0], position[1], float(np.cos(heading)), float(np.sin(heading))],
                dtype=np.float32,
            )
            control = policy(position, heading)

            obs[episode, step] = observation
            actions[episode, step] = control

            heading = wrap_to_pi(heading + control[1] * traj_cfg.dt)
            delta = np.array([np.cos(heading), np.sin(heading)], dtype=float) * control[0] * traj_cfg.dt
            proposed_position = position + delta
            clipped_position, _ = environment.clip(proposed_position)
            position = np.asarray(clipped_position, dtype=float)

    return obs, actions


def create_sequences(
    obs: np.ndarray,
    actions: np.ndarray,
    sequence_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert episode trajectories into sliding-window sequences."""

    episodes, steps, obs_dim = obs.shape
    _, _, action_dim = actions.shape
    if steps < sequence_length:
        raise ValueError("Number of steps must exceed sequence_length.")

    num_sequences = (steps - sequence_length + 1) * episodes
    obs_sequences = np.zeros((num_sequences, sequence_length, obs_dim), dtype=np.float32)
    action_sequences = np.zeros((num_sequences, sequence_length, action_dim), dtype=np.float32)

    idx = 0
    for ep in range(episodes):
        for start in range(steps - sequence_length + 1):
            stop = start + sequence_length
            obs_sequences[idx] = obs[ep, start:stop]
            action_sequences[idx] = actions[ep, start:stop]
            idx += 1

    obs_tensor = torch.from_numpy(obs_sequences)
    action_tensor = torch.from_numpy(action_sequences)
    return obs_tensor, action_tensor


def compute_normalisation(
    obs_sequences: torch.Tensor,
    action_sequences: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-feature statistics for normalising inputs and outputs."""

    obs_flat = obs_sequences.reshape(-1, obs_sequences.shape[-1]).numpy()
    action_flat = action_sequences.reshape(-1, action_sequences.shape[-1]).numpy()

    obs_mean = obs_flat.mean(axis=0)
    obs_std = obs_flat.std(axis=0)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)

    action_scale = np.max(np.abs(action_flat), axis=0)
    action_scale = np.where(action_scale < 1e-6, 1.0, action_scale)

    return obs_mean.astype(np.float32), obs_std.astype(np.float32), action_scale.astype(np.float32)


def prepare_training_data(
    arena_cfg: ArenaConfig,
    traj_cfg: TrajectoryConfig,
    time_steps: int,
) -> Tuple[OfflineTrajectoryDataset, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data and package tensors used for training."""

    if time_steps <= 1:
        raise ValueError("time_steps must be greater than 1.")

    obs, actions = simulate_expert_trajectories(arena_cfg, traj_cfg)
    obs_sequences, action_sequences = create_sequences(
        obs,
        actions,
        sequence_length=time_steps,
    )
    obs_mean, obs_std, action_scale = compute_normalisation(obs_sequences, action_sequences)
    dataset = OfflineTrajectoryDataset(obs_sequences, action_sequences)
    return dataset, obs_mean, obs_std, action_scale


def train_model(
    dataset: OfflineTrajectoryDataset,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    action_scale: np.ndarray,
    train_cfg: TrainingConfig,
    *,
    trial: Any | None = None,
) -> Tuple[PolicySNN, float, int]:
    """Train the SNN policy and return the best model plus loss."""

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    device = torch.device(train_cfg.device)
    obs_dim = dataset.obs_sequences.shape[-1]
    action_dim = dataset.action_sequences.shape[-1]

    model = PolicySNN(
        obs_dim=obs_dim,
        hidden_dim=train_cfg.hidden_dim,
        action_dim=action_dim,
        beta=train_cfg.beta,
        spike_grad=train_cfg.surrogate,
    )
    model.to(device)

    normalise_obs = torch.from_numpy(obs_mean).to(device)
    normalise_std = torch.from_numpy(obs_std).to(device)
    action_scale_tensor = torch.from_numpy(action_scale).to(device)

    total_samples = len(dataset)
    val_size = max(int(0.1 * total_samples), 1)
    train_size = total_samples - val_size
    split_generator = torch.Generator()
    split_generator.manual_seed(train_cfg.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=3,
    )

    best_loss = float("inf")
    best_state = None
    epochs_without_improve = 0

    for epoch in range(train_cfg.epochs):
        model.train()
        train_loss_total = 0.0
        for obs_batch, action_batch in train_loader:
            obs_batch = obs_batch.permute(1, 0, 2).to(device)  # (T, B, obs_dim)
            action_batch = action_batch.permute(1, 0, 2).to(device)  # (T, B, action_dim)

            normed_obs = (obs_batch - normalise_obs) / normalise_std
            target_actions = torch.tanh(action_batch / action_scale_tensor)

            optimizer.zero_grad()
            outputs = model(normed_obs)
            predictions = torch.tanh(outputs)
            loss = F.mse_loss(predictions, target_actions)
            loss.backward()
            if train_cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            train_loss_total += float(loss.item()) * obs_batch.shape[1]

        avg_train_loss = train_loss_total / max(train_size, 1)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for obs_batch, action_batch in val_loader:
                obs_batch = obs_batch.permute(1, 0, 2).to(device)
                action_batch = action_batch.permute(1, 0, 2).to(device)
                normed_obs = (obs_batch - normalise_obs) / normalise_std
                target_actions = torch.tanh(action_batch / action_scale_tensor)
                predictions = torch.tanh(model(normed_obs))
                loss = F.mse_loss(predictions, target_actions)
                val_loss_total += float(loss.item()) * obs_batch.shape[1]

        avg_val_loss = val_loss_total / max(val_size, 1)
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch + 1:03d} | train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f}",
            flush=True,
        )

        if trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                try:
                    import optuna  # type: ignore import-not-found
                except ImportError as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError(
                        "Optuna is required to use pruning callbacks in train_model."
                    ) from exc
                raise optuna.exceptions.TrialPruned()

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
            }
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= train_cfg.patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
    best_epoch = best_state["epoch"] if best_state is not None else train_cfg.epochs
    return model, best_loss, int(best_epoch)


def save_artifacts(
    model: PolicySNN,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    action_scale: np.ndarray,
    train_cfg: TrainingConfig,
    arena_cfg: ArenaConfig,
    traj_cfg: TrajectoryConfig,
    output_dir: Path,
    best_val_loss: float,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Optional[Path]]:
    """Persist trained model and normalisation metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    controller_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "model_state": controller_state,
        "model_hparams": {
            "input_dim": int(model.obs_dim),
            "hidden_dim": int(model.hidden_dim),
            "output_dim": int(model.action_dim),
            "beta": float(model.lif.beta.item()),
            "spike_grad": str(model.spike_grad_name),
        },
        "obs_mean": obs_mean.astype(np.float32).tolist(),
        "obs_std": obs_std.astype(np.float32).tolist(),
        "action_scale": action_scale.astype(np.float32).tolist(),
        "time_steps": int(train_cfg.time_steps),
        "metadata": {
            "train_loss": float("nan"),
            "val_loss": float(best_val_loss),
            "dataset": "synthetic_expert",
            "time_steps": train_cfg.time_steps,
            "epochs": train_cfg.epochs,
            "batch_size": train_cfg.batch_size,
            "learning_rate": train_cfg.learning_rate,
            "hidden_dim": train_cfg.hidden_dim,
            "beta": train_cfg.beta,
            "surrogate": train_cfg.surrogate,
            "seed": train_cfg.seed,
            "arena": {
                "width": arena_cfg.width,
                "height": arena_cfg.height,
                "obstacle_center": arena_cfg.obstacle_center,
                "obstacle_radius": arena_cfg.obstacle_radius,
                "goal": arena_cfg.goal,
            },
            "trajectory": {
                "dt": traj_cfg.dt,
                "steps": traj_cfg.steps,
                "episodes": traj_cfg.episodes,
                "linear_limit": traj_cfg.linear_limit,
                "angular_limit": traj_cfg.angular_limit,
            },
        },
    }
    if extra_metadata:
        checkpoint["metadata"].update(extra_metadata)
    model_path = output_dir / "snn_controller.pt"
    torch.save(checkpoint, model_path)

    torchscript_path: Optional[Path] = None
    try:
        controller_net = SnnControllerNet(
            input_dim=model.obs_dim,
            hidden_dim=model.hidden_dim,
            output_dim=model.action_dim,
            beta=train_cfg.beta,
            spike_grad=train_cfg.surrogate,
        )
        controller_net.load_state_dict(controller_state)
        controller_net.eval()
        script_wrapper = ScriptableSnnController(controller_net)
        script_module = torch.jit.script(script_wrapper)
        torchscript_path = output_dir / "snn_controller.ts"
        script_module.save(str(torchscript_path))
        print(f"Exported TorchScript module to {torchscript_path}")
    except Exception as exc:  # pragma: no cover - best effort export
        print(f"Warning: TorchScript export failed: {exc}")

    print(f"Saved model to {model_path}")

    return {"checkpoint": model_path, "torchscript": torchscript_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline snnTorch policy training pipeline.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory to store checkpoints.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training.")
    parser.add_argument("--episodes", type=int, default=128, help="Number of expert episodes to simulate.")
    parser.add_argument("--steps", type=int, default=256, help="Number of steps per expert trajectory.")
    parser.add_argument(
        "--time-steps",
        "--sequence-length",
        dest="time_steps",
        type=int,
        default=6,
        help="Sliding window length/unroll horizon for BPTT.",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=5e-3, help="Optimizer learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=48, help="Hidden neuron count.")
    parser.add_argument("--beta", type=float, default=0.9, help="Leaky beta parameter.")
    parser.add_argument(
        "--surrogate",
        type=str,
        default="atan",
        choices=sorted(SURROGATE_FNS.keys()),
        help="Surrogate gradient to use in the LIF layer.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.time_steps <= 1:
        raise ValueError("time_steps must be > 1.")

    arena_cfg = ArenaConfig()
    traj_cfg = TrajectoryConfig(
        steps=int(args.steps),
        episodes=int(args.episodes),
        seed=int(args.seed),
    )
    train_cfg = TrainingConfig(
        hidden_dim=int(args.hidden_dim),
        beta=float(args.beta),
        time_steps=int(args.time_steps),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        device=args.device,
        surrogate=str(args.surrogate),
        seed=int(args.seed),
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Simulating expert trajectories...")
    dataset, obs_mean, obs_std, action_scale = prepare_training_data(
        arena_cfg,
        traj_cfg,
        train_cfg.time_steps,
    )

    print("Training snnTorch policy...")
    model, best_loss, best_epoch = train_model(dataset, obs_mean, obs_std, action_scale, train_cfg)
    print(f"Best validation loss: {best_loss:.6f} (epoch {best_epoch})")
    save_artifacts(
        model,
        obs_mean,
        obs_std,
        action_scale,
        train_cfg,
        arena_cfg,
        traj_cfg,
        args.output_dir,
        best_loss,
        extra_metadata={"best_epoch": best_epoch},
    )


if __name__ == "__main__":
    main()


