"""Optuna-powered hyperparameter optimisation for the snnTorch policy trainer."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import optuna
    from optuna.pruners import MedianPruner
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Optuna is required for hyperparameter optimisation. Install with 'python -m pip install optuna'."
    ) from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "PyTorch is required for the snnTorch trainer. Install with 'python -m pip install torch snntorch'."
    ) from exc

from .train_snntorch_policy import (
    ArenaConfig,
    TrajectoryConfig,
    TrainingConfig,
    SURROGATE_FNS,
    OfflineTrajectoryDataset,
    compute_normalisation,
    create_sequences,
    save_artifacts,
    simulate_expert_trajectories,
    train_model,
)

SURROGATE_CHOICES = tuple(
    name for name in ("atan", "fast_sigmoid", "sigmoid", "erf") if name in SURROGATE_FNS
)
if not SURROGATE_CHOICES:  # pragma: no cover - defensive guard
    raise RuntimeError("No supported surrogate gradients found for HPO.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna study for snnTorch policy hyperparameters.")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials to run.")
    parser.add_argument("--study-name", type=str, default="snntorch-hpo", help="Name of the Optuna study.")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///optuna.db).")
    parser.add_argument("--resume", action="store_true", help="Resume an existing study if storage is provided.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument("--episodes", type=int, default=64, help="Number of expert episodes to simulate.")
    parser.add_argument("--steps", type=int, default=256, help="Number of steps per expert trajectory.")
    parser.add_argument("--max-epochs", type=int, default=35, help="Max epochs per trial during optimisation.")
    parser.add_argument(
        "--refit-epochs",
        type=int,
        default=None,
        help="Epoch count for retraining the best trial before exporting checkpoints (defaults to --max-epochs).",
    )
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience during trials.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for training ('cpu', 'cuda', ...).")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory to save best artifacts.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout (seconds) for the study.")
    parser.add_argument(
        "--pruner-warmup",
        type=int,
        default=5,
        help="Number of optimisation steps to run before the median pruner can stop a trial.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_hyperparameters(trial: optuna.trial.Trial) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    params["hidden_dim"] = trial.suggest_int("hidden_dim", 32, 128, step=16)
    params["beta"] = trial.suggest_float("beta", 0.80, 0.98)
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    params["time_steps"] = trial.suggest_int("time_steps", 4, 10)
    params["batch_size"] = trial.suggest_categorical("batch_size", [16, 24, 32, 48, 64])
    params["surrogate"] = trial.suggest_categorical("surrogate", SURROGATE_CHOICES)
    return params


def build_dataset(
    obs: np.ndarray,
    actions: np.ndarray,
    time_steps: int,
) -> tuple[OfflineTrajectoryDataset, np.ndarray, np.ndarray, np.ndarray]:
    obs_sequences, action_sequences = create_sequences(obs, actions, sequence_length=time_steps)
    dataset = OfflineTrajectoryDataset(obs_sequences, action_sequences)
    obs_mean, obs_std, action_scale = compute_normalisation(obs_sequences, action_sequences)
    return dataset, obs_mean, obs_std, action_scale


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_cfg = TrajectoryConfig(
        steps=int(args.steps),
        episodes=int(args.episodes),
        seed=int(args.seed),
    )
    arena_cfg = ArenaConfig()

    print("Generating expert dataset once for all trials...")
    base_obs, base_actions = simulate_expert_trajectories(arena_cfg, trajectory_cfg)

    pruner = MedianPruner(
        n_startup_trials=2,
        n_warmup_steps=max(int(args.pruner_warmup), 1),
        interval_steps=1,
    )
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name=args.study_name if args.storage else None,
        storage=args.storage,
        load_if_exists=args.resume if args.storage else False,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = sample_hyperparameters(trial)
        train_cfg = TrainingConfig(
            hidden_dim=int(params["hidden_dim"]),
            beta=float(params["beta"]),
            time_steps=int(params["time_steps"]),
            epochs=int(args.max_epochs),
            batch_size=int(params["batch_size"]),
            learning_rate=float(params["learning_rate"]),
            patience=min(int(args.patience), int(args.max_epochs)),
            device=args.device,
            surrogate=str(params["surrogate"]),
            seed=int(args.seed + trial.number),
        )

        dataset, obs_mean, obs_std, action_scale = build_dataset(
            base_obs,
            base_actions,
            train_cfg.time_steps,
        )

        model, val_loss, best_epoch = train_model(
            dataset,
            obs_mean,
            obs_std,
            action_scale,
            train_cfg,
            trial=trial,
        )
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("val_loss", val_loss)

        # Release GPU/CPU memory tied to the model weights between trials.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return float(val_loss)

    print(
        f"Starting Optuna study '{study.study_name}' for up to {args.trials} trials "
        f"(timeout={args.timeout!r})."
    )
    study.optimize(objective, n_trials=int(args.trials), timeout=args.timeout)

    if not study.best_trial:
        raise RuntimeError("Optuna study finished without any successful trials.")

    best_trial = study.best_trial
    print(
        f"Best trial #{best_trial.number} achieved val_loss={best_trial.value:.6f} "
        f"with params={best_trial.params}."
    )

    refit_epochs = int(args.refit_epochs or args.max_epochs)
    best_cfg = TrainingConfig(
        hidden_dim=int(best_trial.params["hidden_dim"]),
        beta=float(best_trial.params["beta"]),
        time_steps=int(best_trial.params["time_steps"]),
        epochs=refit_epochs,
        batch_size=int(best_trial.params["batch_size"]),
        learning_rate=float(best_trial.params["learning_rate"]),
        patience=min(int(args.patience), refit_epochs),
        device=args.device,
        surrogate=str(best_trial.params["surrogate"]),
        seed=int(args.seed),
    )

    dataset, obs_mean, obs_std, action_scale = build_dataset(
        base_obs,
        base_actions,
        best_cfg.time_steps,
    )

    print("Retraining best configuration to export checkpoints...")
    best_model, best_loss, best_epoch = train_model(
        dataset,
        obs_mean,
        obs_std,
        action_scale,
        best_cfg,
    )

    metadata = {
        "hpo": {
            "study_name": study.study_name,
            "trial_number": best_trial.number,
            "objective_value": float(best_trial.value),
            "params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in best_trial.params.items()},
            "best_epoch": best_epoch,
        }
    }

    artifacts = save_artifacts(
        best_model,
        obs_mean,
        obs_std,
        action_scale,
        best_cfg,
        arena_cfg,
        trajectory_cfg,
        output_dir,
        best_loss,
        extra_metadata=metadata,
    )
    del best_model

    best_record = {
        "study_name": study.study_name,
        "trial_number": best_trial.number,
        "objective_value": float(best_trial.value),
        "params": best_trial.params,
        "best_epoch": best_epoch,
        "artifacts": {name: str(path) if path is not None else None for name, path in artifacts.items()},
    }
    best_json_path = output_dir / "best_trial.json"
    best_json_path.write_text(json.dumps(best_record, indent=2))
    print(f"Wrote best trial summary to {best_json_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

