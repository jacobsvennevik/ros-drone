"""Collect observation-action demonstrations for offline SNN training.

This script runs the toy navigation simulator with an existing controller
backend (currently the R-STDP controller) and records tuples of
``(obs_t, action_t)``. The resulting dataset is written to an ``.npz`` file
alongside metadata describing the normalisation used for observations and the
action clamps applied during collection.

Example:
    python -m experiments.collect_imitation --episodes 5 --steps 2000 \
        --output experiments/data/imitation_rstdp_seed0.npz
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from hippocampus_core.controllers.rstdp_controller import (
    RSTDPController,
    RSTDPControllerConfig,
)
from hippocampus_core.env import Environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect imitation datasets from existing controllers in the toy simulator.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to roll out.")
    parser.add_argument("--steps", type=int, default=2000, help="Simulation steps per episode.")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation time step (seconds).")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/data/imitation_rstdp.npz"),
        help="Destination .npz file for the collected dataset.",
    )

    parser.add_argument("--arena-width", type=float, default=1.0, help="Width of the arena (m).")
    parser.add_argument("--arena-height", type=float, default=1.0, help="Height of the arena (m).")

    parser.add_argument(
        "--obstacle-radius",
        type=float,
        default=0.15,
        help="Radius of the central obstacle used for shaping (m).",
    )
    parser.add_argument(
        "--obstacle-margin",
        type=float,
        default=0.05,
        help="Safety margin around the obstacle (m).",
    )

    parser.add_argument(
        "--max-linear",
        type=float,
        default=0.3,
        help="Clamp applied to linear velocity commands (m/s).",
    )
    parser.add_argument(
        "--max-angular",
        type=float,
        default=1.0,
        help="Clamp applied to angular velocity commands (rad/s).",
    )

    parser.add_argument(
        "--include-velocity",
        action="store_true",
        help="Augment observations with (vx, vy) in addition to (x, y).",
    )

    parser.add_argument(
        "--rstdp-hidden",
        type=int,
        default=48,
        help="Hidden layer size for the teacher R-STDP controller.",
    )
    parser.add_argument(
        "--rstdp-learning-rate",
        type=float,
        default=0.0,
        help="Learning rate for the teacher controller (0 disables online updates).",
    )
    parser.add_argument(
        "--rstdp-weights",
        type=Path,
        default=None,
        help="Optional path to an .npz file with 'w_in' and 'w_out' arrays to initialise the teacher.",
    )

    parser.add_argument(
        "--controller-reset-every-episode",
        action="store_true",
        help="Reset the controller state (but keep weights) before each episode.",
    )

    return parser.parse_args()


def _sample_start_position(
    rng: np.random.Generator,
    env: Environment,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
) -> np.ndarray:
    """Draw a random start location that lies outside the obstacle."""

    bounds = env.bounds
    for _ in range(1_000):
        candidate = np.array(
            [
                rng.uniform(bounds.min_x, bounds.max_x),
                rng.uniform(bounds.min_y, bounds.max_y),
            ],
            dtype=float,
        )
        if np.linalg.norm(candidate - obstacle_center) >= obstacle_radius * 1.5:
            return candidate
    # Fallback: place near the boundary if sampling fails (very unlikely).
    return np.array([bounds.min_x + 0.05 * bounds.width, bounds.min_y + 0.05 * bounds.height])


def _build_teacher_controller(args: argparse.Namespace, include_velocity: bool) -> RSTDPController:
    """Initialise the teacher controller used to generate actions."""

    input_dim = 4 if include_velocity else 2
    obstacle_center = (
        float(0.5 * args.arena_width),
        float(0.5 * args.arena_height),
    )
    controller_rng = np.random.default_rng(args.seed + 1)
    config = RSTDPControllerConfig(
        input_dim=input_dim,
        hidden_size=args.rstdp_hidden,
        output_size=2,
        action_gain_linear=args.max_linear,
        action_gain_angular=args.max_angular,
        learning_rate=args.rstdp_learning_rate,
        obstacle_center=obstacle_center,
        obstacle_radius=args.obstacle_radius,
        obstacle_margin=args.obstacle_margin,
        keep_weights_on_reset=True,
        rng=controller_rng,
    )
    controller = RSTDPController(config=config)

    if args.rstdp_weights is not None:
        state = np.load(args.rstdp_weights)
        if "w_in" not in state or "w_out" not in state:
            raise ValueError("Expected 'w_in' and 'w_out' arrays in the provided weights file.")
        controller.import_state({"w_in": state["w_in"], "w_out": state["w_out"]})

    return controller


def _compute_observation(
    position: np.ndarray,
    velocity: np.ndarray,
    include_velocity: bool,
    pos_center: np.ndarray,
    pos_scale: np.ndarray,
    max_linear: float,
) -> Tuple[np.ndarray, np.ndarray]:
    raw_obs = position if not include_velocity else np.concatenate([position, velocity])
    raw_obs = raw_obs.astype(np.float32, copy=True)

    norm_obs = raw_obs.copy()
    norm_obs[:2] = (norm_obs[:2] - pos_center) / pos_scale
    norm_obs[:2] = np.clip(norm_obs[:2], -1.0, 1.0)

    if include_velocity:
        if max_linear > 0.0:
            norm_obs[2:] = np.clip(norm_obs[2:] / max_linear, -1.0, 1.0)
        else:
            norm_obs[2:] = 0.0

    return raw_obs, norm_obs


def _simulate_episode(
    controller: RSTDPController,
    env: Environment,
    *,
    dt: float,
    steps: int,
    rng: np.random.Generator,
    include_velocity: bool,
    max_linear: float,
    max_angular: float,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
    obstacle_margin: float,
    pos_center: np.ndarray,
    pos_scale: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = _sample_start_position(rng, env, obstacle_center, obstacle_radius + obstacle_margin)
    heading = rng.uniform(-math.pi, math.pi)
    velocity = np.zeros(2, dtype=float)

    obs_raw_buffer: list[np.ndarray] = []
    obs_norm_buffer: list[np.ndarray] = []
    action_buffer: list[np.ndarray] = []
    time_buffer: list[float] = []

    time = 0.0
    for _ in range(steps):
        raw_obs, norm_obs = _compute_observation(
            position,
            velocity,
            include_velocity=include_velocity,
            pos_center=pos_center,
            pos_scale=pos_scale,
            max_linear=max_linear,
        )

        controller_action = controller.step(raw_obs, dt)
        linear_cmd = float(np.clip(controller_action[0], -max_linear, max_linear))
        angular_cmd = float(np.clip(controller_action[1], -max_angular, max_angular))

        heading += angular_cmd * dt
        dx = linear_cmd * math.cos(heading)
        dy = linear_cmd * math.sin(heading)
        proposed_position = position + np.array([dx, dy]) * dt

        clipped_position, clipped_mask = env.clip(proposed_position)
        position = clipped_position
        if np.any(clipped_mask):
            heading += math.pi

        offset = position - obstacle_center
        distance_to_obstacle = float(np.linalg.norm(offset))
        if distance_to_obstacle < obstacle_radius:
            if distance_to_obstacle > 1e-6:
                correction = offset / distance_to_obstacle * obstacle_radius
                position = obstacle_center + correction
            heading += math.pi / 2.0
        velocity = np.array([dx, dy], dtype=float)

        obs_raw_buffer.append(raw_obs)
        obs_norm_buffer.append(norm_obs)
        action_buffer.append(np.array([linear_cmd, angular_cmd], dtype=np.float32))
        time_buffer.append(time)

        time += dt

    return (
        np.stack(obs_raw_buffer, axis=0),
        np.stack(obs_norm_buffer, axis=0),
        np.stack(action_buffer, axis=0),
        np.asarray(time_buffer, dtype=np.float32),
    )


def _aggregate_episodes(
    controller: RSTDPController,
    env: Environment,
    *,
    episodes: int,
    steps: int,
    dt: float,
    rng: np.random.Generator,
    include_velocity: bool,
    max_linear: float,
    max_angular: float,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
    obstacle_margin: float,
    pos_center: np.ndarray,
    pos_scale: np.ndarray,
    reset_each_episode: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_raw_list: list[np.ndarray] = []
    obs_norm_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    time_list: list[np.ndarray] = []

    cumulative_time_offset = 0.0
    for episode in range(episodes):
        if reset_each_episode or episode == 0:
            controller.reset()
        (
            obs_raw,
            obs_norm,
            actions,
            times,
        ) = _simulate_episode(
            controller,
            env,
            dt=dt,
            steps=steps,
            rng=rng,
            include_velocity=include_velocity,
            max_linear=max_linear,
            max_angular=max_angular,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
            obstacle_margin=obstacle_margin,
            pos_center=pos_center,
            pos_scale=pos_scale,
        )
        obs_raw_list.append(obs_raw)
        obs_norm_list.append(obs_norm)
        action_list.append(actions)
        times = times + cumulative_time_offset
        cumulative_time_offset = times[-1] + dt
        time_list.append(times)

    return (
        np.concatenate(obs_raw_list, axis=0),
        np.concatenate(obs_norm_list, axis=0),
        np.concatenate(action_list, axis=0),
        np.concatenate(time_list, axis=0),
    )


def _build_meta(
    *,
    args: argparse.Namespace,
    obs_dim: int,
    total_steps: int,
    pos_center: np.ndarray,
    pos_scale: np.ndarray,
) -> str:
    obs_fields = ["x", "y"]
    if args.include_velocity:
        obs_fields.extend(["vx", "vy"])

    meta = {
        "schema_version": 1,
        "dt": args.dt,
        "episodes": args.episodes,
        "steps_per_episode": args.steps,
        "total_steps": total_steps,
        "obs_dim": obs_dim,
        "obs_fields": obs_fields,
        "obs_center": pos_center.tolist(),
        "obs_scale": pos_scale.tolist(),
        "velocity_scale": args.max_linear if args.include_velocity else None,
        "max_linear": args.max_linear,
        "max_angular": args.max_angular,
        "arena": {"width": args.arena_width, "height": args.arena_height},
        "obstacle": {
            "radius": args.obstacle_radius,
            "margin": args.obstacle_margin,
            "center": [0.5 * args.arena_width, 0.5 * args.arena_height],
        },
        "controller": {
            "backend": "rstdp",
            "hidden_size": args.rstdp_hidden,
            "learning_rate": args.rstdp_learning_rate,
            "weights_source": str(args.rstdp_weights) if args.rstdp_weights else None,
            "reset_each_episode": args.controller_reset_every_episode,
        },
        "seed": args.seed,
        "include_velocity": args.include_velocity,
    }
    return json.dumps(meta, indent=2)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive.")
    if args.steps <= 0:
        raise ValueError("steps must be positive.")
    if args.dt <= 0.0:
        raise ValueError("dt must be positive.")
    if args.arena_width <= 0.0 or args.arena_height <= 0.0:
        raise ValueError("Arena dimensions must be positive.")
    if args.max_linear <= 0.0:
        raise ValueError("max_linear must be positive for normalisation.")
    if args.obstacle_radius <= 0.0:
        raise ValueError("obstacle_radius must be positive.")
    if args.obstacle_margin < 0.0:
        raise ValueError("obstacle_margin must be non-negative.")

    rng = np.random.default_rng(args.seed)
    env = Environment(width=args.arena_width, height=args.arena_height)
    include_velocity = bool(args.include_velocity)

    controller = _build_teacher_controller(args, include_velocity)
    controller.reset()
    pos_center = np.array([0.5 * args.arena_width, 0.5 * args.arena_height], dtype=np.float32)
    pos_scale = np.array([0.5 * args.arena_width, 0.5 * args.arena_height], dtype=np.float32)

    obstacle_center = np.array([pos_center[0], pos_center[1]], dtype=float)

    (
        obs_raw,
        obs_norm,
        actions,
        times,
    ) = _aggregate_episodes(
        controller,
        env,
        episodes=args.episodes,
        steps=args.steps,
        dt=args.dt,
        rng=rng,
        include_velocity=include_velocity,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        obstacle_center=obstacle_center,
        obstacle_radius=args.obstacle_radius,
        obstacle_margin=args.obstacle_margin,
        pos_center=pos_center,
        pos_scale=pos_scale,
        reset_each_episode=args.controller_reset_every_episode,
    )

    meta_json = _build_meta(
        args=args,
        obs_dim=obs_norm.shape[1],
        total_steps=obs_norm.shape[0],
        pos_center=pos_center,
        pos_scale=pos_scale,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        obs=obs_norm.astype(np.float32),
        actions=actions.astype(np.float32),
        times=times.astype(np.float32),
        obs_raw=obs_raw.astype(np.float32),
        meta=np.array(meta_json, dtype=np.string_),
    )
    print(
        f"Saved {obs_norm.shape[0]} samples "
        f"(obs_dim={obs_norm.shape[1]}) to '{args.output}'.",
    )


if __name__ == "__main__":
    main()


