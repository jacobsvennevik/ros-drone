"""Run an online R-STDP navigation experiment in the toy 2D arena.

This script couples the :class:`hippocampus_core.controllers.rstdp_controller.RSTDPController`
with a simple unicycle agent in a rectangular arena containing a circular
central obstacle. The controller learns online via reward-modulated STDP while
attempting to navigate without colliding with the obstacle.

Example:
    python -m experiments.rstdp_online_run --episodes 20 --steps 1500 --dt 0.05
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from hippocampus_core.controllers.rstdp_controller import (
    RSTDPController,
    RSTDPControllerConfig,
)
from hippocampus_core.env import Environment


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
    # Fallback: place on boundary if sampling fails (highly unlikely).
    return np.array([bounds.min_x + 0.05 * bounds.width, bounds.min_y + 0.05 * bounds.height])


def _simulate_episode(
    controller: RSTDPController,
    env: Environment,
    *,
    dt: float,
    steps: int,
    rng: np.random.Generator,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
    obstacle_margin: float,
) -> Tuple[Dict[str, float], List[Tuple[float, float, float, float, float]]]:
    """Run one episode and return summary metrics plus per-step log data."""

    controller.reset()

    position = _sample_start_position(rng, env, obstacle_center, obstacle_radius + obstacle_margin)
    heading = rng.uniform(-math.pi, math.pi)
    velocity = np.zeros(2, dtype=float)

    total_collision_steps = 0
    safe_steps = 0
    speed_accumulator = 0.0

    step_log: List[Tuple[float, float, float, float, float]] = []
    prev_reward_total = controller.episode_reward

    for step in range(steps):
        observation = np.array([position[0], position[1], velocity[0], velocity[1]], dtype=float)

        action = controller.step(observation, dt)
        step_reward = controller.episode_reward - prev_reward_total
        prev_reward_total = controller.episode_reward

        linear_vel = float(action[0])
        angular_vel = float(action[1])

        # Unicycle integration.
        heading += angular_vel * dt
        dx = linear_vel * math.cos(heading)
        dy = linear_vel * math.sin(heading)
        proposed_position = position + np.array([dx, dy]) * dt

        clipped_position, clipped_mask = env.clip(proposed_position)
        position = clipped_position
        if np.any(clipped_mask):
            # Bounce off the wall by mirroring heading.
            heading += math.pi

        # Obstacle handling.
        offset = position - obstacle_center
        distance_to_obstacle = float(np.linalg.norm(offset))
        if distance_to_obstacle < obstacle_radius:
            total_collision_steps += 1
            if distance_to_obstacle > 1e-6:
                correction = offset / distance_to_obstacle * obstacle_radius
                position = obstacle_center + correction
            heading += math.pi / 2.0
        else:
            clearance = distance_to_obstacle - obstacle_radius
            if clearance >= obstacle_margin:
                safe_steps += 1

        velocity = np.array([dx, dy], dtype=float)
        speed = float(np.linalg.norm(velocity))
        speed_accumulator += speed

        step_time = step * dt
        step_log.append((step_time, step_reward, linear_vel, angular_vel, speed))

    metrics = {
        "episode_return": controller.episode_reward,
        "mean_speed": speed_accumulator / max(steps, 1),
        "collision_fraction": total_collision_steps / max(steps, 1),
        "safe_fraction": safe_steps / max(steps, 1),
    }
    return metrics, step_log


def _write_step_log(path: Path, rows: Iterable[Tuple[float, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "reward", "linear", "angular", "speed"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R-STDP controller online learning experiments.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run.")
    parser.add_argument("--steps", type=int, default=1500, help="Steps per episode.")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation time step.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--keep-weights",
        action="store_true",
        help="Preserve learned weights across episode resets for continual learning.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/rstdp"),
        help="Directory for per-episode CSV logs and weight checkpoints.",
    )
    parser.add_argument(
        "--save-weights-every",
        type=int,
        default=5,
        help="Save controller weights every N episodes (0 to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive.")
    if args.steps <= 0:
        raise ValueError("steps must be positive.")
    if args.dt <= 0.0:
        raise ValueError("dt must be positive.")

    rng = np.random.default_rng(args.seed)

    env = Environment(width=1.0, height=1.0)
    obstacle_center = np.array([0.5 * env.bounds.width, 0.5 * env.bounds.height], dtype=float)
    obstacle_radius = 0.15
    obstacle_margin = 0.05

    controller_config = RSTDPControllerConfig(
        input_dim=4,
        hidden_size=48,
        output_size=2,
        obstacle_center=(float(obstacle_center[0]), float(obstacle_center[1])),
        obstacle_radius=obstacle_radius,
        obstacle_margin=obstacle_margin,
        keep_weights_on_reset=args.keep_weights,
        rng=np.random.default_rng(args.seed + 1),
    )
    controller = RSTDPController(config=controller_config)

    episode_returns = []
    mean_speeds = []
    collision_rates = []

    for episode in range(args.episodes):
        metrics, step_log = _simulate_episode(
            controller,
            env,
            dt=args.dt,
            steps=args.steps,
            rng=rng,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
            obstacle_margin=obstacle_margin,
        )

        episode_returns.append(metrics["episode_return"])
        mean_speeds.append(metrics["mean_speed"])
        collision_rates.append(metrics["collision_fraction"])

        log_path = args.log_dir / f"episode_{episode:03d}.csv"
        _write_step_log(log_path, step_log)

        if args.save_weights_every > 0 and (episode + 1) % args.save_weights_every == 0:
            state = controller.export_state()
            weights_path = args.log_dir / f"weights_ep{episode + 1:03d}.npz"
            np.savez(weights_path, **state)

        print(
            f"Episode {episode + 1:03d} | return={metrics['episode_return']:.3f} "
            f"| mean|action|≈{metrics['mean_speed']:.3f} m/s "
            f"| collision_fraction={metrics['collision_fraction']:.3f} "
            f"| safe_fraction={metrics['safe_fraction']:.3f}"
        )

    print("\nSummary over episodes")
    print(f"Average return: {np.mean(episode_returns):.3f} ± {np.std(episode_returns):.3f}")
    print(f"Average speed:  {np.mean(mean_speeds):.3f} ± {np.std(mean_speeds):.3f} m/s")
    print(f"Collision rate: {np.mean(collision_rates):.3f}")


if __name__ == "__main__":
    main()


