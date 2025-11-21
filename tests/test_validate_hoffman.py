import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable when tests run from arbitrary locations
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hippocampus_core.env import CircularObstacle, Environment

from experiments.validate_hoffman_2016 import (
    _generate_obstacle_ring_positions,
    _generate_ring_spoke_positions,
    _sample_uniform_positions,
    run_learning_experiment,
)


@pytest.fixture
def obstacle_env():
    obstacle = CircularObstacle(center_x=0.5, center_y=0.5, radius=0.15)
    env = Environment(width=1.0, height=1.0, obstacles=[obstacle])
    return env, obstacle


def test_sample_uniform_positions_respect_bounds():
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(0)

    samples = _sample_uniform_positions(env, num_points=10, rng=rng)
    assert samples.shape == (10, 2)
    for point in samples:
        assert env.contains(tuple(point))


def test_generate_obstacle_ring_positions_wraps_obstacle(obstacle_env):
    env, obstacle = obstacle_env
    rng = np.random.default_rng(1)

    centers = _generate_obstacle_ring_positions(
        env=env,
        obstacle=obstacle,
        num_cells=12,
        rng=rng,
        ring_fraction=0.5,
        ring_offset=0.03,
        ring_jitter=0.005,
    )

    assert centers.shape == (12, 2)
    for point in centers:
        assert env.contains(tuple(point))
        dx = point[0] - obstacle.center_x
        dy = point[1] - obstacle.center_y
        distance = np.sqrt(dx * dx + dy * dy)
        assert distance >= obstacle.radius  # stays outside obstacle


def test_generate_ring_spoke_positions_adds_bridges(obstacle_env):
    env, obstacle = obstacle_env
    rng = np.random.default_rng(7)
    centers = _generate_ring_spoke_positions(
        env=env,
        obstacle=obstacle,
        num_cells=24,
        rng=rng,
        ring_fraction=0.4,
        spoke_fraction=0.3,
        ring_offset=0.03,
        ring_jitter=0.002,
        spoke_extension=0.12,
        spoke_jitter=0.002,
        num_spokes=4,
    )
    assert centers.shape == (24, 2)
    ring_radius = obstacle.radius + 0.03
    spoke_outer = ring_radius + 0.12

    distances = np.linalg.norm(centers - np.array([obstacle.center_x, obstacle.center_y]), axis=1)
    assert np.any(distances >= ring_radius - 1e-3)
    assert np.any(distances >= spoke_outer - 0.02)


def test_run_learning_experiment_accepts_custom_parameters(obstacle_env):
    env, obstacle = obstacle_env
    rng = np.random.default_rng(2)
    centers = _generate_obstacle_ring_positions(
        env=env,
        obstacle=obstacle,
        num_cells=8,
        rng=rng,
        ring_fraction=0.75,
        ring_offset=0.02,
        ring_jitter=0.0,
    )

    results = run_learning_experiment(
        env=env,
        integration_window=None,
        duration_seconds=1.0,
        dt=0.05,
        num_place_cells=centers.shape[0],
        sigma=0.1,
        seed=123,
        expected_b1=1,
        coactivity_threshold=8.0,
        max_edge_distance=0.15,
        place_cell_positions=centers,
    )

    assert set(results.keys()) == {"times", "edges", "components", "betti_0", "betti_1", "betti_2"}
    assert len(results["times"]) > 0
    assert len(results["edges"]) == len(results["times"])
    assert len(results["betti_0"]) == len(results["times"])


def test_run_learning_experiment_orbit_trajectory(obstacle_env):
    env, obstacle = obstacle_env
    centers = _generate_obstacle_ring_positions(
        env=env,
        obstacle=obstacle,
        num_cells=6,
        rng=np.random.default_rng(3),
        ring_fraction=1.0,
        ring_offset=0.02,
        ring_jitter=0.0,
    )

    results = run_learning_experiment(
        env=env,
        integration_window=None,
        duration_seconds=1.0,
        dt=0.05,
        num_place_cells=centers.shape[0],
        sigma=0.1,
        seed=321,
        expected_b1=1,
        coactivity_threshold=4.0,
        max_edge_distance=0.2,
        place_cell_positions=centers,
        trajectory_mode="orbit_then_random",
        trajectory_params={"orbit_duration": 0.5, "orbit_radius": 0.2, "orbit_speed": 0.4},
    )

    assert results["times"]
    assert len(results["edges"]) == len(results["times"])

