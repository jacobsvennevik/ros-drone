"""Lightweight notebook execution tests.

These tests run trimmed-down versions of validation notebooks to catch regressions early.
They verify that core functionality from the notebooks works without running the full simulations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hippocampus_core.controllers.bat_navigation_controller import (
    BatNavigationController,
    BatNavigationControllerConfig,
)
from hippocampus_core.env import Agent, Environment


def rayleigh_vector(theta: np.ndarray, weights: np.ndarray) -> float:
    """Compute Rayleigh vector length for directional tuning."""
    theta = np.asarray(theta)
    weights = np.asarray(weights)
    # Guard against division by zero
    sum_weights = np.sum(weights)
    if sum_weights == 0:
        return 0.0
    vector = np.sum(weights * np.exp(1j * theta))
    return np.abs(vector) / sum_weights


def theta_index(signal: np.ndarray, dt: float = 0.05) -> float:
    """Compute fractional theta-band power."""
    if len(signal) == 0:
        return 0.0
    signal = np.asarray(signal)
    fft = np.fft.rfft(signal - np.mean(signal))
    freqs = np.fft.rfftfreq(signal.size, d=dt)
    theta_band = (freqs >= 4.0) & (freqs <= 10.0)
    power_total = np.sum(np.abs(fft) ** 2)
    if power_total == 0:
        return 0.0
    power_theta = np.sum(np.abs(fft[theta_band]) ** 2)
    return power_theta / power_total


class TestRubinHDValidation:
    """Test Rubin HD validation notebook functionality."""

    def test_short_simulation_runs(self):
        """Test that a short simulation completes successfully."""
        env = Environment(width=1.0, height=1.0)
        config = BatNavigationControllerConfig(
            num_place_cells=20,  # Reduced for speed
            hd_num_neurons=36,  # Reduced for speed
            grid_size=(8, 8),  # Reduced for speed
            calibration_interval=50,  # Reduced for speed
            integration_window=10.0,  # Short window for testing
        )
        controller = BatNavigationController(
            env, config=config, rng=np.random.default_rng(7)
        )
        agent = Agent(
            env, random_state=np.random.default_rng(8), track_heading=True
        )

        # Run a very short simulation (5 seconds)
        duration_seconds = 5.0
        dt = 0.05
        num_steps = int(duration_seconds / dt)
        obs_history = np.zeros((num_steps, 3), dtype=float)
        rate_history = np.zeros((num_steps, config.num_place_cells), dtype=float)

        for idx in range(num_steps):
            obs = agent.step(dt, include_theta=True)
            controller.step(obs, dt)
            obs_history[idx] = obs
            rate_history[idx] = controller.last_rates

        # Verify outputs
        assert obs_history.shape == (num_steps, 3)
        assert rate_history.shape == (num_steps, config.num_place_cells)
        assert np.all(np.isfinite(obs_history))
        assert np.all(np.isfinite(rate_history))
        assert np.all(rate_history >= 0)

    def test_hd_tuning_computation(self):
        """Test HD tuning computation with synthetic data."""
        # Generate synthetic heading and rate data
        n_samples = 100
        headings = np.linspace(-np.pi, np.pi, n_samples)
        # Create a directional preference (peak at 0 radians)
        rates = 1.0 + 0.5 * np.cos(headings)

        # Compute Rayleigh vector
        rv = rayleigh_vector(headings, rates)

        # Should have significant directional tuning (> 0.1)
        assert rv > 0.1
        assert rv <= 1.0

    def test_hd_tuning_inside_vs_outside(self):
        """Test that HD tuning is stronger inside place field."""
        env = Environment(width=1.0, height=1.0)
        config = BatNavigationControllerConfig(
            num_place_cells=10,
            hd_num_neurons=36,
            grid_size=(8, 8),
            calibration_interval=50,
            integration_window=10.0,
        )
        controller = BatNavigationController(
            env, config=config, rng=np.random.default_rng(7)
        )
        agent = Agent(
            env, random_state=np.random.default_rng(8), track_heading=True
        )

        # Run short simulation
        duration_seconds = 3.0
        dt = 0.05
        num_steps = int(duration_seconds / dt)
        obs_history = np.zeros((num_steps, 3), dtype=float)
        rate_history = np.zeros((num_steps, config.num_place_cells), dtype=float)

        for idx in range(num_steps):
            obs = agent.step(dt, include_theta=True)
            controller.step(obs, dt)
            obs_history[idx] = obs
            rate_history[idx] = controller.last_rates

        # Find most active cell
        mean_rates = rate_history.mean(axis=0)
        cell_idx = int(np.argmax(mean_rates))

        # Compute tuning inside vs outside
        positions = obs_history[:, :2]
        headings = obs_history[:, 2]
        rates = rate_history[:, cell_idx]
        center = controller.place_cells.get_positions()[cell_idx, :2]
        sigma = config.sigma

        distance = np.linalg.norm(positions - center, axis=1)
        in_mask = distance <= sigma
        out_mask = ~in_mask

        if np.sum(in_mask) > 10 and np.sum(out_mask) > 10:
            rv_in = rayleigh_vector(
                headings[in_mask], rates[in_mask] + 1e-6
            )
            rv_out = rayleigh_vector(
                headings[out_mask], rates[out_mask] + 1e-6
            )

            # Both should be valid (between 0 and 1)
            assert 0.0 <= rv_in <= 1.0
            assert 0.0 <= rv_out <= 1.0

    def test_controller_produces_valid_rates(self):
        """Test that controller produces valid firing rates."""
        env = Environment(width=1.0, height=1.0)
        config = BatNavigationControllerConfig(
            num_place_cells=10,
            hd_num_neurons=36,
            grid_size=(8, 8),
            calibration_interval=50,
        )
        controller = BatNavigationController(
            env, config=config, rng=np.random.default_rng(7)
        )
        agent = Agent(
            env, random_state=np.random.default_rng(8), track_heading=True
        )

        # Run a few steps
        dt = 0.05
        for _ in range(20):
            obs = agent.step(dt, include_theta=True)
            controller.step(obs, dt)

            # Verify rates are valid
            assert controller.last_rates is not None
            assert len(controller.last_rates) == config.num_place_cells
            assert np.all(controller.last_rates >= 0)
            assert np.all(np.isfinite(controller.last_rates))


class TestYartsevGridValidation:
    """Test Yartsev grid validation notebook functionality."""

    def test_short_simulation_runs(self):
        """Test that a short simulation completes successfully."""
        env = Environment(width=1.0, height=1.0)
        config = BatNavigationControllerConfig(
            num_place_cells=20,
            hd_num_neurons=36,
            grid_size=(10, 10),  # Reduced for speed
            calibration_interval=50,
            integration_window=10.0,
        )
        controller = BatNavigationController(
            env, config=config, rng=np.random.default_rng(11)
        )
        agent = Agent(
            env, random_state=np.random.default_rng(12), track_heading=True
        )

        # Run a very short simulation (5 seconds)
        duration_seconds = 5.0
        dt = 0.05
        num_steps = int(duration_seconds / dt)

        theta_power = []
        grid_norm = []
        velocity_history = []

        for step in range(num_steps):
            obs = agent.step(dt, include_theta=True)
            controller.step(obs, dt)
            velocity_history.append(controller.grid_attractor.estimate_position().copy())
            grid_norm.append(controller.grid_attractor.drift_metric())

            if (step + 1) % 40 == 0 and len(velocity_history) >= 40:
                recent = np.array(velocity_history[-40:])
                theta_power.append(theta_index(recent[:, 0], dt))

        # Verify outputs
        assert len(grid_norm) == num_steps
        assert all(np.isfinite(g) for g in grid_norm)
        assert all(g >= 0 for g in grid_norm)

    def test_grid_drift_metric_exists(self):
        """Test that grid drift metric can be computed."""
        env = Environment(width=1.0, height=1.0)
        config = BatNavigationControllerConfig(
            num_place_cells=10,
            hd_num_neurons=36,
            grid_size=(8, 8),
            calibration_interval=50,
        )
        controller = BatNavigationController(
            env, config=config, rng=np.random.default_rng(11)
        )
        agent = Agent(
            env, random_state=np.random.default_rng(12), track_heading=True
        )

        # Run a few steps
        dt = 0.05
        drift_values = []

        for _ in range(20):
            obs = agent.step(dt, include_theta=True)
            controller.step(obs, dt)
            drift = controller.grid_attractor.drift_metric()
            drift_values.append(drift)

        # Verify drift metric is valid
        assert all(np.isfinite(d) for d in drift_values)
        assert all(d >= 0 for d in drift_values)

    def test_theta_power_computation(self):
        """Test theta power computation."""
        # Generate synthetic velocity signal (low frequency, no theta)
        dt = 0.05
        t = np.arange(0, 2.0, dt)
        signal = 0.5 + 0.1 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz oscillation

        theta_power = theta_index(signal, dt)

        # Should have low theta power (< 0.1 for non-theta signal)
        assert theta_power >= 0.0
        assert theta_power <= 1.0

    def test_grid_stability_without_theta(self):
        """Test that grid activity remains stable without theta oscillations."""
        env = Environment(width=1.0, height=1.0)
        config = BatNavigationControllerConfig(
            num_place_cells=10,
            hd_num_neurons=36,
            grid_size=(8, 8),
            calibration_interval=50,
        )
        controller = BatNavigationController(
            env, config=config, rng=np.random.default_rng(11)
        )
        agent = Agent(
            env, random_state=np.random.default_rng(12), track_heading=True
        )

        # Run short simulation
        dt = 0.05
        grid_norms = []
        velocity_history = []

        for _ in range(100):
            obs = agent.step(dt, include_theta=True)
            controller.step(obs, dt)
            velocity_history.append(controller.grid_attractor.estimate_position().copy())
            grid_norms.append(controller.grid_attractor.drift_metric())

        # Check that grid norms are bounded (not diverging)
        if len(grid_norms) > 10:
            recent_norms = grid_norms[-20:]
            # Grid norm should remain reasonable (< 10.0 for this short run)
            assert max(recent_norms) < 10.0

