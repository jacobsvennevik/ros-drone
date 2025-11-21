"""Reward-modulated STDP controller for online navigation learning.

This controller implements a minimal reward-modulated STDP (R-STDP) network
that maps low-dimensional observations to velocity commands. The network uses
fixed random projections from the observation vector into a hidden spiking
layer and learns the hidden-to-output synapses using a three-factor learning
rule (pre/post eligibility traces gated by an external reward signal).

The implementation is intentionally Numpy-only to keep the update equations
transparent and lightweight for experimentation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .base import SNNController


@dataclass
class RSTDPControllerConfig:
    """Configuration parameters for :class:`RSTDPController`.

    Attributes
    ----------
    input_dim:
        Expected length of the observation vector. Observations shorter than
        this value are zero-padded; longer observations are truncated.
    hidden_size:
        Number of hidden spiking units receiving random projections from the
        observation vector.
    output_size:
        Number of output neurons. Fixed to two for (v, ω) but kept flexible for
        experimentation.
    hidden_decay:
        Exponential decay factor for hidden membrane potentials per step.
    output_decay:
        Exponential decay factor for output membrane potentials per step.
    hidden_threshold:
        Spike threshold for hidden units.
    output_threshold:
        Spike threshold for output units.
    reset_voltage:
        Amount to subtract from a membrane immediately after a spike.
    pre_trace_decay:
        Decay applied to pre-synaptic traces each control step.
    post_trace_decay:
        Decay applied to post-synaptic traces each control step.
    eligibility_decay:
        Decay factor for eligibility traces (pre/post pairings).
    learning_rate:
        Step size for R-STDP weight updates.
    reward_scale:
        Multiplicative factor applied to the scalar reward before learning.
    weight_min, weight_max:
        Bounds enforced on hidden-to-output synapses.
    input_scale:
        Gain applied to the observation vector prior to projection.
    action_gain_linear, action_gain_angular:
        Scaling factors that convert output neuron activations into velocity
        commands (m/s and rad/s respectively).
    obstacle_center, obstacle_radius, obstacle_margin:
        Parameters describing a circular obstacle used for reward shaping.
    forward_reward_gain:
        Positive reward per metre of forward progress.
    clearance_reward:
        Additional reward proportional to the clearance from the obstacle
        boundary (encourages keeping a margin).
    collision_penalty:
        Penalty applied when the agent intrudes into the obstacle margin.
    angular_penalty_gain:
        Penalty for large angular commands (encourages smoother motion).
    reward_clip:
        Maximum absolute reward magnitude before scaling.
    reward_timescale:
        Time scale for reward signals (seconds). Used to normalize reward
        scaling across different simulation timesteps. Default: 1.0 s.
    weight_init_scale:
        Standard deviation for random initial weights.
    keep_weights_on_reset:
        If True, `reset()` only clears state variables and preserves synapses.
        Otherwise, both `W_in` and `W_out` are re-sampled on reset.
    rng:
        Optional random number generator for deterministic experiments.
    """

    input_dim: int = 4
    hidden_size: int = 32
    output_size: int = 2

    hidden_decay: float = 0.9
    output_decay: float = 0.9
    hidden_threshold: float = 1.0
    output_threshold: float = 0.6
    reset_voltage: float = 1.0

    pre_trace_decay: float = 0.8
    post_trace_decay: float = 0.8
    eligibility_decay: float = 0.85

    learning_rate: float = 5e-3
    reward_scale: float = 1.0
    weight_min: float = -1.5
    weight_max: float = 1.5

    input_scale: float = 1.0
    action_gain_linear: float = 0.25
    action_gain_angular: float = 0.8

    obstacle_center: Tuple[float, float] = (0.5, 0.5)
    obstacle_radius: float = 0.15
    obstacle_margin: float = 0.05

    forward_reward_gain: float = 2.0
    clearance_reward: float = 0.2
    collision_penalty: float = -1.0
    angular_penalty_gain: float = 0.2
    reward_clip: float = 1.0
    reward_timescale: float = 1.0

    weight_init_scale: float = 0.2
    keep_weights_on_reset: bool = False
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        if self.output_size != 2:
            raise ValueError("RSTDPController currently expects exactly two outputs (v, ω).")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if not (0.0 < self.hidden_decay <= 1.0):
            raise ValueError("hidden_decay must lie in (0, 1].")
        if not (0.0 < self.output_decay <= 1.0):
            raise ValueError("output_decay must lie in (0, 1].")
        if self.weight_min >= self.weight_max:
            raise ValueError("weight_min must be strictly less than weight_max.")
        if self.obstacle_radius <= 0.0:
            raise ValueError("obstacle_radius must be positive.")
        if self.obstacle_margin < 0.0:
            raise ValueError("obstacle_margin must be non-negative.")
        if self.reward_clip <= 0.0:
            raise ValueError("reward_clip must be positive.")


class RSTDPController(SNNController):
    """Online controller that learns velocity commands using R-STDP.

    The controller expects observations containing at least the agent position
    ``(x, y)`` in metres. Optional components such as velocity can be appended
    to form the configured ``input_dim``. Internally, observations are projected
    into a hidden spiking layer whose activity is read out by two output neurons
    representing linear and angular velocity proposals.
    """

    def __init__(self, config: Optional[RSTDPControllerConfig] = None) -> None:
        self.config = config or RSTDPControllerConfig()
        self._rng = self.config.rng or np.random.default_rng()

        self._w_in = np.zeros((self.config.hidden_size, self.config.input_dim), dtype=float)
        self._w_out = np.zeros((self.config.output_size, self.config.hidden_size), dtype=float)
        self._mem_hidden = np.zeros(self.config.hidden_size, dtype=float)
        self._mem_out = np.zeros(self.config.output_size, dtype=float)
        self._pre_trace = np.zeros(self.config.hidden_size, dtype=float)
        self._post_trace = np.zeros(self.config.output_size, dtype=float)
        self._eligibility = np.zeros((self.config.output_size, self.config.hidden_size), dtype=float)

        self._last_position: Optional[np.ndarray] = None
        self._last_action = np.zeros(self.config.output_size, dtype=float)
        self._step_count = 0
        self._episode_reward = 0.0

        self._initialise_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset state between episodes."""

        if not self.config.keep_weights_on_reset:
            self._initialise_weights()

        self._mem_hidden.fill(0.0)
        self._mem_out.fill(0.0)
        self._pre_trace.fill(0.0)
        self._post_trace.fill(0.0)
        self._eligibility.fill(0.0)

        self._last_position = None
        self._last_action.fill(0.0)
        self._step_count = 0
        self._episode_reward = 0.0

    def step(self, obs: np.ndarray, dt: float) -> np.ndarray:
        """Advance the controller dynamics by one step."""

        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        observation = self._prepare_observation(obs)

        hidden_spikes = self._update_hidden(observation)
        action = self._update_output(hidden_spikes)

        reward = self._compute_reward(observation, action, dt)
        self._apply_learning(reward, dt)

        self._last_action = action
        self._step_count += 1
        self._episode_reward += reward

        return action.copy()

    # ------------------------------------------------------------------
    # Diagnostics & serialization helpers
    # ------------------------------------------------------------------
    @property
    def episode_reward(self) -> float:
        """Return the accumulated reward for the current episode."""

        return self._episode_reward

    def export_state(self) -> Dict[str, np.ndarray]:
        """Return a copy of learnable parameters for logging/checkpointing."""

        return {
            "w_in": self._w_in.copy(),
            "w_out": self._w_out.copy(),
        }

    def import_state(self, state: Dict[str, np.ndarray]) -> None:
        """Load learnable parameters from a state dictionary."""

        w_in = np.asarray(state.get("w_in"))
        w_out = np.asarray(state.get("w_out"))
        if w_in.shape != self._w_in.shape or w_out.shape != self._w_out.shape:
            raise ValueError("State shapes do not match controller configuration.")
        self._w_in[...] = w_in
        self._w_out[...] = w_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialise_weights(self) -> None:
        scale = self.config.weight_init_scale
        self._w_in = self._rng.normal(scale=scale, size=self._w_in.shape)
        self._w_out = self._rng.normal(scale=scale, size=self._w_out.shape)
        np.clip(self._w_out, self.config.weight_min, self.config.weight_max, out=self._w_out)

    def _prepare_observation(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=float).reshape(-1)
        if arr.size < self.config.input_dim:
            padded = np.zeros(self.config.input_dim, dtype=float)
            padded[: arr.size] = arr
            arr = padded
        elif arr.size > self.config.input_dim:
            arr = arr[: self.config.input_dim]

        return arr * self.config.input_scale

    def _update_hidden(self, observation: np.ndarray) -> np.ndarray:
        self._mem_hidden *= self.config.hidden_decay
        self._mem_hidden += self._w_in @ observation

        spikes = self._mem_hidden >= self.config.hidden_threshold
        if np.any(spikes):
            self._mem_hidden[spikes] -= self.config.reset_voltage

        spike_vector = spikes.astype(float)
        self._pre_trace *= self.config.pre_trace_decay
        self._pre_trace += spike_vector
        return spike_vector

    def _update_output(self, hidden_spikes: np.ndarray) -> np.ndarray:
        self._mem_out *= self.config.output_decay
        self._mem_out += self._w_out @ hidden_spikes

        post_spikes = self._mem_out >= self.config.output_threshold
        if np.any(post_spikes):
            self._mem_out[post_spikes] -= self.config.reset_voltage

        post_vector = post_spikes.astype(float)
        self._post_trace *= self.config.post_trace_decay
        self._post_trace += post_vector

        # Eligibility trace accumulates outer product of post activity and pre traces.
        self._eligibility *= self.config.eligibility_decay
        self._eligibility += np.outer(post_vector, self._pre_trace)

        # Continuous readout via tanh of membrane voltages.
        linear_activation = np.tanh(self._mem_out[0])
        angular_activation = np.tanh(self._mem_out[1])
        action = np.array(
            [
                self.config.action_gain_linear * linear_activation,
                self.config.action_gain_angular * angular_activation,
            ],
            dtype=float,
        )
        return action

    def _compute_reward(self, observation: np.ndarray, action: np.ndarray, dt: float) -> float:
        position = observation[:2]
        reward = 0.0

        if self._last_position is not None:
            delta = position - self._last_position
            distance = float(np.linalg.norm(delta))
            reward += self.config.forward_reward_gain * distance

        obstacle_center = np.asarray(self.config.obstacle_center, dtype=float)
        distance_to_obstacle = float(np.linalg.norm(position - obstacle_center))
        clearance = distance_to_obstacle - self.config.obstacle_radius

        if clearance < self.config.obstacle_margin:
            # Penalize intrusion into the margin (or the obstacle itself).
            penetration = self.config.obstacle_margin - clearance
            reward += self.config.collision_penalty * (1.0 + penetration / max(dt, 1e-6))
        else:
            reward += self.config.clearance_reward * np.tanh(
                (clearance - self.config.obstacle_margin) / (self.config.obstacle_margin + 1e-6)
            )

        # Penalize aggressive turns.
        angular_cost = self.config.angular_penalty_gain * abs(action[1]) / max(
            self.config.action_gain_angular, 1e-6
        )
        reward -= angular_cost

        self._last_position = position.copy()

        reward = float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))
        return reward * self.config.reward_scale

    def _apply_learning(self, reward: float, dt: float) -> None:
        """Apply R-STDP learning with reward scaling for time invariance.
        
        Parameters
        ----------
        reward:
            Raw reward signal (per timestep).
        dt:
            Time step in seconds. Used to scale reward to reward_timescale.
        """
        if reward == 0.0:
            # Still decay eligibility to keep traces bounded.
            self._eligibility *= self.config.eligibility_decay
            return

        # Scale reward by dt/reward_timescale to ensure invariance across simulation rates
        # If reward is per-second and we call this every dt seconds, scale appropriately
        scaled_reward = reward * (dt / self.config.reward_timescale)
        
        delta_w = self.config.learning_rate * scaled_reward * self._eligibility
        self._w_out += delta_w
        np.clip(self._w_out, self.config.weight_min, self.config.weight_max, out=self._w_out)

