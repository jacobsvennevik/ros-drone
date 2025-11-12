"""Abstract controller interfaces for hippocampus-driven agents."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SNNController(ABC):
    """Abstract interface for controllers driven by spiking neural networks.

    Controllers consume observations from the environment and advance their
    internal dynamics over discrete simulation steps. They produce an action
    vector that can be used to influence the agent or robot. Implementations
    must remain framework-agnostic and avoid ROS-specific dependencies.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the controller's internal state before a new episode/run."""

    @abstractmethod
    def step(self, obs: np.ndarray, dt: float) -> np.ndarray:
        """Advance the controller by one time step.

        Parameters
        ----------
        obs:
            Observation vector at the current time step (e.g. agent position,
            velocity, and other sensory features). Implementations should
            document the expected contents.
        dt:
            Simulation time-step duration in seconds.

        Returns
        -------
        np.ndarray
            Action vector for the agent. Implementations define semantics,
            such as velocity commands or control inputs.
        """
