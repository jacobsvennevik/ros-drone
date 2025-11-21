"""Decision decoding from SNN outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .data_structures import (
    FeatureVector,
    LocalContext,
    Mission,
    PolicyDecision,
    ActionProposal,
)


@dataclass
class DecoderConfig:
    """Configuration for decision decoder."""

    max_linear: float = 0.3
    max_angular: float = 1.0
    max_vertical: Optional[float] = None


class DecisionDecoder:
    """Decodes SNN output to PolicyDecision."""

    def __init__(
        self,
        max_linear: float = 0.3,
        max_angular: float = 1.0,
        max_vertical: Optional[float] = None,
    ):
        """Initialize decoder.

        Parameters
        ----------
        max_linear:
            Maximum linear velocity (m/s).
        max_angular:
            Maximum angular velocity (rad/s).
        max_vertical:
            Maximum vertical velocity (m/s) for 3D.
        """
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.max_vertical = max_vertical

    def decode(
        self,
        snn_output: torch.Tensor,  # (batch, output_dim) or (time_steps, batch, output_dim)
        features: FeatureVector,
        local_context: LocalContext,
        mission: Mission,
    ) -> PolicyDecision:
        """Decode SNN output to policy decision.

        Parameters
        ----------
        snn_output:
            Network output, shape (batch, output_dim) or (time_steps, batch, output_dim).
            If multi-step, use the last timestep.
        features:
            Input features (for context).
        local_context:
            Local context (for waypoint selection).
        mission:
            Mission goals (for validation).

        Returns
        -------
        PolicyDecision
            Policy decision.
        """
        if torch is None:
            raise ImportError("PyTorch required for DecisionDecoder")

        # Handle multi-step output
        if snn_output.dim() == 3:
            snn_output = snn_output[-1]  # Use last timestep

        # Extract actions (assumed to be in [-1, 1] from tanh)
        action_np = snn_output.squeeze(0).cpu().numpy()

        # Scale to physical units
        v = float(action_np[0] * self.max_linear)
        omega = float(action_np[1] * self.max_angular)
        vz = None
        if len(action_np) > 2 and self.max_vertical is not None:
            vz = float(action_np[2] * self.max_vertical)

        # Compute confidence from output magnitude
        # Higher magnitude = more confident
        output_magnitude = np.linalg.norm(action_np)
        confidence = float(np.clip(output_magnitude, 0.0, 1.0))

        # Select waypoint (if using hierarchical planning)
        waypoint = self._select_waypoint(features, local_context, mission)

        return PolicyDecision(
            next_waypoint=waypoint,
            action_proposal=ActionProposal(v=v, omega=omega, vz=vz),
            confidence=confidence,
            reason="snn",
        )

    def _select_waypoint(
        self,
        features: FeatureVector,
        local_context: LocalContext,
        mission: Mission,
    ) -> Optional[int]:
        """Select target waypoint from graph (optional, for hierarchical planning).

        For reactive control (Milestone A/B), return None.

        Parameters
        ----------
        features:
            Feature vector.
        local_context:
            Local context.
        mission:
            Mission.

        Returns
        -------
        Optional[int]
            Waypoint node ID or None.
        """
        # For reactive control, return None
        # For hierarchical planning (Milestone C+), implement waypoint selection
        return None


def compute_confidence(
    snn_output: torch.Tensor,
    spike_rate: Optional[float] = None,
    feature_quality: float = 1.0,
) -> float:
    """Compute confidence score from SNN outputs.

    Parameters
    ----------
    snn_output:
        SNN output tensor.
    spike_rate:
        Optional spike rate (Hz).
    feature_quality:
        Feature quality factor [0, 1].

    Returns
    -------
    float
        Confidence score [0, 1].
    """
    if torch is None:
        raise ImportError("PyTorch required for compute_confidence")

    # Base confidence from output magnitude
    output_mag = torch.norm(snn_output).item()
    base_confidence = min(output_mag, 1.0)

    # Adjust by spike rate (if available)
    if spike_rate is not None:
        # Normalize spike rate to [0, 1]
        rate_confidence = min(spike_rate / 50.0, 1.0)  # Assume 50 Hz is "high"
        base_confidence = 0.7 * base_confidence + 0.3 * rate_confidence

    # Adjust by feature quality
    final_confidence = base_confidence * feature_quality

    return float(np.clip(final_confidence, 0.0, 1.0))

