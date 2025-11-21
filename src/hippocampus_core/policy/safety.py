"""Action Arbitration & Safety for filtering policy decisions."""
from __future__ import annotations

from typing import Optional
import numpy as np

from .data_structures import (
    PolicyDecision,
    SafeCommand,
    RobotState,
    GraphSnapshot,
    Mission,
)


class GraphStalenessDetector:
    """Detects when topological graph is stale."""

    def __init__(
        self,
        warning_threshold: float = 2.0,  # seconds
        stale_threshold: float = 5.0,
        critical_threshold: float = 10.0,
    ):
        """Initialize staleness detector.

        Parameters
        ----------
        warning_threshold:
            Time before warning (seconds).
        stale_threshold:
            Time before marking as stale (seconds).
        critical_threshold:
            Time before critical (seconds).
        """
        self.warning_threshold = warning_threshold
        self.stale_threshold = stale_threshold
        self.critical_threshold = critical_threshold

    def check_staleness(
        self,
        graph_snapshot: GraphSnapshot,
        current_time: float,
    ) -> dict:
        """Check if graph is stale.

        Parameters
        ----------
        graph_snapshot:
            Current graph snapshot.
        current_time:
            Current time.

        Returns
        -------
        dict
            Staleness status with keys: is_stale, time_since_update,
            staleness_level, recommended_action.
        """
        time_since_update = current_time - graph_snapshot.meta.last_updated

        if time_since_update < self.warning_threshold:
            level = "fresh"
            action = "continue"
        elif time_since_update < self.stale_threshold:
            level = "warning"
            action = "degrade"
        elif time_since_update < self.critical_threshold:
            level = "stale"
            action = "hold"
        else:
            level = "critical"
            action = "estop"

        return {
            "is_stale": time_since_update >= self.stale_threshold,
            "time_since_update": time_since_update,
            "staleness_level": level,
            "recommended_action": action,
        }


class ActionArbitrationSafety:
    """Filters policy decisions through safety constraints."""

    def __init__(
        self,
        max_linear: float = 0.3,
        max_angular: float = 1.0,
        max_vertical: Optional[float] = None,
        collision_margin: float = 0.1,
    ):
        """Initialize safety arbitrator.

        Parameters
        ----------
        max_linear:
            Maximum linear velocity (m/s).
        max_angular:
            Maximum angular velocity (rad/s).
        max_vertical:
            Maximum vertical velocity (m/s) for 3D.
        collision_margin:
            Collision margin (m).
        """
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.max_vertical = max_vertical
        self.collision_margin = collision_margin
        self.staleness_detector = GraphStalenessDetector()

    def filter(
        self,
        decision: PolicyDecision,
        robot_state: RobotState,
        graph_snapshot: GraphSnapshot,
        mission: Mission,
    ) -> SafeCommand:
        """Filter policy decision through safety constraints.

        Parameters
        ----------
        decision:
            Policy decision to filter.
        robot_state:
            Current robot state.
        graph_snapshot:
            Current graph snapshot.
        mission:
            Current mission.

        Returns
        -------
        SafeCommand
            Safe command after filtering.
        """
        import time

        start_time = time.perf_counter()

        # 1. Check graph staleness
        staleness = self.staleness_detector.check_staleness(
            graph_snapshot,
            robot_state.time,
        )
        decision = self._apply_staleness_degradation(decision, staleness)

        # 2. Check constraints
        decision = self._check_constraints(decision, robot_state, mission)

        # 3. Rate limiting
        decision = self._apply_rate_limiting(decision, robot_state)

        # 4. Hard limits
        v = np.clip(decision.action_proposal.v, -self.max_linear, self.max_linear)
        omega = np.clip(decision.action_proposal.omega, -self.max_angular, self.max_angular)
        
        # 3D: clamp vertical velocity
        vz = None
        if decision.action_proposal.vz is not None and self.max_vertical is not None:
            vz = np.clip(decision.action_proposal.vz, -self.max_vertical, self.max_vertical)

        # Check if clamped
        clamped = (
            abs(v) != abs(decision.action_proposal.v)
            or abs(omega) != abs(decision.action_proposal.omega)
            or (vz is not None and decision.action_proposal.vz is not None
                and abs(vz) != abs(decision.action_proposal.vz))
        )

        # Build safe command
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if vz is not None:
            cmd = (v, omega, vz)
        else:
            cmd = (v, omega)

        return SafeCommand(
            cmd=cmd,
            safety_flags={
                "clamped": clamped,
                "slowed": staleness["staleness_level"] != "fresh",
                "stop": staleness["recommended_action"] in ["hold", "estop"],
            },
            latency_ms=elapsed_ms,
        )

    def _apply_staleness_degradation(
        self,
        decision: PolicyDecision,
        staleness: dict,
    ) -> PolicyDecision:
        """Apply degradation based on staleness."""
        action = staleness["recommended_action"]

        if action == "continue":
            return decision
        elif action == "degrade":
            # Reduce velocity by 50%
            decision.action_proposal.v *= 0.5
            decision.action_proposal.omega *= 0.5
            decision.confidence *= 0.7
            decision.reason = "graph_stale_warning"
        elif action == "hold":
            # Zero velocity and freeze policy state
            decision.action_proposal.v = 0.0
            decision.action_proposal.omega = 0.0
            decision.confidence = 0.0
            decision.reason = "graph_stale"
            # Note: Policy service should call freeze() when staleness detected
            # This is handled at the service level, not in the arbitrator
        elif action == "estop":
            # Emergency stop and freeze policy state
            decision.action_proposal.v = 0.0
            decision.action_proposal.omega = 0.0
            if decision.action_proposal.vz is not None:
                decision.action_proposal.vz = 0.0
            decision.confidence = 0.0
            decision.reason = "graph_critical_stale"
            # Note: Policy service should call freeze() when critical staleness detected

        return decision

    def _check_constraints(
        self,
        decision: PolicyDecision,
        robot_state: RobotState,
        mission: Mission,
    ) -> PolicyDecision:
        """Check mission constraints."""
        # Check geofence (if specified)
        if mission.constraints.geofence is not None:
            # Simplified: would need proper region checking
            # For Milestone A, assume valid
            pass

        # Check no-fly zones
        if mission.constraints.no_fly_zones:
            # Simplified: would need proper region checking
            # For Milestone A, assume valid
            pass

        return decision

    def _apply_rate_limiting(
        self,
        decision: PolicyDecision,
        robot_state: RobotState,
    ) -> PolicyDecision:
        """Apply rate limiting to prevent sudden changes."""
        if robot_state.previous_action is not None:
            prev_v, prev_omega = robot_state.previous_action[0], robot_state.previous_action[1]
            max_dv = 0.1  # m/s per step
            max_domega = 0.5  # rad/s per step

            dv = decision.action_proposal.v - prev_v
            domega = decision.action_proposal.omega - prev_omega

            if abs(dv) > max_dv:
                decision.action_proposal.v = prev_v + np.sign(dv) * max_dv
            if abs(domega) > max_domega:
                decision.action_proposal.omega = prev_omega + np.sign(domega) * max_domega

        return decision

