"""Spiking Policy Service for navigation decisions."""
from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ..controllers.base import SNNController
from .data_structures import (
    FeatureVector,
    LocalContext,
    PolicyDecision,
    ActionProposal,
    RobotState,
    Mission,
)
from .feature_service import SpatialFeatureService
from .spike_encoding import FeatureEncoder, EncodingConfig
from .snn_network import PolicySNN, SNNConfig
from .decision_decoding import DecisionDecoder, DecoderConfig
from .temporal_context import TemporalContext
from .graph_navigation import GraphNavigationService, WaypointTarget
from .rstdp_network import RSTDPPolicySNN, RSTDPConfig
from .reward_function import NavigationRewardFunction, RewardConfig


class SpikingPolicyService(SNNController):
    """Spiking neural network policy that decides where to fly.

    This implements the SNNController interface so it can be used
    in the same way as PlaceCellController or SnnTorchController.

    Supports both heuristic (fallback) and SNN inference modes.
    """

    def __init__(
        self,
        feature_service: SpatialFeatureService,
        config: Optional[dict] = None,
        snn_model: Optional[PolicySNN] = None,
        navigation_service: Optional[GraphNavigationService] = None,
        rstdp_model: Optional[RSTDPPolicySNN] = None,
        reward_function: Optional[NavigationRewardFunction] = None,
    ):
        """Initialize policy service.

        Parameters
        ----------
        feature_service:
            SpatialFeatureService instance.
        config:
            Optional configuration dictionary.
        snn_model:
            Optional pre-trained SNN model (PyTorch-based, uses backprop).
            If None and rstdp_model is None, uses heuristic.
        navigation_service:
            Optional GraphNavigationService for hierarchical planning.
        rstdp_model:
            Optional R-STDP model (biologically plausible, no backprop).
            If provided, uses R-STDP learning instead of PyTorch SNN.
        reward_function:
            Optional reward function for R-STDP learning.
            Required if rstdp_model is provided.
        """
        self._feature_service = feature_service
        self._config = config or {}
        self._max_linear = self._config.get("max_linear", 0.3)
        self._max_angular = self._config.get("max_angular", 1.0)
        self._max_vertical = self._config.get("max_vertical", 0.2)  # For 3D
        self._snn_model = snn_model
        self._rstdp_model = rstdp_model
        self._use_snn = snn_model is not None
        self._use_rstdp = rstdp_model is not None
        self._navigation_service = navigation_service
        self._use_hierarchical = navigation_service is not None
        self._current_waypoint: Optional[WaypointTarget] = None

        # Reward function for R-STDP learning
        self._reward_function = reward_function
        if self._use_rstdp and self._reward_function is None:
            # Create default reward function
            self._reward_function = NavigationRewardFunction()

        # SNN components initialized lazily on first use (when feature_dim is known)
        self._encoder = None
        self._decoder = None
        self._temporal_context = None
        self._membrane = None
        self._device = torch.device(self._config.get("device", "cpu")) if torch is not None else None

        # Validate configuration
        if self._use_snn and self._use_rstdp:
            raise ValueError("Cannot use both PyTorch SNN and R-STDP model simultaneously")
        
        self._is_frozen = False

    def reset(self) -> None:
        """Reset policy state."""
        if self._temporal_context:
            self._temporal_context.reset()
        if self._encoder:
            self._encoder.reset()
        if self._snn_model and self._device is not None:
            if self._membrane is not None:
                self._membrane = self._snn_model.init_state(1, self._device)
            else:
                # Initialize if not yet initialized
                hidden_dim = self._snn_model.hidden_dim
                self._membrane = torch.zeros(1, hidden_dim, device=self._device)
        if self._rstdp_model:
            self._rstdp_model.reset()
        if self._reward_function:
            self._reward_function.reset()
        self._is_frozen = False

    def freeze(self) -> None:
        """Freeze policy state (e.g., during safety hold mode).
        
        This resets membrane potentials and internal state to prevent
        stale actions when unfreezing.
        """
        self._is_frozen = True
        # Reset membrane states to prevent stale actions
        if self._snn_model and self._device is not None and self._membrane is not None:
            self._membrane = self._snn_model.init_state(1, self._device)
        if self._temporal_context:
            self._temporal_context.reset()
        if self._rstdp_model:
            # R-STDP models maintain state - reset to prevent stale eligibility
            self._rstdp_model.reset()

    def unfreeze(self) -> None:
        """Unfreeze policy state after safety hold."""
        self._is_frozen = False

    def step(self, obs: np.ndarray, dt: float) -> np.ndarray:
        """Advance policy by one step.

        Parameters
        ----------
        obs:
            Observation vector [x, y, cos(heading), sin(heading), ...]
        dt:
            Time step duration.

        Returns
        -------
        np.ndarray
            Action vector [v, ω].
        """
        # Extract robot state from observation
        robot_state = self._parse_observation(obs)

        # Get mission (for now, use default goal)
        # In real implementation, mission would come from mission manager
        from .data_structures import Mission, MissionGoal, GoalType, PointGoal

        mission = Mission(
            goal=MissionGoal(
                type=GoalType.POINT,
                value=PointGoal(position=(0.9, 0.9)),
            )
        )

        # Build features
        features, local_context = self._feature_service.build_features(
            robot_state=robot_state,
            mission=mission,
        )

        # Make decision (pass mission for hierarchical planning)
        decision = self.decide(features, local_context, dt, mission=mission)

        # If using R-STDP, compute reward and update weights
        if self._use_rstdp and self._reward_function and mission:
            reward = self._reward_function.compute(
                robot_state=robot_state,
                action=decision,
                mission=mission,
                features=features,
                local_context=local_context,
                dt=dt,
            )
            self._rstdp_model.update_weights(reward)

        # Return action proposal as numpy array
        action = decision.action_proposal
        if action.vz is not None:
            # 3D: return [v, omega, vz]
            return np.array([action.v, action.omega, action.vz], dtype=float)
        else:
            # 2D: return [v, omega]
            return np.array([action.v, action.omega], dtype=float)

    def decide(
        self,
        features: FeatureVector,
        local_context: LocalContext,
        dt: float,
        mission: Optional[Mission] = None,
    ) -> PolicyDecision:
        """Make a policy decision from features.

        Uses hierarchical planning if navigation service available,
        otherwise uses reactive control (SNN or heuristic).

        Parameters
        ----------
        features:
            Feature vector.
        local_context:
            Local context.
        dt:
            Time step duration.
        mission:
            Optional mission (for hierarchical planning).

        Returns
        -------
        PolicyDecision
            Policy decision with action proposal.
        """
        # Hierarchical planning: use waypoint from GNS
        if self._use_hierarchical and mission is not None and local_context.graph_snapshot:
            # Update navigation service graph
            self._navigation_service.update_graph(local_context.graph_snapshot)
            
            # Get current pose from robot state (would come from local_context in real implementation)
            # For now, extract from features or use default
            current_pose = self._extract_pose_from_context(local_context)
            
            # Select waypoint
            waypoint = self._navigation_service.select_next_waypoint(current_pose, mission.goal)
            
            if waypoint is not None:
                # Modify features to bias toward waypoint
                features = self._add_waypoint_bias(features, waypoint)
                self._current_waypoint = waypoint
        
        # Make decision (R-STDP, SNN, or heuristic)
        if self._use_rstdp and self._rstdp_model is not None:
            # R-STDP: biologically plausible, no backprop
            try:
                decision = self._rstdp_decide(features, local_context, dt)
            except Exception as e:
                # Fallback to heuristic on error
                import warnings
                warnings.warn(f"R-STDP inference failed, using heuristic: {e}")
                decision = self._heuristic_decide(features, local_context)
        elif self._use_snn and self._snn_model is not None:
            # PyTorch SNN: uses backprop (not biologically plausible)
            try:
                decision = self._snn_decide(features, local_context, dt)
            except Exception as e:
                # Fallback to heuristic on error
                import warnings
                warnings.warn(f"SNN inference failed, using heuristic: {e}")
                decision = self._heuristic_decide(features, local_context)
        else:
            # Heuristic fallback
            decision = self._heuristic_decide(features, local_context)
        
        # Set waypoint in decision if using hierarchical planning
        if self._current_waypoint is not None:
            decision.next_waypoint = self._current_waypoint.node_id
        
        return decision
    
    def _extract_pose_from_context(self, local_context: LocalContext) -> Tuple[float, float, float]:
        """Extract pose from local context (simplified - would come from robot state)."""
        # In real implementation, this would come from robot_state
        # For now, return default
        return (0.5, 0.5, 0.0)
    
    def _add_waypoint_bias(
        self,
        features: FeatureVector,
        waypoint: WaypointTarget,
    ) -> FeatureVector:
        """Add waypoint information to features (bias toward waypoint).

        Parameters
        ----------
        features:
            Original features.
        waypoint:
            Current waypoint target.

        Returns
        -------
        FeatureVector
            Modified features with waypoint bias.
        """
        # Modify goal_ego to point toward waypoint instead of final goal
        # This biases the policy toward the waypoint
        max_range = self._feature_service.max_range
        waypoint_ego = [
            min(waypoint.distance / max_range, 1.0),  # Normalized distance
            np.cos(waypoint.bearing),  # cos(bearing)
            np.sin(waypoint.bearing),  # sin(bearing)
        ]
        
        # Add elevation if 3D
        if self._feature_service.is_3d and len(waypoint.position) > 2:
            # Would compute elevation here
            waypoint_ego.extend([0.0, 1.0])  # Default: no elevation
        
        # Create modified feature vector
        return FeatureVector(
            goal_ego=waypoint_ego,  # Override with waypoint
            neighbors_k=features.neighbors_k,
            topo_ctx=features.topo_ctx,
            safety=features.safety,
            dynamics=features.dynamics,
        )
    
    def _snn_decide(
        self,
        features: FeatureVector,
        local_context: LocalContext,
        dt: float,
    ) -> PolicyDecision:
        """SNN-based decision making.

        Parameters
        ----------
        features:
            Feature vector.
        local_context:
            Local context.
        dt:
            Time step duration.

        Returns
        -------
        PolicyDecision
            Policy decision from SNN.
        """
        if torch is None:
            raise RuntimeError("PyTorch required for SNN inference")
        
        if self._snn_model is None:
            raise RuntimeError("SNN model not available")
        
        # Initialize encoder/decoder on first use (when we know feature_dim)
        if self._encoder is None:
            feature_dim = features.dim
            encoding_config = EncodingConfig(
                encoding_scheme=self._config.get("encoding_scheme", "rate"),
                num_steps=self._config.get("num_steps", 1),
                gain=self._config.get("encoding_gain", 1.0),
            )
            self._encoder = FeatureEncoder(encoding_config)
            self._decoder = DecisionDecoder(
                max_linear=self._max_linear,
                max_angular=self._max_angular,
            )
            self._temporal_context = TemporalContext(
                history_length=self._config.get("history_length", 10),
            )
            # Verify SNN model matches feature dimension
            if self._snn_model.feature_dim != feature_dim:
                raise ValueError(
                    f"SNN model feature_dim ({self._snn_model.feature_dim}) "
                    f"does not match actual features ({feature_dim})"
                )

        # Encode features to spikes
        spike_train = self._encoder.encode(features, device=self._device)
        
        # SNN inference (single step)
        # Remove time dimension for single-step: (num_steps, 1, dim) -> (1, dim)
        spike_input = spike_train.squeeze(1) if spike_train.dim() == 3 else spike_train
        if spike_input.dim() == 3:
            spike_input = spike_input[-1]  # Use last timestep
        
        # Forward pass
        with torch.no_grad():
            action_tensor, self._membrane = self._snn_model.forward_step(
                spike_input,
                self._membrane,
            )
            self._membrane = self._membrane.detach()  # Detach for efficiency
        
        # Decode to decision
        # Get mission from context (simplified - would come from mission manager)
        from .data_structures import Mission, MissionGoal, GoalType, PointGoal
        mission = Mission(
            goal=MissionGoal(
                type=GoalType.POINT,
                value=PointGoal(position=(0.9, 0.9)),
            )
        )
        
        decision = self._decoder.decode(
            action_tensor,
            features,
            local_context,
            mission,
        )
        
        # Update temporal context
        if self._temporal_context:
            self._temporal_context.update(features, decision, self._membrane)
        
        return decision

    def _rstdp_decide(
        self,
        features: FeatureVector,
        local_context: LocalContext,
        dt: float,
    ) -> PolicyDecision:
        """R-STDP-based decision making (biologically plausible, no backprop).

        Parameters
        ----------
        features:
            Feature vector.
        local_context:
            Local context.
        dt:
            Time step duration.

        Returns
        -------
        PolicyDecision
            Policy decision from R-STDP network.
        """
        if self._rstdp_model is None:
            raise RuntimeError("R-STDP model not available")

        # Forward pass through R-STDP network
        action_array = self._rstdp_model.forward(features)

        # Convert to PolicyDecision
        # R-STDP outputs are in [-1, 1], scale to physical units
        v = action_array[0] * self._max_linear
        omega = action_array[1] * self._max_angular
        vz = action_array[2] * self._max_vertical if len(action_array) > 2 else None

        action_proposal = ActionProposal(
            v=v,
            omega=omega,
            vz=vz,
        )

        # Get mission from context (simplified - would come from mission manager)
        from .data_structures import Mission, MissionGoal, GoalType, PointGoal
        mission = Mission(
            goal=MissionGoal(
                type=GoalType.POINT,
                value=PointGoal(position=(0.9, 0.9)),
            )
        )

        decision = PolicyDecision(
            action_proposal=action_proposal,
            confidence=0.8,  # R-STDP doesn't provide explicit confidence
            reason="rstdp",
            next_waypoint=-1,
        )

        return decision
    
    def _heuristic_decide(
        self,
        features: FeatureVector,
        local_context: LocalContext,
    ) -> PolicyDecision:
        """Heuristic decision maker (Milestone A stub).

        Simple strategy: move toward goal, avoid obstacles.

        Parameters
        ----------
        features:
            Feature vector.
        local_context:
            Local context.

        Returns
        -------
        PolicyDecision
            Policy decision.
        """
        # Extract goal features
        goal_ego = features.goal_ego
        dg = goal_ego[0]  # Distance to goal (normalized)
        theta_g_cos = goal_ego[1]  # cos(bearing to goal)
        theta_g_sin = goal_ego[2]  # sin(bearing to goal)
        theta_g = np.arctan2(theta_g_sin, theta_g_cos)  # Bearing to goal

        # Basic velocity command: proportional to distance and bearing
        # Closer to goal → slower, far from goal → faster
        v = min(dg * self._max_linear * 0.5, self._max_linear * 0.8)

        # Angular velocity: proportional to bearing error
        # Larger error → faster rotation
        omega = theta_g * self._max_angular * 0.5

        # Consider safety features (obstacles)
        safety = features.safety
        if len(safety) >= 4:
            front_safety = safety[0]  # Front obstacle distance (normalized)
            # If obstacle ahead, reduce speed and turn
            if front_safety < 0.3:  # Close obstacle
                v *= 0.3  # Slow down
                # Turn away from obstacle
                if len(safety) >= 3:
                    left_safety = safety[1]
                    right_safety = safety[2]
                    if left_safety > right_safety:
                        omega += 0.5  # Turn left
                    else:
                        omega -= 0.5  # Turn right

        # Clamp actions
        v = np.clip(v, -self._max_linear, self._max_linear)
        omega = np.clip(omega, -self._max_angular, self._max_angular)

        # Compute confidence (lower when far from goal or near obstacles)
        front_safety = safety[0] if len(safety) > 0 else 1.0
        confidence = min(dg, 1.0) * front_safety

        # For 3D, add vertical velocity (simple: maintain altitude)
        vz = None
        if self._feature_service.is_3d:
            vz = 0.0  # Maintain current altitude (heuristic)

        return PolicyDecision(
            next_waypoint=None,  # No explicit waypoint in reactive mode
            action_proposal=ActionProposal(v=v, omega=omega, vz=vz),
            confidence=float(confidence),
            reason="heuristic",
        )

    def _parse_observation(self, obs: np.ndarray) -> RobotState:
        """Parse observation array to RobotState.

        Parameters
        ----------
        obs:
            Observation array [x, y, cos(heading), sin(heading), ...]

        Returns
        -------
        RobotState
            Parsed robot state.
        """
        from .data_structures import RobotState

        if obs.shape[0] < 2:
            raise ValueError("Observation must include at least (x, y) position")

        x, y = float(obs[0]), float(obs[1])

        # Extract heading
        if obs.shape[0] >= 4:
            cos_h = float(obs[2])
            sin_h = float(obs[3])
            yaw = np.arctan2(sin_h, cos_h)
        else:
            yaw = 0.0

        return RobotState(
            pose=(x, y, yaw),
            time=0.0,  # Would come from system time in real implementation
        )

