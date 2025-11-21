"""SNN Policy Service for navigation and flight control."""

from .data_structures import (
    FeatureVector,
    RobotState,
    Mission,
    MissionGoal,
    GoalType,
    PolicyDecision,
    ActionProposal,
    SafeCommand,
    GraphSnapshot,
    GraphSnapshotMetadata,
    NodeData,
    EdgeData,
)

from .topology_service import TopologyService
from .feature_service import SpatialFeatureService
from .policy_service import SpikingPolicyService
from .safety import ActionArbitrationSafety
from .graph_navigation import GraphNavigationService, NavigationPath, WaypointTarget

# R-STDP components (biologically plausible, no PyTorch required)
from .rstdp_network import RSTDPPolicySNN, RSTDPConfig
from .reward_function import NavigationRewardFunction, RewardConfig

# SNN components (optional - require PyTorch/snnTorch)
try:
    from .spike_encoding import FeatureEncoder, EncodingConfig
    from .snn_network import PolicySNN, SNNConfig
    from .decision_decoding import DecisionDecoder, DecoderConfig
    from .temporal_context import TemporalContext
    
    __all__ = [
        "FeatureVector",
        "RobotState",
        "Mission",
        "MissionGoal",
        "GoalType",
        "PolicyDecision",
        "ActionProposal",
        "SafeCommand",
        "GraphSnapshot",
        "GraphSnapshotMetadata",
        "NodeData",
        "EdgeData",
        "TopologyService",
        "SpatialFeatureService",
        "SpikingPolicyService",
        "ActionArbitrationSafety",
        "GraphNavigationService",
        "NavigationPath",
        "WaypointTarget",
        "RSTDPPolicySNN",
        "RSTDPConfig",
        "NavigationRewardFunction",
        "RewardConfig",
        "FeatureEncoder",
        "EncodingConfig",
        "PolicySNN",
        "SNNConfig",
        "DecisionDecoder",
        "DecoderConfig",
        "TemporalContext",
    ]
except ImportError:
    # SNN components not available (PyTorch/snnTorch not installed)
    # R-STDP components are still available (no PyTorch required)
    __all__ = [
        "FeatureVector",
        "RobotState",
        "Mission",
        "MissionGoal",
        "GoalType",
        "PolicyDecision",
        "ActionProposal",
        "SafeCommand",
        "GraphSnapshot",
        "GraphSnapshotMetadata",
        "NodeData",
        "EdgeData",
        "TopologyService",
        "SpatialFeatureService",
        "SpikingPolicyService",
        "ActionArbitrationSafety",
        "GraphNavigationService",
        "NavigationPath",
        "WaypointTarget",
        "RSTDPPolicySNN",
        "RSTDPConfig",
        "NavigationRewardFunction",
        "RewardConfig",
    ]

