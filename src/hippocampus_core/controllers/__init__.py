"""Controller implementations for hippocampus-inspired navigation."""

from .base import SNNController
from .place_cell_controller import PlaceCellController, PlaceCellControllerConfig
from .bat_navigation_controller import (
    BatNavigationController,
    BatNavigationControllerConfig,
)
from .rstdp_controller import RSTDPController, RSTDPControllerConfig

# SnnTorchController is optional - may not be available if snntorch isn't installed
# or if there are version compatibility issues
try:
    from .snntorch_controller import SnnTorchController, SnnTorchControllerConfig
except (ImportError, AttributeError):
    SnnTorchController = None
    SnnTorchControllerConfig = None

__all__ = [
    "SNNController",
    "PlaceCellController",
    "PlaceCellControllerConfig",
    "BatNavigationController",
    "BatNavigationControllerConfig",
    "RSTDPController",
    "RSTDPControllerConfig",
    "SnnTorchController",
    "SnnTorchControllerConfig",
]

