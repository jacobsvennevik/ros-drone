"""Preset configurations matching published research papers.

This module provides preset configurations that match parameters used in
published research papers, enabling direct replication of results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .controllers.place_cell_controller import PlaceCellControllerConfig


@dataclass
class PaperPreset:
    """Preset matching Hoffman et al. (2016) parameters.

    Paper: arXiv:1601.04253v1 [q-bio.NC] - "Topological mapping of space in bat hippocampus"

    Parameters match the paper's experimental setup:
    - 343 place cells arranged in 7×7×7 grid (3D)
    - Place field size: 95 cm (σ = 95/2√2 ≈ 33.6 cm)
    - Mean speed: 66 cm/s
    - Duration: 120 minutes
    - Integration window: 8 minutes (480 seconds)

    Note: For 2D simulations, we use 2D equivalents of these parameters.
    """

    # Place cell parameters
    num_place_cells: int = 343  # 7×7×7 grid in 3D, or 343 cells in 2D
    sigma: float = 0.336  # 95 cm / 2√2, normalized to 1.0 m arena
    max_rate: float = 20.0  # Hz (typical firing rate)

    # Coactivity parameters
    coactivity_window: float = 0.25  # 250 ms (paper's w)
    coactivity_threshold: float = 5.0  # Minimum coactivity for edge formation
    max_edge_distance: float = 0.4  # Place field overlap distance

    # Integration window
    integration_window: float = 480.0  # 8 minutes (paper's ϖ)

    # Agent parameters
    agent_base_speed: float = 0.66  # 66 cm/s normalized to 1.0 m arena
    agent_max_speed: float = 1.32  # 2× base speed (132 cm/s)
    agent_velocity_noise: float = 0.2  # Velocity noise parameter

    # Simulation parameters
    duration: float = 7200.0  # 120 minutes (7200 seconds)

    # Arena parameters (for 2D)
    arena_width: float = 1.0  # 1 meter (normalized)
    arena_height: float = 1.0  # 1 meter (normalized)

    def __post_init__(self) -> None:
        """Validate preset parameters."""
        if self.num_place_cells <= 0:
            raise ValueError("num_place_cells must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.coactivity_window <= 0:
            raise ValueError("coactivity_window must be positive")
        if self.integration_window < 0:
            raise ValueError("integration_window must be non-negative")
        if self.agent_base_speed <= 0:
            raise ValueError("agent_base_speed must be positive")
        if self.agent_max_speed <= self.agent_base_speed:
            raise ValueError("agent_max_speed must be greater than agent_base_speed")
        if self.duration <= 0:
            raise ValueError("duration must be positive")


def get_paper_preset() -> Tuple[PlaceCellControllerConfig, Dict[str, float]]:
    """Return configuration matching Hoffman et al. (2016) paper parameters.

    Returns
    -------
    Tuple[PlaceCellControllerConfig, Dict[str, float]]
        Controller configuration and agent parameters dictionary.
    """
    preset = PaperPreset()

    # Create controller config
    config = PlaceCellControllerConfig(
        num_place_cells=preset.num_place_cells,
        sigma=preset.sigma,
        max_rate=preset.max_rate,
        coactivity_window=preset.coactivity_window,
        coactivity_threshold=preset.coactivity_threshold,
        max_edge_distance=preset.max_edge_distance,
        integration_window=preset.integration_window,
    )

    # Agent parameters
    agent_params = {
        "base_speed": preset.agent_base_speed,
        "max_speed": preset.agent_max_speed,
        "velocity_noise": preset.agent_velocity_noise,
    }

    return config, agent_params


def get_paper_preset_2d() -> Tuple[PlaceCellControllerConfig, Dict[str, float]]:
    """Return 2D-optimized version of paper preset.

    For 2D simulations, we can use fewer place cells while maintaining
    similar coverage density. This uses ~100 cells for faster computation
    while preserving the key parameter ratios.

    Returns
    -------
    Tuple[PlaceCellControllerConfig, Dict[str, float]]
        Controller configuration and agent parameters dictionary.
    """
    preset = PaperPreset()

    # Scale down for 2D (maintain similar density)
    # 343 cells in 3D → ~100 cells in 2D for similar coverage
    num_cells_2d = 100

    config = PlaceCellControllerConfig(
        num_place_cells=num_cells_2d,
        sigma=preset.sigma,
        max_rate=preset.max_rate,
        coactivity_window=preset.coactivity_window,
        coactivity_threshold=preset.coactivity_threshold,
        max_edge_distance=preset.max_edge_distance,
        integration_window=preset.integration_window,
    )

    agent_params = {
        "base_speed": preset.agent_base_speed,
        "max_speed": preset.agent_max_speed,
        "velocity_noise": preset.agent_velocity_noise,
    }

    return config, agent_params


def get_paper_preset_quick() -> Tuple[PlaceCellControllerConfig, Dict[str, float]]:
    """Return quick-test version of paper preset (shorter duration).

    Uses paper parameters but with reduced duration for faster testing.
    Useful for validation without running full 120-minute simulation.

    Returns
    -------
    Tuple[PlaceCellControllerConfig, Dict[str, float]]
        Controller configuration and agent parameters dictionary.
    """
    preset = PaperPreset()

    config = PlaceCellControllerConfig(
        num_place_cells=100,  # Reduced for speed
        sigma=preset.sigma,
        max_rate=preset.max_rate,
        coactivity_window=preset.coactivity_window,
        coactivity_threshold=preset.coactivity_threshold,
        max_edge_distance=preset.max_edge_distance,
        integration_window=240.0,  # 4 minutes (reduced from 8)
    )

    agent_params = {
        "base_speed": preset.agent_base_speed,
        "max_speed": preset.agent_max_speed,
        "velocity_noise": preset.agent_velocity_noise,
    }

    return config, agent_params

