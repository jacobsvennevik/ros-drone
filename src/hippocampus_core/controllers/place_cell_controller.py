"""Controller that wraps place-cell, coactivity, and topology logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..coactivity import CoactivityTracker
from ..env import Environment
from ..place_cells import PlaceCellPopulation
from ..topology import TopologicalGraph
from .base import SNNController


@dataclass
class PlaceCellControllerConfig:
    """Configuration parameters for :class:`PlaceCellController`."""

    num_place_cells: int = 120
    sigma: float = 0.1
    max_rate: float = 15.0
    coactivity_window: float = 0.2
    coactivity_threshold: float = 5.0
    max_edge_distance: Optional[float] = None
    integration_window: Optional[float] = None
    place_cell_positions: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.max_edge_distance is None:
            self.max_edge_distance = 2.0 * self.sigma
        if self.integration_window is not None and self.integration_window < 0:
            raise ValueError("integration_window must be non-negative if provided")
        if self.place_cell_positions is not None:
            centers = np.asarray(self.place_cell_positions, dtype=float)
            if centers.shape != (self.num_place_cells, 2):
                raise ValueError(
                    "place_cell_positions must have shape "
                    f"({self.num_place_cells}, 2); got {centers.shape}"
                )
            self.place_cell_positions = centers


class PlaceCellController(SNNController):
    """Controller that maintains hippocampal state using place-cell coactivity.

    The controller tracks place-cell firing rates, generates Poisson spikes,
    accumulates coactivity statistics, and builds a topological graph whenever
    requested. It currently returns a zero action vector; future controllers can
    override this behaviour to drive the agent.

    Examples
    --------
    >>> from hippocampus_core.controllers.place_cell_controller import (
    ...     PlaceCellController,
    ...     PlaceCellControllerConfig,
    ... )
    >>> from hippocampus_core.env import Environment
    >>> import numpy as np
    >>> 
    >>> # Create controller with integration window
    >>> env = Environment(width=1.0, height=1.0)
    >>> config = PlaceCellControllerConfig(
    ...     num_place_cells=50,
    ...     coactivity_window=0.2,      # w: coincidence window
    ...     coactivity_threshold=5.0,
    ...     integration_window=60.0,    # ϖ: integration window (60 seconds)
    ... )
    >>> controller = PlaceCellController(environment=env, config=config)
    >>> 
    >>> # Run simulation
    >>> for step in range(100):
    ...     position = np.array([0.5, 0.5])  # Example position
    ...     action = controller.step(position, dt=0.05)
    >>> 
    >>> # Get learned graph
    >>> graph = controller.get_graph()
    >>> print(f"Nodes: {graph.num_nodes()}, Edges: {graph.num_edges()}")
    >>> 
    >>> # Compute Betti numbers (if ripser/gudhi available)
    >>> try:
    ...     betti = graph.compute_betti_numbers()
    ...     print(f"Betti numbers: b_0={betti[0]}, b_1={betti[1]}")
    ... except ImportError:
    ...     print("Install ripser or gudhi for Betti number computation")
    """

    def __init__(
        self,
        environment: Environment,
        config: Optional[PlaceCellControllerConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.environment = environment
        self.config = config or PlaceCellControllerConfig()
        self.rng = rng or np.random.default_rng()

        self.place_cells = PlaceCellPopulation(
            environment=self.environment,
            num_cells=self.config.num_place_cells,
            sigma=self.config.sigma,
            max_rate=self.config.max_rate,
            rng=self.rng,
            centers=self.config.place_cell_positions,
        )
        self.coactivity = CoactivityTracker(
            num_cells=self.config.num_place_cells,
            window=self.config.coactivity_window,
        )
        self._graph: Optional[TopologicalGraph] = None
        self._graph_dirty = True

        self._time = 0.0
        self._steps = 0
        self._cell_rate_sums = np.zeros(self.config.num_place_cells, dtype=float)
        self._mean_rate_sum = 0.0
        self._spike_counts = np.zeros(self.config.num_place_cells, dtype=float)

    def reset(self) -> None:
        self._time = 0.0
        self._steps = 0
        self._cell_rate_sums.fill(0.0)
        self._mean_rate_sum = 0.0
        self._spike_counts.fill(0.0)
        self.coactivity.reset()
        self._graph_dirty = True

    def step(self, obs: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0:
            raise ValueError("dt must be positive")
        observation = np.asarray(obs, dtype=float)
        if observation.ndim != 1 or observation.shape[0] < 2:
            raise ValueError("Observation must include at least (x, y) position")

        rates = self._compute_rates(observation, dt)
        self._cell_rate_sums += rates
        self._mean_rate_sum += float(rates.mean())

        spikes = self._sample_spikes(rates, dt)
        self._spike_counts += spikes.astype(float)

        self._time += dt
        # Pass threshold to track when pairs first exceed it in real-time
        threshold = (
            self.config.coactivity_threshold
            if self.config.integration_window is not None
            else None
        )
        self.coactivity.register_spikes(self._time, spikes, threshold=threshold)
        self._graph_dirty = True
        self._steps += 1

        return np.zeros(2, dtype=float)

    def _compute_rates(self, observation: np.ndarray, dt: float) -> np.ndarray:
        """Compute place-cell firing rates for the provided observation."""
        x, y = float(observation[0]), float(observation[1])
        return self.place_cells.get_rates(x, y)

    def _sample_spikes(self, rates: np.ndarray, dt: float) -> np.ndarray:
        """Sample spikes for the provided firing rates."""
        return self.place_cells.sample_spikes(rates, dt)

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def place_cell_positions(self) -> np.ndarray:
        return self.place_cells.get_positions()

    @property
    def spike_counts(self) -> np.ndarray:
        return self._spike_counts.copy()

    @property
    def average_rate_per_cell(self) -> np.ndarray:
        if self._steps == 0:
            return np.zeros_like(self._cell_rate_sums)
        return self._cell_rate_sums / self._steps

    @property
    def overall_mean_rate(self) -> float:
        if self._steps == 0:
            return 0.0
        return self._mean_rate_sum / self._steps

    @property
    def current_time(self) -> float:
        """Return the current simulation time in seconds."""
        return self._time

    def get_coactivity_matrix(self) -> np.ndarray:
        return self.coactivity.get_coactivity_matrix()

    def get_graph(self) -> TopologicalGraph:
        if self._graph is None or self._graph_dirty:
            self._graph = TopologicalGraph(self.place_cell_positions)
            
            # Get integration times if integration window is enabled
            integration_times = None
            if self.config.integration_window is not None:
                integration_times = self.coactivity.check_threshold_exceeded(
                    threshold=self.config.coactivity_threshold,
                    current_time=self._time,
                )
            
            self._graph.build_from_coactivity(
                self.get_coactivity_matrix(),
                c_min=self.config.coactivity_threshold,
                max_distance=self.config.max_edge_distance,
                integration_window=self.config.integration_window,
                current_time=self._time if self.config.integration_window is not None else None,
                integration_times=integration_times,
                environment=self.environment,  # Pass environment for obstacle-aware filtering
            )
            self._graph_dirty = False
        return self._graph
