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

    def __post_init__(self) -> None:
        if self.max_edge_distance is None:
            self.max_edge_distance = 2.0 * self.sigma


class PlaceCellController(SNNController):
    """Controller that maintains hippocampal state using place-cell coactivity.

    The controller tracks place-cell firing rates, generates Poisson spikes,
    accumulates coactivity statistics, and builds a topological graph whenever
    requested. It currently returns a zero action vector; future controllers can
    override this behaviour to drive the agent.
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

        x, y = float(observation[0]), float(observation[1])
        rates = self.place_cells.get_rates(x, y)
        self._cell_rate_sums += rates
        self._mean_rate_sum += float(rates.mean())

        spikes = self.place_cells.sample_spikes(rates, dt)
        self._spike_counts += spikes.astype(float)

        self._time += dt
        self.coactivity.register_spikes(self._time, spikes)
        self._graph_dirty = True
        self._steps += 1

        return np.zeros(2, dtype=float)

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

    def get_coactivity_matrix(self) -> np.ndarray:
        return self.coactivity.get_coactivity_matrix()

    def get_graph(self) -> TopologicalGraph:
        if self._graph is None or self._graph_dirty:
            self._graph = TopologicalGraph(self.place_cell_positions)
            self._graph.build_from_coactivity(
                self.get_coactivity_matrix(),
                c_min=self.config.coactivity_threshold,
                max_distance=self.config.max_edge_distance,
            )
            self._graph_dirty = False
        return self._graph
