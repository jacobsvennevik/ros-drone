"""Matplotlib plotting utilities for the hippocampal/topological mapping simulator."""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .env import Environment
from .topology import TopologicalGraph


def _ensure_axes(ax: Optional[plt.Axes]) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots()
    return ax


def plot_trajectory(env: Environment, trajectory: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot the agent trajectory inside the environment bounds."""

    ax = _ensure_axes(ax)
    bounds = env.bounds
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:blue", linewidth=1.0, alpha=0.8)
    ax.set_xlim(bounds.min_x, bounds.max_x)
    ax.set_ylim(bounds.min_y, bounds.max_y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Agent trajectory")
    ax.set_aspect("equal")
    return ax


def plot_place_cells(
    env: Environment,
    positions: np.ndarray,
    sigma: float,
    ax: Optional[plt.Axes] = None,
    field_scale: float = 1.0,
) -> plt.Axes:
    """Plot place-cell centers and approximate receptive-field extents."""

    ax = _ensure_axes(ax)
    bounds = env.bounds
    ax.scatter(positions[:, 0], positions[:, 1], s=20, color="tab:orange", alpha=0.9, label="Centers")

    if sigma > 0:
        radius = field_scale * sigma
        for x, y in positions:
            circle = plt.Circle((x, y), radius=radius, color="tab:orange", alpha=0.1)
            ax.add_patch(circle)

    ax.set_xlim(bounds.min_x, bounds.max_x)
    ax.set_ylim(bounds.min_y, bounds.max_y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Place-cell centers")
    ax.set_aspect("equal")
    return ax


def plot_graph(
    env: Environment,
    positions: np.ndarray,
    graph: TopologicalGraph,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot the topological graph over place-cell centers."""

    ax = _ensure_axes(ax)
    bounds = env.bounds
    ax.scatter(positions[:, 0], positions[:, 1], s=20, color="tab:green", alpha=0.9)

    for i, j in graph.graph.edges():
        x_values = (positions[i, 0], positions[j, 0])
        y_values = (positions[i, 1], positions[j, 1])
        ax.plot(x_values, y_values, color="tab:green", linewidth=0.5, alpha=0.6)

    ax.set_xlim(bounds.min_x, bounds.max_x)
    ax.set_ylim(bounds.min_y, bounds.max_y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Topological graph")
    ax.set_aspect("equal")
    return ax


def plot_summary(
    env: Environment,
    trajectory: np.ndarray,
    positions: np.ndarray,
    sigma: float,
    graph: TopologicalGraph,
) -> Tuple[plt.Figure, Iterable[plt.Axes]]:
    """Create a multi-panel summary figure for the current simulation state."""

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_trajectory(env, trajectory, ax=axes[0])
    plot_place_cells(env, positions, sigma, ax=axes[1])
    plot_graph(env, positions, graph, ax=axes[2])
    fig.suptitle("Simulation summary")
    fig.tight_layout()
    return fig, axes
