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
        ax.plot(x_values, y_values, color="tab:green", linewidth=1.0, alpha=0.8)

    ax.set_xlim(bounds.min_x, bounds.max_x)
    ax.set_ylim(bounds.min_y, bounds.max_y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Topological graph")
    ax.set_aspect("equal")
    return ax


def plot_edge_length_histogram(
    graph: TopologicalGraph,
    max_distance: float,
    ax: Optional[plt.Axes] = None,
    bins: int = 30,
) -> plt.Axes:
    """Plot histogram of edge lengths with max_distance threshold marked.

    Parameters
    ----------
    graph:
        TopologicalGraph to extract edge lengths from.
    max_distance:
        Maximum allowed edge distance (shown as vertical line).
    ax:
        Optional axes to plot on.
    bins:
        Number of histogram bins.

    Returns
    -------
    plt.Axes
        The axes with the histogram plotted.
    """

    ax = _ensure_axes(ax)
    lengths = graph.get_edge_lengths()

    if len(lengths) > 0:
        ax.hist(lengths, bins=bins, alpha=0.7, color="tab:green", edgecolor="black")
        ax.axvline(max_distance, color="red", linestyle="--", linewidth=2, label=f"Max distance ({max_distance:.3f})")
        ax.set_xlabel("Edge length")
        ax.set_ylabel("Frequency")
        ax.set_title("Edge length distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No edges to plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Edge length")
        ax.set_ylabel("Frequency")
        ax.set_title("Edge length distribution")

    return ax


def plot_degree_distribution(
    graph: TopologicalGraph,
    ax: Optional[plt.Axes] = None,
    bins: int | None = None,
) -> plt.Axes:
    """Plot histogram of node degrees.

    Parameters
    ----------
    graph:
        TopologicalGraph to extract degrees from.
    ax:
        Optional axes to plot on.
    bins:
        Number of histogram bins. If None, uses automatic binning.

    Returns
    -------
    plt.Axes
        The axes with the histogram plotted.
    """

    ax = _ensure_axes(ax)
    degrees = graph.get_node_degrees()
    stats = graph.get_degree_statistics()

    if bins is None:
        # Auto-determine bins based on degree range
        max_degree = int(stats["max"])
        bins = min(30, max(5, max_degree + 1))

    ax.hist(degrees, bins=bins, alpha=0.7, color="tab:blue", edgecolor="black")
    ax.axvline(stats["mean"], color="red", linestyle="--", linewidth=2, label=f"Mean ({stats['mean']:.1f})")
    ax.axvline(stats["median"], color="orange", linestyle="--", linewidth=2, label=f"Median ({stats['median']:.1f})")
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Degree distribution (mean={stats['mean']:.1f}, median={stats['median']:.1f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_summary(
    env: Environment,
    trajectory: np.ndarray,
    positions: np.ndarray,
    sigma: float,
    graph: TopologicalGraph,
    max_edge_distance: float | None = None,
    show_diagnostics: bool = False,
) -> Tuple[plt.Figure, Iterable[plt.Axes]]:
    """Create a multi-panel summary figure for the current simulation state.

    Parameters
    ----------
    env:
        Environment bounds.
    trajectory:
        Agent trajectory array.
    positions:
        Place-cell center positions.
    sigma:
        Place-field size parameter.
    graph:
        Topological graph to visualize.
    max_edge_distance:
        Optional maximum edge distance for diagnostic plots.
    show_diagnostics:
        If True, includes diagnostic plots (edge length histogram, degree distribution).

    Returns
    -------
    Tuple[plt.Figure, Iterable[plt.Axes]]
        Figure and axes objects.
    """

    if show_diagnostics and max_edge_distance is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        plot_trajectory(env, trajectory, ax=axes[0, 0])
        plot_place_cells(env, positions, sigma, ax=axes[0, 1])
        plot_graph(env, positions, graph, ax=axes[0, 2])
        plot_edge_length_histogram(graph, max_edge_distance, ax=axes[1, 0])
        plot_degree_distribution(graph, ax=axes[1, 1])
        # Leave last subplot empty or add trajectory heatmap later
        axes[1, 2].axis("off")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        plot_trajectory(env, trajectory, ax=axes[0])
        plot_place_cells(env, positions, sigma, ax=axes[1])
        plot_graph(env, positions, graph, ax=axes[2])

    fig.suptitle("Simulation summary")
    fig.tight_layout()
    return fig, axes
