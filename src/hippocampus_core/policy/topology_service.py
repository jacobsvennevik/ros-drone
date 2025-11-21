"""Topology Service that wraps TopologicalGraph and provides snapshots."""
from __future__ import annotations

from typing import Optional
import numpy as np

from ..topology import TopologicalGraph
from ..controllers.place_cell_controller import PlaceCellController
from .data_structures import (
    GraphSnapshot,
    GraphSnapshotMetadata,
    NodeData,
    EdgeData,
)


class TopologyService:
    """Service that wraps TopologicalGraph and provides snapshots for policy."""

    def __init__(self, graph: Optional[TopologicalGraph] = None, frame_id: str = "map"):
        """Initialize topology service.

        Parameters
        ----------
        graph:
            Optional initial graph. If None, must be set via update_from_controller.
        frame_id:
            Coordinate frame ID for snapshots.
        """
        self._graph = graph
        self._frame_id = frame_id
        self._epoch_id = 0
        self._last_update_time = 0.0
        self._node_visit_counts: dict[int, int] = {}
        self._staleness_threshold = 5.0  # seconds

    def update_from_controller(self, controller: PlaceCellController) -> None:
        """Update the graph from a PlaceCellController.

        This should be called periodically (e.g., every 1-5 seconds)
        to refresh the graph snapshot.

        Parameters
        ----------
        controller:
            PlaceCellController instance to get graph from.
        """
        self._graph = controller.get_graph()
        self._last_update_time = controller.current_time
        self._epoch_id += 1

    def set_graph(self, graph: TopologicalGraph, current_time: float) -> None:
        """Set graph directly (for testing or manual updates).

        Parameters
        ----------
        graph:
            TopologicalGraph instance.
        current_time:
            Current simulation time.
        """
        self._graph = graph
        self._last_update_time = current_time
        self._epoch_id += 1

    def get_graph_snapshot(self, current_time: float) -> GraphSnapshot:
        """Get an immutable snapshot of the current graph state.

        Parameters
        ----------
        current_time:
            Current simulation time in seconds.

        Returns
        -------
        GraphSnapshot
            Snapshot with nodes, edges, and metadata.
        """
        if self._graph is None:
            # Return empty snapshot if no graph
            return GraphSnapshot(
                V=[],
                E=[],
                meta=GraphSnapshotMetadata(
                    epoch_id=self._epoch_id,
                    frame_id=self._frame_id,
                    stamp=current_time,
                    last_updated=self._last_update_time,
                    update_rate=0.0,
                    staleness_warning=True,
                    node_visit_counts={},
                ),
            )

        # Check staleness
        time_since_update = current_time - self._last_update_time
        staleness_warning = time_since_update > self._staleness_threshold

        # Build node list
        nodes = []
        positions = self._graph.positions
        degrees = self._graph.get_node_degrees()
        for i in range(self._graph.num_nodes()):
            pos = positions[i]
            nodes.append(
                NodeData(
                    node_id=i,
                    position=(float(pos[0]), float(pos[1])),
                    normal=None,
                    degree=int(degrees[i]),
                    tags=[],
                )
            )

        # Build edge list
        edges = []
        for u, v in self._graph.graph.edges():
            edge_data = self._graph.graph[u][v]
            distance = np.linalg.norm(positions[u] - positions[v])
            edges.append(
                EdgeData(
                    u=u,
                    v=v,
                    length=float(distance),
                    traversable=True,
                    integrator_value=float(edge_data.get("weight", 0.0)),
                    updated_at=self._last_update_time,
                )
            )

        # Compute update rate
        update_rate = 1.0 / max(time_since_update, 0.1) if time_since_update > 0 else 0.0

        return GraphSnapshot(
            V=nodes,
            E=edges,
            meta=GraphSnapshotMetadata(
                epoch_id=self._epoch_id,
                frame_id=self._frame_id,
                stamp=current_time,
                last_updated=self._last_update_time,
                update_rate=update_rate,
                staleness_warning=staleness_warning,
                node_visit_counts=self._node_visit_counts.copy(),
            ),
        )

    def mark_node_visited(self, node_id: int) -> None:
        """Mark a node as visited (for exploration tracking).

        Parameters
        ----------
        node_id:
            Node ID to mark as visited.
        """
        self._node_visit_counts[node_id] = self._node_visit_counts.get(node_id, 0) + 1

