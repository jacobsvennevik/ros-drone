"""Tests for Graph Navigation Service."""
from __future__ import annotations

import pytest
import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from hippocampus_core.policy import (
    GraphNavigationService,
    GraphSnapshot,
    GraphSnapshotMetadata,
    NodeData,
    EdgeData,
    MissionGoal,
    GoalType,
    PointGoal,
    NodeGoal,
)


@pytest.fixture
def simple_graph_snapshot():
    """Create a simple graph snapshot for testing."""
    nodes = [
        NodeData(node_id=0, position=(0.0, 0.0)),
        NodeData(node_id=1, position=(1.0, 0.0)),
        NodeData(node_id=2, position=(1.0, 1.0)),
        NodeData(node_id=3, position=(0.0, 1.0)),
    ]
    
    edges = [
        EdgeData(u=0, v=1, length=1.0, traversable=True),
        EdgeData(u=1, v=2, length=1.0, traversable=True),
        EdgeData(u=2, v=3, length=1.0, traversable=True),
        EdgeData(u=3, v=0, length=1.0, traversable=True),
        EdgeData(u=0, v=2, length=np.sqrt(2.0), traversable=True),  # Diagonal
    ]
    
    return GraphSnapshot(
        V=nodes,
        E=edges,
        meta=GraphSnapshotMetadata(
            epoch_id=0,
            frame_id="map",
            stamp=0.0,
            last_updated=0.0,
            update_rate=1.0,
            staleness_warning=False,
        ),
    )


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
def test_navigation_service_initialization():
    """Test navigation service initialization."""
    gns = GraphNavigationService(algorithm="dijkstra")
    assert gns.algorithm == "dijkstra"
    assert gns.graph_snapshot is None


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
def test_update_graph(simple_graph_snapshot):
    """Test graph update."""
    gns = GraphNavigationService()
    gns.update_graph(simple_graph_snapshot)
    assert gns.graph_snapshot is not None
    assert len(gns.graph_snapshot.V) == 4


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
def test_find_path_dijkstra(simple_graph_snapshot):
    """Test Dijkstra path finding."""
    gns = GraphNavigationService(algorithm="dijkstra")
    gns.update_graph(simple_graph_snapshot)
    
    # Path from 0 to 2
    path = gns.find_path(0, 2)
    assert path is not None
    assert path.is_complete
    assert path.nodes[0] == 0
    assert path.nodes[-1] == 2
    assert len(path.nodes) >= 2


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
def test_find_path_astar(simple_graph_snapshot):
    """Test A* path finding."""
    gns = GraphNavigationService(algorithm="astar")
    gns.update_graph(simple_graph_snapshot)
    
    path = gns.find_path(0, 2)
    assert path is not None
    assert path.is_complete


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
def test_find_path_greedy(simple_graph_snapshot):
    """Test greedy path finding."""
    gns = GraphNavigationService(algorithm="greedy")
    gns.update_graph(simple_graph_snapshot)
    
    path = gns.find_path(0, 2)
    assert path is not None
    assert path.is_complete


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
def test_same_node_path(simple_graph_snapshot):
    """Test path from node to itself."""
    gns = GraphNavigationService()
    gns.update_graph(simple_graph_snapshot)
    
    path = gns.find_path(0, 0)
    assert path is not None
    assert path.nodes == [0]
    assert path.total_length == 0.0
    assert path.is_complete


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
def test_unreachable_path(simple_graph_snapshot):
    """Test path to unreachable node."""
    gns = GraphNavigationService()
    gns.update_graph(simple_graph_snapshot)
    
    # Create disconnected node
    disconnected_snapshot = GraphSnapshot(
        V=simple_graph_snapshot.V + [NodeData(node_id=99, position=(10.0, 10.0))],
        E=simple_graph_snapshot.E,
        meta=simple_graph_snapshot.meta,
    )
    gns.update_graph(disconnected_snapshot)
    
    path = gns.find_path(0, 99)
    assert path is None


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
def test_select_next_waypoint(simple_graph_snapshot):
    """Test waypoint selection."""
    gns = GraphNavigationService()
    gns.update_graph(simple_graph_snapshot)
    
    # Point goal
    goal = MissionGoal(
        type=GoalType.POINT,
        value=PointGoal(position=(1.0, 1.0)),
    )
    
    current_pose = (0.0, 0.0, 0.0)  # At node 0
    waypoint = gns.select_next_waypoint(current_pose, goal)
    
    assert waypoint is not None
    assert waypoint.node_id in [1, 2]  # Next step toward goal
    assert waypoint.distance > 0


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
def test_node_goal(simple_graph_snapshot):
    """Test node goal resolution."""
    gns = GraphNavigationService()
    gns.update_graph(simple_graph_snapshot)
    
    goal = MissionGoal(
        type=GoalType.NODE,
        value=NodeGoal(node_id=2),
    )
    
    current_pose = (0.0, 0.0, 0.0)
    waypoint = gns.select_next_waypoint(current_pose, goal)
    
    assert waypoint is not None
    assert waypoint.node_id in [1, 2]


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
def test_is_reachable(simple_graph_snapshot):
    """Test reachability check."""
    gns = GraphNavigationService()
    gns.update_graph(simple_graph_snapshot)
    
    assert gns.is_reachable(0, 2)
    assert gns.is_reachable(0, 0)  # Same node
    assert not gns.is_reachable(0, 99)  # Non-existent node


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
def test_path_caching(simple_graph_snapshot):
    """Test path caching."""
    gns = GraphNavigationService()
    gns.update_graph(simple_graph_snapshot)
    
    # First call
    path1 = gns.find_path(0, 2)
    assert path1 is not None
    
    # Second call (should use cache)
    path2 = gns.find_path(0, 2)
    assert path2 is not None
    assert path2.nodes == path1.nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

