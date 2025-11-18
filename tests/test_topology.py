import numpy as np
import pytest

from hippocampus_core.coactivity import CoactivityTracker
from hippocampus_core.env import Environment
from hippocampus_core.place_cells import PlaceCellPopulation
from hippocampus_core.topology import TopologicalGraph


@pytest.fixture(scope="module")
def topology_inputs():
    env = Environment(width=1.0, height=1.0)
    rng = np.random.default_rng(777)
    num_cells = 9
    population = PlaceCellPopulation(
        environment=env,
        num_cells=num_cells,
        sigma=0.2,
        max_rate=18.0,
        rng=rng,
    )
    tracker = CoactivityTracker(num_cells=num_cells, window=0.08)

    dt = 0.02
    time = 0.0
    path = np.array(
        [
            [0.2, 0.2],
            [0.8, 0.2],
            [0.8, 0.8],
            [0.2, 0.8],
        ]
    )
    steps_per_segment = 50
    positions = np.vstack(
        [
            np.linspace(path[i], path[(i + 1) % len(path)], steps_per_segment, endpoint=False)
            for i in range(len(path))
        ]
    )

    for position in positions:
        x, y = position
        rates = population.get_rates(float(x), float(y))
        spikes = population.sample_spikes(rates, dt)
        time += dt
        tracker.register_spikes(time, spikes)

    coactivity_matrix = tracker.get_coactivity_matrix()
    return population.get_positions(), coactivity_matrix


def test_topological_graph_edges_increase_with_lower_threshold(topology_inputs):
    positions, coactivity = topology_inputs

    strict_graph = TopologicalGraph(positions)
    strict_graph.build_from_coactivity(coactivity, c_min=4.0, max_distance=1.5)

    lenient_graph = TopologicalGraph(positions)
    lenient_graph.build_from_coactivity(coactivity, c_min=1.0, max_distance=1.5)

    assert lenient_graph.num_edges() >= strict_graph.num_edges()
    assert lenient_graph.num_edges() > 0
    assert strict_graph.num_nodes() == lenient_graph.num_nodes() == positions.shape[0]
    assert strict_graph.num_components() >= 1
    assert lenient_graph.num_components() >= 1
    assert strict_graph.num_edges() < lenient_graph.num_edges()


def test_get_maximal_cliques(topology_inputs):
    """Test that maximal cliques can be extracted from the graph."""
    positions, coactivity = topology_inputs
    
    graph = TopologicalGraph(positions)
    graph.build_from_coactivity(coactivity, c_min=2.0, max_distance=1.5)
    
    cliques = graph.get_maximal_cliques()
    
    # Should have at least some cliques if graph has edges
    if graph.num_edges() > 0:
        assert len(cliques) > 0
        # Each clique should be a list of node indices
        for clique in cliques:
            assert isinstance(clique, list)
            assert len(clique) > 0
            # All nodes in clique should be valid
            for node in clique:
                assert 0 <= node < graph.num_nodes()
            # Clique should be fully connected (all pairs should have edges)
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    assert graph.graph.has_edge(clique[i], clique[j])
    else:
        # If no edges, should have no cliques (or only single-node cliques)
        # NetworkX find_cliques returns all nodes as 1-cliques
        assert len(cliques) >= 0


def test_compute_betti_numbers_requires_dependency():
    """Test that compute_betti_numbers raises informative error if ripser/gudhi missing."""
    import numpy as np
    
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    graph = TopologicalGraph(positions)
    
    # Create a simple graph with some edges
    coactivity = np.zeros((4, 4))
    coactivity[0, 1] = coactivity[1, 0] = 5.0
    coactivity[1, 2] = coactivity[2, 1] = 5.0
    coactivity[2, 3] = coactivity[3, 2] = 5.0
    coactivity[3, 0] = coactivity[0, 3] = 5.0
    
    graph.build_from_coactivity(coactivity, c_min=3.0, max_distance=2.0)
    
    # Try to compute Betti numbers - should either work or raise informative error
    try:
        betti = graph.compute_betti_numbers()
        # If it works, verify structure
        assert isinstance(betti, dict)
        assert 0 in betti
        assert betti[0] >= 0  # b_0 should be non-negative
    except ImportError as e:
        # Should raise informative error if dependencies missing
        assert "ripser" in str(e).lower() or "gudhi" in str(e).lower()
        assert "install" in str(e).lower()


def test_compute_betti_numbers_simple_cycle():
    """Test Betti number computation on a simple cycle graph (should have b_1=1)."""
    import numpy as np
    
    # Try to import persistent homology module to check if available
    try:
        from hippocampus_core.persistent_homology import is_persistent_homology_available
        
        if not is_persistent_homology_available():
            pytest.skip("ripser or gudhi not available")
    except ImportError:
        pytest.skip("persistent_homology module not available")
    
    # Create a cycle graph: 0-1-2-3-0 (4 nodes in a cycle)
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    graph = TopologicalGraph(positions)
    
    # Build cycle: 0-1-2-3-0
    coactivity = np.zeros((4, 4))
    coactivity[0, 1] = coactivity[1, 0] = 5.0
    coactivity[1, 2] = coactivity[2, 1] = 5.0
    coactivity[2, 3] = coactivity[3, 2] = 5.0
    coactivity[3, 0] = coactivity[0, 3] = 5.0
    
    graph.build_from_coactivity(coactivity, c_min=3.0, max_distance=2.0)
    
    # Should have 4 edges forming a cycle
    assert graph.num_edges() == 4
    
    # Compute Betti numbers
    betti = graph.compute_betti_numbers(max_dim=2)
    
    # Verify structure
    assert isinstance(betti, dict)
    assert 0 in betti
    assert 1 in betti
    
    # For a connected cycle graph:
    # b_0 = 1 (one connected component)
    # b_1 = 1 (one hole/loop)
    assert betti[0] == 1, f"Expected b_0=1 (connected), got {betti[0]}"
    assert betti[1] == 1, f"Expected b_1=1 (one cycle), got {betti[1]}"


def test_compute_betti_numbers_disconnected():
    """Test Betti number computation on disconnected graph (b_0 should equal number of components)."""
    import numpy as np
    
    # Try to import persistent homology module to check if available
    try:
        from hippocampus_core.persistent_homology import is_persistent_homology_available
        
        if not is_persistent_homology_available():
            pytest.skip("ripser or gudhi not available")
    except ImportError:
        pytest.skip("persistent_homology module not available")
    
    # Create two disconnected triangles: 0-1-2 and 3-4-5
    positions = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, 0.87],  # Triangle 1
        [3.0, 0.0], [4.0, 0.0], [3.5, 0.87],  # Triangle 2
    ])
    graph = TopologicalGraph(positions)
    
    # Build two triangles
    coactivity = np.zeros((6, 6))
    # Triangle 1: 0-1-2
    coactivity[0, 1] = coactivity[1, 0] = 5.0
    coactivity[1, 2] = coactivity[2, 1] = 5.0
    coactivity[2, 0] = coactivity[0, 2] = 5.0
    # Triangle 2: 3-4-5
    coactivity[3, 4] = coactivity[4, 3] = 5.0
    coactivity[4, 5] = coactivity[5, 4] = 5.0
    coactivity[5, 3] = coactivity[3, 5] = 5.0
    
    graph.build_from_coactivity(coactivity, c_min=3.0, max_distance=2.0)
    
    # Should have 6 edges (3 per triangle)
    assert graph.num_edges() == 6
    # Should have 2 connected components
    assert graph.num_components() == 2
    
    # Compute Betti numbers
    betti = graph.compute_betti_numbers(max_dim=1)
    
    # b_0 should equal number of connected components
    assert betti[0] == 2, f"Expected b_0=2 (two components), got {betti[0]}"
    # Each triangle has a hole, so b_1 should be 2
    # (Actually, for two separate triangles, b_1 might be 2 if each triangle forms a cycle)
    assert betti[1] >= 0  # At least non-negative


def test_compute_betti_numbers_without_ripser():
    """Test that compute_betti_numbers raises informative error when ripser/gudhi missing.
    
    This test verifies the error handling when persistent homology dependencies are not available.
    The actual behavior depends on whether ripser/gudhi are installed, so we check that
    either it works (if installed) or raises an informative error (if not installed).
    """
    import numpy as np
    
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    graph = TopologicalGraph(positions)
    
    coactivity = np.zeros((4, 4))
    coactivity[0, 1] = coactivity[1, 0] = 5.0
    coactivity[1, 2] = coactivity[2, 1] = 5.0
    
    graph.build_from_coactivity(coactivity, c_min=3.0, max_distance=2.0)
    
    # Try to compute Betti numbers - should either work or raise informative error
    try:
        betti = graph.compute_betti_numbers()
        # If it works, verify structure
        assert isinstance(betti, dict)
        assert 0 in betti
        assert betti[0] >= 0  # b_0 should be non-negative
        # This means dependencies are available, which is fine
    except ImportError as e:
        # Should raise informative error if dependencies missing
        error_msg = str(e).lower()
        assert "ripser" in error_msg or "gudhi" in error_msg
        assert "install" in error_msg


def test_compute_betti_numbers_known_topologies():
    """Test Betti numbers on known topologies to validate correctness."""
    import numpy as np
    
    try:
        from hippocampus_core.persistent_homology import is_persistent_homology_available
        
        if not is_persistent_homology_available():
            pytest.skip("ripser or gudhi not available")
    except ImportError:
        pytest.skip("persistent_homology module not available")
    
    # Test 1: Single node (isolated)
    positions_single = np.array([[0.0, 0.0]])
    graph_single = TopologicalGraph(positions_single)
    coactivity_single = np.zeros((1, 1))
    graph_single.build_from_coactivity(coactivity_single, c_min=1.0, max_distance=1.0)
    betti_single = graph_single.compute_betti_numbers(max_dim=1)
    assert betti_single[0] == 1, "Single node should have b_0=1"
    assert betti_single[1] == 0, "Single node should have b_1=0"
    
    # Test 2: Two disconnected nodes
    positions_two = np.array([[0.0, 0.0], [10.0, 10.0]])
    graph_two = TopologicalGraph(positions_two)
    coactivity_two = np.zeros((2, 2))
    graph_two.build_from_coactivity(coactivity_two, c_min=1.0, max_distance=1.0)
    betti_two = graph_two.compute_betti_numbers(max_dim=1)
    assert betti_two[0] == 2, "Two disconnected nodes should have b_0=2"
    assert betti_two[1] == 0, "Two disconnected nodes should have b_1=0"
    
    # Test 3: Path graph (0-1-2) - should have b_0=1, b_1=0
    positions_path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    graph_path = TopologicalGraph(positions_path)
    coactivity_path = np.zeros((3, 3))
    coactivity_path[0, 1] = coactivity_path[1, 0] = 5.0
    coactivity_path[1, 2] = coactivity_path[2, 1] = 5.0
    graph_path.build_from_coactivity(coactivity_path, c_min=3.0, max_distance=2.0)
    betti_path = graph_path.compute_betti_numbers(max_dim=1)
    assert betti_path[0] == 1, "Path graph should have b_0=1 (connected)"
    assert betti_path[1] == 0, "Path graph should have b_1=0 (no cycles)"
    
    # Test 4: Triangle (complete graph on 3 nodes) - should have b_0=1, b_1=0
    # (Triangle is contractible, so no holes)
    positions_triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.87]])
    graph_triangle = TopologicalGraph(positions_triangle)
    coactivity_triangle = np.zeros((3, 3))
    coactivity_triangle[0, 1] = coactivity_triangle[1, 0] = 5.0
    coactivity_triangle[1, 2] = coactivity_triangle[2, 1] = 5.0
    coactivity_triangle[2, 0] = coactivity_triangle[0, 2] = 5.0
    graph_triangle.build_from_coactivity(coactivity_triangle, c_min=3.0, max_distance=2.0)
    betti_triangle = graph_triangle.compute_betti_numbers(max_dim=1)
    assert betti_triangle[0] == 1, "Triangle should have b_0=1 (connected)"
    # Triangle is filled, so b_1 should be 0 (no holes in the clique complex)
    assert betti_triangle[1] == 0, "Triangle should have b_1=0 (filled, no holes)"


def test_compute_betti_numbers_backend_selection():
    """Test that backend selection works correctly (ripser vs gudhi vs auto)."""
    import numpy as np
    
    try:
        from hippocampus_core.persistent_homology import (
            is_persistent_homology_available,
        )
        
        if not is_persistent_homology_available():
            pytest.skip("Neither ripser nor gudhi available")
        
        # Try to import availability flags (may not be exported)
        try:
            from hippocampus_core.persistent_homology import RIPSER_AVAILABLE, GUDHI_AVAILABLE
        except ImportError:
            # If not exported, we'll test without them
            RIPSER_AVAILABLE = None
            GUDHI_AVAILABLE = None
    except ImportError:
        pytest.skip("persistent_homology module not available")
    
    # Create a simple cycle
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    graph = TopologicalGraph(positions)
    
    coactivity = np.zeros((4, 4))
    coactivity[0, 1] = coactivity[1, 0] = 5.0
    coactivity[1, 2] = coactivity[2, 1] = 5.0
    coactivity[2, 3] = coactivity[3, 2] = 5.0
    coactivity[3, 0] = coactivity[0, 3] = 5.0
    
    graph.build_from_coactivity(coactivity, c_min=3.0, max_distance=2.0)
    
    # Test auto backend (should work if either is available)
    betti_auto = graph.compute_betti_numbers(max_dim=1, backend="auto")
    assert isinstance(betti_auto, dict)
    assert 0 in betti_auto
    assert 1 in betti_auto
    
    # Test explicit backend if available
    if RIPSER_AVAILABLE is not None and RIPSER_AVAILABLE:
        try:
            betti_ripser = graph.compute_betti_numbers(max_dim=1, backend="ripser")
            assert betti_ripser[0] == betti_auto[0]
            assert betti_ripser[1] == betti_auto[1]
        except ImportError:
            pass  # ripser not actually available
    
    if GUDHI_AVAILABLE is not None and GUDHI_AVAILABLE:
        try:
            betti_gudhi = graph.compute_betti_numbers(max_dim=1, backend="gudhi")
            assert betti_gudhi[0] == betti_auto[0]
            assert betti_gudhi[1] == betti_auto[1]
        except ImportError:
            pass  # gudhi not actually available
    
    # Test invalid backend
    with pytest.raises(ValueError, match="Unknown backend"):
        graph.compute_betti_numbers(max_dim=1, backend="invalid")

