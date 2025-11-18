"""Optional persistent homology computation for topological verification.

This module provides functions to compute Betti numbers from clique complexes,
enabling verification that the learned graph topology matches the physical environment.

Requires ``ripser`` or ``gudhi`` for persistent homology computation. Install via:

    python3 -m pip install ripser

or

    python3 -m pip install gudhi

"""
from __future__ import annotations

from typing import Optional

try:
    import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


def compute_betti_numbers_from_cliques(
    cliques: list[list[int]],
    max_dim: int = 2,
    backend: str = "auto",
) -> dict[int, int]:
    """Compute Betti numbers from a list of cliques (clique complex).

    Parameters
    ----------
    cliques:
        List of cliques, where each clique is a list of node indices.
        Each k-clique (k nodes) will be treated as a (k-1)-simplex.
    max_dim:
        Maximum dimension for which to compute Betti numbers (default: 2).
        b_0, b_1, ..., b_max_dim will be computed.
    backend:
        Backend to use: "ripser", "gudhi", or "auto" (default: "auto").
        If "auto", uses ripser if available, otherwise gudhi.

    Returns
    -------
    dict[int, int]
        Dictionary mapping dimension to Betti number: {0: b_0, 1: b_1, ...}
        - b_0: number of connected components
        - b_1: number of 1D holes (loops)
        - b_2: number of 2D holes (voids)
        - etc.

    Raises
    ------
    ImportError
        If neither ripser nor gudhi is available and backend is "auto",
        or if the specified backend is not available.

    Notes
    -----
    The clique complex is built by treating each k-clique as a (k-1)-simplex.
    For example:
    - 2-clique (edge) → 1-simplex
    - 3-clique (triangle) → 2-simplex
    - 4-clique (tetrahedron) → 3-simplex

    This matches the approach in Hoffman et al. (2016) for building
    the clique coactivity complex.

    Examples
    --------
    >>> from hippocampus_core.persistent_homology import compute_betti_numbers_from_cliques
    >>> 
    >>> # Example: cycle graph (4 nodes in a square)
    >>> cliques = [
    ...     [0, 1], [1, 2], [2, 3], [3, 0],  # Edges
    ...     [0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1],  # Triangles (if present)
    ... ]
    >>> betti = compute_betti_numbers_from_cliques(cliques, max_dim=1)
    >>> print(f"b_0 (components): {betti[0]}")
    >>> print(f"b_1 (holes): {betti[1]}")
    """

    if backend == "auto":
        if RIPSER_AVAILABLE:
            backend = "ripser"
        elif GUDHI_AVAILABLE:
            backend = "gudhi"
        else:
            raise ImportError(
                "Neither ripser nor gudhi is available. "
                "Install one with: python3 -m pip install ripser"
            )
    elif backend == "ripser" and not RIPSER_AVAILABLE:
        raise ImportError(
            "ripser backend requested but ripser is not installed. "
            "Install with: python3 -m pip install ripser"
        )
    elif backend == "gudhi" and not GUDHI_AVAILABLE:
        raise ImportError(
            "gudhi backend requested but gudhi is not installed. "
            "Install with: python3 -m pip install gudhi"
        )

    if backend == "ripser":
        return _compute_betti_ripser(cliques, max_dim)
    elif backend == "gudhi":
        return _compute_betti_gudhi(cliques, max_dim)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _compute_betti_ripser(cliques: list[list[int]], max_dim: int) -> dict[int, int]:
    """Compute Betti numbers using ripser backend."""
    # Build simplicial complex from cliques
    # ripser expects a distance matrix or point cloud, but we can build
    # a flag complex from the cliques directly
    
    # For ripser, we need to build a flag complex
    # We'll create a graph where edges exist if nodes are in a clique together
    # Then use ripser on the flag complex
    
    # Alternative: build distance matrix from clique structure
    # For now, we'll use a simpler approach: build a graph and use ripser's
    # flag complex construction
    
    import numpy as np
    
    # Find all unique nodes
    all_nodes = set()
    for clique in cliques:
        all_nodes.update(clique)
    nodes = sorted(all_nodes)
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Build adjacency matrix from cliques
    # Two nodes are adjacent if they appear together in at least one clique
    adj_matrix = np.zeros((n, n), dtype=bool)
    for clique in cliques:
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                idx_i = node_to_idx[clique[i]]
                idx_j = node_to_idx[clique[j]]
                adj_matrix[idx_i, idx_j] = True
                adj_matrix[idx_j, idx_i] = True
    
    # Build distance matrix (0 if adjacent, inf if not)
    # For flag complex, we use 0 for edges, 1 for non-edges
    dist_matrix = np.where(adj_matrix, 0.0, 1.0)
    np.fill_diagonal(dist_matrix, 0.0)
    
    # Compute persistent homology using ripser
    # Use distance matrix mode
    # Note: Suppress warning about square matrix - we're correctly using metric="precomputed"
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*distance_matrix.*", category=UserWarning)
        result = ripser.ripser(dist_matrix, maxdim=max_dim, metric="precomputed")
    
    # Extract Betti numbers from persistence diagram
    # Betti number = number of bars that persist to infinity
    betti = {}
    for dim in range(max_dim + 1):
        if dim < len(result["dgms"]):
            # Count bars that persist (death == inf)
            dgm = result["dgms"][dim]
            if len(dgm) > 0:
                # Bars with death == inf are persistent features
                persistent = np.isinf(dgm[:, 1])
                betti[dim] = int(np.sum(persistent))
            else:
                betti[dim] = 0
        else:
            betti[dim] = 0
    
    return betti


def _compute_betti_gudhi(cliques: list[list[int]], max_dim: int) -> dict[int, int]:
    """Compute Betti numbers using gudhi backend."""
    # Build simplicial complex from cliques
    st = gudhi.SimplexTree()
    
    # Add simplices from cliques
    # A k-clique becomes a (k-1)-simplex
    for clique in cliques:
        if len(clique) > 0:
            # Sort clique for consistency
            sorted_clique = sorted(clique)
            # Add simplex with dimension = len(clique) - 1
            st.insert(sorted_clique)
    
    # Compute persistence
    st.compute_persistence()
    
    # Extract Betti numbers
    betti = {}
    for dim in range(max_dim + 1):
        betti[dim] = st.betti_number(dim)
    
    return betti


def is_persistent_homology_available() -> bool:
    """Check if persistent homology computation is available.

    Returns
    -------
    bool
        True if either ripser or gudhi is available, False otherwise.
    """
    return RIPSER_AVAILABLE or GUDHI_AVAILABLE


# Export availability flags for testing
__all__ = [
    "compute_betti_numbers_from_cliques",
    "is_persistent_homology_available",
    "RIPSER_AVAILABLE",
    "GUDHI_AVAILABLE",
]

