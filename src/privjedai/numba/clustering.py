import numpy as np
from numba import njit

@njit
def numba_get_unique_edges(candidates: np.ndarray) -> np.ndarray:
    """Numba-optimized edge pruning."""
    if candidates.shape[0] == 0:
        return candidates

    max_val = int(candidates.max())
    used = np.zeros(max_val + 1, dtype=np.bool_)

    n = candidates.shape[0]
    # Pre-allocate array instead of appending to a list (Massive speedup in Numba)
    unique_edges_out = np.empty((n, 2), dtype=candidates.dtype)
    count = 0

    for i in range(n):
        u = candidates[i, 0]
        v = candidates[i, 1]

        if not used[u] and not used[v]:
            used[u] = True
            used[v] = True
            unique_edges_out[count, 0] = u
            unique_edges_out[count, 1] = v
            count += 1

    # Slice to return only the filled portion
    return unique_edges_out[:count]


@njit
def numba_isolated_edges(edges: np.ndarray) -> np.ndarray:
    """
    Numba-optimized version of the scipi_connected_components logic.
    Fuses bincount and masking to avoid intermediate memory allocations.
    """
    if edges.shape[0] == 0:
        return np.empty((0, 2), dtype=edges.dtype)

    n_nodes = edges.max() + 1

    # Numba supports np.bincount perfectly
    degrees = np.bincount(edges.ravel(), minlength=n_nodes)

    n = edges.shape[0]
    out = np.empty((n, 2), dtype=edges.dtype)
    count = 0

    for i in range(n):
        u = edges[i, 0]
        v = edges[i, 1]
        # Direct check instead of boolean masking arrays
        if degrees[u] == 1 and degrees[v] == 1:
            out[count, 0] = u
            out[count, 1] = v
            count += 1

    return out[:count]