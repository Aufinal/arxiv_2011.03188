"""
    Module `generator.py`
    Efficient generation of directed SBMs.

    Authors: S. Coste and L. Stephan
"""

import numpy as np
from scipy.sparse import coo_matrix


def fast_sbm(n_nodes, F, left_proportions=None, right_proportions=None):
    """
    Fast generation of a directed SBM graph, efficient in the sparse regime.

    Input:
        - n_nodes: number of nodes in the graph
        - F: connectivity matrix
        - left_proportions: proportions of vertices that are in the respective left
        clusters. Must have size equal to F.shape[0]. If not supplied, uniform clusters
        are assumed.
        - right_proportions: idem, but for the right clusters. Must have size equal to
        F.shape[1].

    Output:
        - G: a sparse random graph such that a edge (u, v) with u in the i-th left
        cluster and v in the i-th right cluster has probability F[i, j]/n_nodes of
        appearing. Returned in sparse COO form.
        - (row_labels, col_labels): left and right clusters of each vertex

    Method: for each pair 1 <= i, j <= k, the subgraph spanned by clusters i, j
    is uniformly random with density F[i, j]. We first sample the number of its edges
    which follows a binomial distribution, and then uniformly choose the edges.
    """

    l_clusters, r_clusters = F.shape

    if right_proportions is None:
        right_proportions = np.ones(r_clusters) / r_clusters

    if left_proportions is None:
        left_proportions = np.ones(l_clusters) / l_clusters

    # Cluster sizes are rounded below
    row_sizes = np.array(n_nodes * left_proportions, dtype=int)
    col_sizes = np.array(n_nodes * right_proportions, dtype=int)

    row_offsets = np.cumsum(row_sizes) - row_sizes
    col_offsets = np.cumsum(col_sizes) - col_sizes

    # Pad the last cluster to reach n nodes in total
    row_sizes[-1] += n_nodes - (row_offsets[-1] + row_sizes[-1])
    col_sizes[-1] += n_nodes - (col_offsets[-1] + col_sizes[-1])

    # True labels
    row_labels = np.repeat(np.arange(l_clusters), row_sizes)
    col_labels = np.repeat(np.arange(r_clusters), col_sizes)

    # We first draw the number of edges per cluster
    n_edges = np.zeros((l_clusters, r_clusters), dtype=int)
    for i in range(l_clusters):
        for j in range(r_clusters):
            n_edges[i, j] = np.random.binomial(
                row_sizes[i] * col_sizes[j], F[i, j] / n_nodes
            )

    tot_edges = np.sum(n_edges)
    rows = np.zeros(tot_edges)
    cols = np.zeros(tot_edges)
    data = np.ones(tot_edges)
    curr_index = 0

    for i in range(l_clusters):
        for j in range(r_clusters):

            # Flattened indices of edges in the block [i,j]
            edge_indices = np.random.choice(
                row_sizes[i] * col_sizes[j],
                size=n_edges[i, j],
                replace=False,
            )

            # Unflatten indices and offset them
            x = np.floor(edge_indices * 1.0 / col_sizes[j]).astype(int, copy=False)
            y = (edge_indices - x * col_sizes[j]).astype(int, copy=False)
            x += row_offsets[i]
            y += col_offsets[j]

            # Put them in rows and cols
            rows[curr_index : (curr_index + n_edges[i, j])] = x
            cols[curr_index : (curr_index + n_edges[i, j])] = y
            curr_index += n_edges[i, j]

    output = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=float)

    return output, (row_labels, col_labels)
