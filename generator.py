"""
GENERATION OF DSBM
----------------------------------------------------------------------------------
The functions create_connectivity matrix and tridiag_toeplitz are for creating the
underlying matrices F.
The function fast_sbm generates an instance of the directed stochastic blockmodel, in full generality
with arbitrary number of clusters and arbitrary clusters sizes.
"""

import numpy as np
from scipy.sparse import coo_matrix, diags


def create_connectivity_matrix(a, b, c, d):
    """
    Input :
        4 positive numbers
    Output :
        the 2x2 connectivity matrix corresponding to the inputs and
        its 2 eigenvalues.
    """

    F = np.array([[a, b], [c, d]]).astype(float)
    eigs = (a + d) / 2 + np.sqrt((a - d) ** 2 / 4 + b * c) * np.array([1.0, -1.0])
    return F, eigs


def tridiag_toeplitz(n, a, b, c, format="coo", return_eigenvectors=True):
    """
    Creates a tridiagonal Toeplitz matrix of size n,
    with a on the main diagonal, b on the upper and c on the lower one.

    Inputs :
        n = the size of the output matrix
        a,b,c, positive numbers

    Returns:
        A sparse COO matrix as requested + its eigenvalues. If return_eigenvectors==True
        returns the left and right eigenvectors.
    """
    main = np.full(n, a)
    upper = np.full(n - 1, b)
    lower = np.full(n - 1, c)
    matrix = diags([main, upper, lower], [0, 1, -1], format=format, dtype=float)

    x = np.linspace(1, n, num=n, endpoint=True)
    eigs = a + 2 * np.sqrt(b * c) * np.cos(np.pi * x * 1.0 / (n + 1))

    if return_eigenvectors:
        eigenvectors = np.sqrt((c / b) ** x[:, np.newaxis]) * np.sin(
            x[:, np.newaxis] * x[np.newaxis, :] * np.pi / (n + 1)
        )
        eigen_inv = np.sqrt((b / c) ** x[:, np.newaxis]) * np.sin(
            x[:, np.newaxis] * x[np.newaxis, :] * np.pi / (n + 1)
        )
        return matrix, (eigs, eigenvectors, eigen_inv)

    return matrix.todense(), eigs


default, _ = create_connectivity_matrix(2, 2, 2, 2)
default_size = np.array([0.5, 0.5])


def fast_sbm(n_nodes, F, left_proportions=None, right_proportions=None):
    """
    Fast generation of a directed SBM graph, efficient in the sparse regime.

    Inputs :
        -n_nodes is the number of nodes.
        -F is the connectivity matrix.
        -left/right proportions are the vectors of cluster proportions,
        ie the size of the i-th cluster is int( n * left_proportion[i] ) on the left
        and  int( n * right_proportion[i] ) on the right.

        By default, F = [2, 2 \\ 2, 2] and p = [1/2, 1/2] which corresponds to the
        directed Erdös-Rényi graph with mean degree 4.

    Returns :
        - a sparse COO matrix with the edges of the SBM with connectivity F[i,j]/n_nodes on
         the cluster (i,j).

    Method : for each pair 1 <= i, j <= k, the subgraph spanned by clusters i, j
    is uniformly random with density F[i, j]. We first sample the number of its edges
    which is Poisson(F[i,j]) and then uniformly choose the edges.
    """

    n_clusters = F.shape[0]

    if right_proportions is None:
        right_proportions = np.ones(n_clusters) / n_clusters

    if left_proportions is None:
        left_proportions = np.ones(n_clusters) / n_clusters

    # Cluster sizes are rounded below
    row_sizes = np.array(n_nodes * right_proportions, dtype=int)
    col_sizes = np.array(n_nodes * left_proportions, dtype=int)

    row_offsets = np.cumsum(row_sizes) - row_sizes
    col_offsets = np.cumsum(col_sizes) - col_sizes

    # Pad the last cluster to reach n nodes in total
    row_sizes[-1] += n_nodes - (row_offsets[-1] + row_sizes[-1])
    col_sizes[-1] += n_nodes - (col_offsets[-1] + col_sizes[-1])

    # True labels
    row_labels = np.repeat(np.arange(n_clusters), row_sizes)
    col_labels = np.repeat(np.arange(n_clusters), col_sizes)

    # We first draw the number of edges per cluster
    n_edges = np.zeros((n_clusters, n_clusters), dtype=int)
    for i in range(n_clusters):
        for j in range(n_clusters):
            n_edges[i, j] = np.random.binomial(
                row_sizes[i] * col_sizes[j], F[i, j] / n_nodes
            )

    tot_edges = np.sum(n_edges)
    rows = np.zeros(tot_edges)
    cols = np.zeros(tot_edges)
    data = np.ones(tot_edges)
    curr_index = 0

    for i in range(n_clusters):
        for j in range(n_clusters):

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

    # Finally, set diag to 0
    # output.setdiag(np.zeros(actual_size))
    # output.eliminate_zeros()
    return output, (row_labels, col_labels)
