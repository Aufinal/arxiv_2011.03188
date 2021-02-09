"""
    Module `pictures_eigvals.py`
    Contains utilities to compute and plot the expected and actual eigenvalues
    of graphs generated according to the directed stochastic block model.

    Authors: S. Coste, L. Stephan

    References:
    [1]: Simon Coste and Ludovic Stephan. A simpler spectral approach for clustering in
    directed networks, 2021. https://arxiv.org/abs/2102.03188
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.sparse.linalg import eigs

from generator import fast_sbm
from utils import compute_rzero

rc("text", usetex=True)
rc("font", family="serif")


def compute_pi(p, q, n_nodes=100):
    """
    Computes the bi-cluster assignment matrix defined in [1].

    Input:
        - p: left cluster relative size
        - q: right cluster relative size
        - n_nodes: number of nodes in the model
    Output:
        - Pi: matrix such that Pi[i, j] is the number of vertices with left cluster
        j and right cluster i (notice the inversion of i and j).
    """
    rowsums = np.zeros(1 + p.size, dtype=int)
    colsums = np.zeros(1 + q.size, dtype=int)

    rowsums[1:] = np.cumsum(n_nodes * p, dtype=int)
    colsums[1:] = np.cumsum(n_nodes * q, dtype=int)
    rowsums[-1] = n_nodes
    colsums[-1] = n_nodes

    sigma_p = np.zeros((p.size, n_nodes), dtype=int)
    sigma_q = np.zeros((q.size, n_nodes), dtype=int)
    for i in range(p.size):
        for n in range(rowsums[i], rowsums[i + 1]):
            sigma_p[i, n] = 1

    for i in range(q.size):
        for n in range(colsums[i], colsums[i + 1]):
            sigma_q[i, n] = 1

    return sigma_q.dot(sigma_p.T) / n_nodes


def matrix_colorplot(F, p, q, filename):
    """
    Plots (and saves) the colorplot of the expectation matrix of a stochastic block
    model. Darker purple indicates higher edge probability.
    The SBM is assumed to have n = 100 nodes.

    Input:
        - F: connectivity matrix
        - p: left cluster relative size
        - q: right cluster relative size
        - filename: file to save plot to

    Output: none, but shows the saved colorplot
    """
    n_nodes = 100
    matrix = np.zeros((n_nodes, n_nodes))
    l_clusters, r_clusters = F.shape
    row = np.zeros(1 + l_clusters, dtype=int)
    col = np.zeros(1 + r_clusters, dtype=int)

    row[1:] = (n_nodes * np.cumsum(p)).astype(int)
    col[1:] = (n_nodes * np.cumsum(q)).astype(int)

    # Fix rounding errors
    row[-1] = n_nodes
    col[-1] = n_nodes
    for i in range(l_clusters):
        for j in range(r_clusters):
            matrix[row[i] : row[i + 1], col[j] : col[j + 1]] = F[i, j]
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.matshow(matrix, cmap="Purples")
    plt.savefig(filename, bbox_inches="tight")

    plt.show()


def plot_eigenvalues(
    n_nodes,
    F,
    p,
    q,
    filename,
    eigenvaluecolor="rebeccapurple",
    circlecolor="wheat",
    circlealpha=0.6,
    linescolor="sienna",
):
    """
    Generates a directed SBM with given parameters, and plots its eigenvalues along
    with the theory predicted in [1].

    Input:
        - F: connectivity matrix
        - p: left cluster relative sizes
        - q: right cluster relative sizes
        - filename: file to save plot to

    Optional:
        - eigenvaluecolor: color for actual eigenvalues
        - linescolor: color for theoretical eigenvalues (vertical lines)
        - circlecolor: color for theoretical bulk boundary
        - circlealpha: transparency of bulk boundary

    Output: none, but shows saved plot
    """
    pi = compute_pi(p, q, n_nodes=n_nodes)
    theory_eigs, _ = np.linalg.eig(F.dot(pi))
    rho, r0 = compute_rzero(theory_eigs)
    print(rho, r0)

    A, _ = fast_sbm(n_nodes, F, left_proportions=p, right_proportions=q)
    spec = eigs(A, k=n_nodes - 2, return_eigenvectors=False)
    spec = np.sort(spec)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=30)

    # Plotting theory : vertical line at each eigenvalue, and circle for bulk
    for z in theory_eigs:
        ax.axvline(z, color=linescolor, linewidth=5, alpha=0.5)
    circle = plt.Circle(
        (0, 0),
        np.sqrt(rho),
        color=circlecolor,
        linestyle="-",
        linewidth=2,
        alpha=circlealpha,
    )
    ax.add_artist(circle)

    # Plotting actual eigenvalues, with emphasis on informative ones
    ax.scatter(spec.real, spec.imag, s=5, c=eigenvaluecolor, zorder=10)
    ax.scatter(spec[-r0:].real, spec[-r0:].imag, s=60, c="k", zorder=10, marker="x")

    plt.savefig(filename, bbox_inches="tight")
    plt.show()


# ------------
# Actual picture generation
# ------------

if __name__ == "__main__":

    # ----------
    # First picture
    # ----------
    # Simple path-SBM with p = 10, eta = 0.8

    F = np.array([[5, 8], [2, 5]])
    p = np.array([1 / 2, 1 / 2])
    matrix_colorplot(F, p, p, "pictures/F1.pdf")
    plot_eigenvalues(2000, F, p, p, "pictures/eigenvalues1.pdf")

    # ----------
    # Second picture
    # ----------
    # Explicitly given F, unbalanced p and q

    F = np.array([[15, 1, 7], [0, 8, 2], [1, 10, 15]])
    p = np.array([0.3, 0.2, 0.5])
    q = np.array([0.7, 0.1, 0.2])
    matrix_colorplot(F, p, q, "pictures/F2.pdf")
    plot_eigenvalues(2000, F, p, q, "pictures/eigenvalues2.pdf")

    # ----------
    # Third picture
    # ----------
    # Balanced blocks, purely triangular connectivity matrix

    b = 4
    F = np.zeros((b, b))
    for i in range(b):
        F[i, i] = 1 + 4 * i
        for j in range(i + 1, b):
            F[i, j] = 10 * np.random.rand()

    p = np.array([1 / b for i in range(b)])
    matrix_colorplot(F, p, p, "pictures/F3.pdf")
    plot_eigenvalues(2000, F, p, p, "pictures/eigenvalues3.pdf")
