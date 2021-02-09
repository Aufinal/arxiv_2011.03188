"""
    Module `utils.py`
    Utility functions for spectral clustering.

    Authors: S. Coste and L. Stephan
"""

import numpy as np
from scipy.sparse import diags


def compute_rzero(eigs):
    """
    Given a vector of eigenvalues, returns the max eigenvalue (in magnitude) ρ and
    the number of eigenvalues above the threshold sqrt(ρ).

    Input:
        - eigs: vector of eigenvalues
    Output:
        - rho: max eigenvalue, in magnitude
        - r0: number of eigenvalues above sqrt(rho)
    """

    # We have rho = max(eigs) by the Perron-Frobenius theorem
    rho = np.amax(eigs)
    r0 = np.count_nonzero(eigs ** 2 > rho)

    if r0 == 0:
        # If there are no informative eigenvalues, we still try to cluster on the
        # top eigenvector
        return rho, 1
    else:
        return rho, int(r0)


def tridiag_toeplitz(n, a, b, c, format="dense", return_eigenvectors=False):
    """
    Creates a tridiagonal Toeplitz matrix of size n, with the specified inputs on its
    diagonals. Also returns its eigenvalues, and if needed its eigenvectors

    Input :
        - n: size of the output matrix
        - a, b, c: values to place on the main, upper and lower diagonals, respectively.
        - format: type of the returned matrix. If "coo", a COO scipy matrix is returned;
        if "dense", a numpy array is returned instead
        - return_eigenvectors: whether to return the eigenvectors along with the
        eigenvalues.

    Output:
        - A: a tridiagonal matrix with the specified format and diagonals
        - v (or (v, left, right)): the eigenvalues (or eigenvalues + eigenvectors) of A
    """
    main = np.full(n, a)
    upper = np.full(n - 1, b)
    lower = np.full(n - 1, c)
    matrix = diags([main, upper, lower], [0, 1, -1], format=format, dtype=float)

    x = 1 + np.arange(n)
    eigs = a + 2 * np.sqrt(b * c) * np.cos(np.pi * x / (n + 1))

    if format == "dense":
        matrix = matrix.toarray()

    if return_eigenvectors:
        left_eigen = np.sqrt((c / b) ** x[:, np.newaxis]) * np.sin(
            x[:, np.newaxis] * x[np.newaxis, :] * np.pi / (n + 1)
        )
        right_eigen = np.sqrt((b / c) ** x[:, np.newaxis]) * np.sin(
            x[:, np.newaxis] * x[np.newaxis, :] * np.pi / (n + 1)
        )
        return matrix, (eigs, left_eigen, right_eigen)

    return matrix, eigs
