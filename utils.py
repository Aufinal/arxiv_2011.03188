import numpy as np


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
