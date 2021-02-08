"""
Module `comparison_benchmarks.py`

Benchmarks clustering algorithms based on the embedding methods in `embeddings.py`.

Authors: S. Coste and L. Stephan
"""

import time

import numpy as np
from clustering import AdjacencyClustering, LaplacianClustering, SVDClustering
from generator import fast_sbm, tridiag_toeplitz
from sklearn.metrics import adjusted_rand_score


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


def compute_performance(d=10, n_clusters=2, n=1000, n_samples=10, n_eta=10):
    results = np.zeros((3, n_eta, 2))
    eta_list = np.linspace(0.5, 1, n_eta)

    for (eta_idx, eta) in enumerate(eta_list):

        F, eigenvalues = tridiag_toeplitz(
            n_clusters, 0.5 * d, d * eta, d * (1 - eta), return_eigenvectors=False
        )

        rho, r0 = compute_rzero(eigenvalues / k)
        algorithms = [
            AdjacencyClustering(n_clusters=n_clusters, rho=rho, r0=r0),
            SVDClustering(n_clusters=n_clusters),
            LaplacianClustering(n_clusters=n_clusters),
        ]

        print(f"Step {eta_idx}/{n_eta}")

        scores = np.zeros((3, n_samples))

        for sample_idx in range(n_samples):
            G, (true_labels, _) = fast_sbm(n, F)

            for i, algorithm in enumerate(algorithms):
                scores[i, sample_idx] = adjusted_rand_score(
                    true_labels, algorithm.run(G)
                )

        results[:, eta_idx, 0] = np.mean(scores, axis=1)
        results[:, eta_idx, 1] = np.std(scores, axis=1)

    return results


st = time.time()
n, n_samples, n_list = 2000, 50, 50

K = [4, 6]
D = [5, 10, 15]
for k in K:
    for d in D:
        filename = f"data/benchmark_d{d}_k{k}"
        res = compute_performance(d, k, n, n_samples, n_list)
        np.savetxt(filename, res.reshape(3, n_list * 2))

print(time.time() - st)
