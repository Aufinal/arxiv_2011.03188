"""
Module `comparison_benchmarks.py`

Benchmarks clustering algorithms based on the embedding methods in `embeddings.py`.
"""

from os import path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from clustering import AdjacencyClustering, LaplacianClustering, SVDClustering
from generator import fast_sbm
from utils import compute_rzero, tridiag_toeplitz


def compute_performance(d=3, n_clusters=2, n=1000, n_samples=10, n_eta=10):
    results = np.zeros((3, n_eta, 2))
    eta_list = np.linspace(0.5, 1, n_eta)

    # Degree parameter s so that the mean degree of the graph is d
    s = n_clusters * d * 1 / (3 / 2 - 1 / n_clusters ** 2)

    for (eta_idx, eta) in enumerate(eta_list):

        F, eigenvalues = tridiag_toeplitz(
            n_clusters, 0.5 * s, s * eta, s * (1 - eta), return_eigenvectors=False
        )

        rho, r0 = compute_rzero(eigenvalues / n_clusters)
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


if __name__ == "__main__":
    n, n_samples, n_list = 2000, 50, 50

    K = [2, 4, 6]
    D = [2, 3, 4]
    for k in K:
        for d in D:
            filename = f"data/benchmark_d{d}_k{k}"
            if not path.exists(filename):
                res = compute_performance(d, k, n, n_samples, n_list)
                np.savetxt(filename, res.reshape(3, n_list * 2))
