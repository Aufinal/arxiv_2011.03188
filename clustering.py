"""
    Module `clustering.py`

    Implements several spectral clustering algorithms with a common interface.

    Authors: S. Coste and L. Stephan

    References:
    [1]: Simon Coste and Ludovic Stephan. A simpler spectral approach for clustering in
    directed networks, 2021. https://arxiv.org/abs/2102.03188
    [2]: Steinar Laenen and He Sun. Higher-order spectral clustering of directed graphs,
    2020. https://arxiv.org/abs/2011.05080
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs, eigsh, svds
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture


class ClusteringAlgorithm:
    """
    Generic class for clustering algorithms.
    Implements the pipeline embedding -> geometric clustering ; the embedding
    details are deferred to subclasses (via `get_embedding`) while the geometric
    clustering algorithm is specified via a `cluster_class` attribute.

    Attributes:
        - n_clusters: number of clusters of the problem
        - cluster_class: sklearn class used for geometric
    """

    cluster_class = KMeans

    def __init__(self, n_clusters=2, cluster_class=None, **kwargs):
        super().__init__()
        self.n_clusters = n_clusters

        if cluster_class is not None:
            self.cluster_class = cluster_class

    def get_embedding(self, G):
        """
        Placeholder function for the embedding method.
        Must return a matrix of size (n_features, embedding_dimension).

        Input:
            - G: adjacency matrix of the graph to embed, in COO form.
        """
        pass

    def _get_cluster_param(self):
        # For gaussian mixture, clustering, the number of clusters is denoted
        # by `n_components`, so we need a small helper function
        if self.cluster_class in [GaussianMixture, BayesianGaussianMixture]:
            return "n_components"
        else:
            return "n_clusters"

    def cluster(self, embedding):
        param = self._get_cluster_param()
        return self.cluster_class(**{param: self.n_clusters}).fit_predict(embedding)

    def run(self, G):
        return self.cluster(self.get_embedding(G))


class AdjacencyClustering(ClusteringAlgorithm):
    cluster_class = GaussianMixture

    def __init__(self, n_clusters=2, rho=0, r0=1):
        super().__init__(n_clusters=n_clusters)
        self.rho = rho
        self.r0 = r0

    def get_embedding(self, G):
        """
        Embedding based on the left/right eigenvectors of the adjacency matrix.
        The parameter r0 denotes the number of informative eigenvalues; see [1] for
        details.
        """
        maxiter = G.shape[0] * 100
        _, vec_r = eigs(
            G, k=self.r0, which="LM", maxiter=maxiter, tol=1e-6, sigma=self.rho
        )
        _, vec_l = eigs(
            G.T, k=self.r0, which="LM", maxiter=maxiter, tol=1e-6, sigma=self.rho
        )
        return np.concatenate((vec_l.real, vec_r.real), axis=1)


class SVDClustering(ClusteringAlgorithm):
    def get_embedding(self, G):
        """
        Embedding based on the singular vectors of the adjacency matrix.
        """
        vec_l, _, vec_r = svds(G, k=self.n_clusters)
        return np.concatenate((vec_l.real, vec_r.real.T), axis=1)


class LaplacianClustering(ClusteringAlgorithm):
    def get_laplacian(self, G):
        """
        Normalized Hermitian Laplacian as defined in [2], in sparse COO form.
        """
        n = G.shape[0]
        # From [2]:  omega is a ceil(2π*n_clusters) root of unity
        omega = np.exp(2 * np.pi * 1j / np.ceil(2 * np.pi * self.n_clusters))
        degrees = G.dot(np.ones(n)) + G.T.dot(np.ones(n))
        invsqrt_deg = np.divide(
            1, np.sqrt(degrees), out=np.zeros_like(degrees), where=(degrees != 0)
        )

        # We compute I - D^{1/2}AD^{-1/2} by hand
        data, row, col = G.data, G.row, G.col
        data1 = omega * data * invsqrt_deg[row] * invsqrt_deg[col]
        new_data = np.concatenate((-data1, -np.conj(data1), np.ones(n)))
        new_row = np.concatenate((row, col, np.arange(n)))
        new_col = np.concatenate((col, row, np.arange(n)))
        return coo_matrix((new_data, (new_row, new_col)), shape=(n, n))

    def get_embedding(self, G):
        """
        Embedding based on the lowest eigenvector of the normalized Hermitian Laplacian.
        """
        L = self.get_laplacian(G)
        maxiter = G.shape[0] * 200
        _, vec = eigsh(
            L,
            k=1,
            which="LM",
            tol=1e-6,
            sigma=-0.1,
            maxiter=maxiter,
        )
        return np.concatenate((vec.real, vec.imag), axis=1)
