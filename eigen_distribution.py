"""
    Module `eigen_distribution.py`.
    Plots the empirical distribution of eigenvectors of random graphs, along with
    Gaussian approximations.

    References:
    [1]: Submitted paper
"""
from os import path

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import eigs

from generator import fast_sbm
from utils import compute_rzero

n = 1000
n_tries = 5


def gaussian_params(F, p):
    """
    Computes the mean and variances of each informative eigenvector in a DSBM for each
    cluster. Details of the computations are in [1].

    Input:
        - F : connectivity matrix
        - p : proportion of vertices in each cluster

    Output:
        - means: array of shape (r, r0) such that means[i, j] is the (theoretical) mean
        of the j-th eigenvector on the i-th cluster.
        - variances : idem for variances
    """
    r = p.size
    mod = F.dot(np.diag(p))
    eigvals, eigvecs = np.linalg.eig(mod)

    # Take real part and positive first coordinate
    eigvals = eigvals.real
    eigvecs = eigvecs.real

    for i in range(r):
        if eigvecs[0, i] < 0:
            eigvecs[:, i] = -eigvecs[:, i]

    # Compute r0
    rho = np.amax(eigvals)
    idx = np.flatnonzero(eigvals ** 2 > rho)
    r0 = idx.size

    # Only keep informative eigenvalues/eigenvectors
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvsq = eigvecs * eigvecs
    means = np.zeros((r, r0))
    variances = np.zeros((r, r0))
    for i in range(r0):
        defects = np.linalg.inv(np.eye(r) - mod / (eigvals[i] ** 2)).dot(eigvsq[:, i])
        normsq_ui = np.dot(p, defects)
        means[:, i] = eigvecs[:, i] / np.sqrt(normsq_ui)
        variances[:, i] = (defects - eigvsq[:, i]) / normsq_ui

    return means, variances


def gen_data(F, p, n_nodes=1000, n_tries=20, datafile=None):
    """
    Input:
        - F : connectivity matrix
        - p : proportion of vertices in each cluster
        - n_nodes: size of the generated graph. Default is 1000
        - n_tries: number of graphs generated. Default is 20.
        - datafile: if provided, the generated data is saved to disk.

    Output:
        - tot: (n * n_tries, r_0) matrix containing the aggregated eigenvectors.
        Each eigenvectors has norm sqrt(n).
    """

    mod = F.dot(np.diag(p))
    eigvals = np.linalg.eigvals(mod)
    _, r0 = compute_rzero(eigvals)
    tot = np.zeros((n_nodes * n_tries, r0))

    for i in range(n_tries):
        G, _ = fast_sbm(n_nodes, F, left_proportions=p, right_proportions=p)
        vals, v = eigs(G, r0, which="LM", return_eigenvectors=True)

        # Take eigenvectors aligned with the theoretical ones
        for k in range(r0):
            if v[0, k] < 0:
                v[:, k] = -v[:, k]
        tot[i * n_nodes : (i + 1) * n_nodes, :] = np.sqrt(n_nodes) * v.real

    if datafile is not None:
        np.savetxt(datafile, tot)

    return tot


def plot_gaussianmix(data, means, variances, proportions, filename=None):
    """
    Plots a density histogram of the given data, and compares it with a gaussian mixture
    with given parameters.

    Input:
        - data: histogram data to plot, of shape (n_features, n_plots)
        - means: means of the gaussian mixtures, of size (n_gaussians, n_plots)
        - variances: idem, for variances
        - proportions: vector of probability for the gaussian mixtures
        - filename: if given, saves plot to disk.

    Output: none, but shows generated plot.
    """

    from matplotlib import rc
    from matplotlib.ticker import FormatStrFormatter

    rc("text", usetex=True)
    rc("font", family="serif")

    if len(data.shape) == 1:
        data = data[:, None]
        r0 = 1
    else:
        r0 = data.shape[1]

    colors = ["lightpink", "skyblue", "lightpink"]
    edgecolors = ["hotpink", "steelblue"]

    fig, ax = plt.subplots(
        ncols=r0,
        sharey=True,
        gridspec_kw={"wspace": 0.1},
        figsize=(11, 2),
        squeeze=False,
    )

    for i in range(r0):
        ax[0, i].hist(
            data[:, i],
            bins=100,
            density=True,
            color=colors[i],
            histtype="stepfilled",
            alpha=1,
            ec=edgecolors[i],
        )
        ax[0, i].set_yticks([])
        x_min, x_max = ax[0, i].get_xlim()
        x = np.arange(x_min, x_max, 0.01)
        y_mat = (
            1
            / np.sqrt(2 * np.pi * variances[:, i])
            * np.exp(-np.square(x[:, np.newaxis] - means[:, i]) / (2 * variances[:, i]))
        )
        y = y_mat.dot(proportions)
        ax[0, i].plot(x, y, color="k", lw=2, alpha=0.7)
        ax[0, i].xaxis.set_major_formatter(FormatStrFormatter("%0.1f"))

    plt.savefig(filename, bbox_inches="tight")
    ax[0, 0].set_zorder(100)

    plt.show()


if __name__ == "__main__":

    F1 = np.array([[6, 4], [5, 3]])
    F2 = np.array([[48, 6], [12, 24]])
    p = np.array((1 / 3, 2 / 3))

    for filename, F in zip(["histograms_one", "histograms_two"], [F1, F2]):
        means, variances = gaussian_params(F, p)

        datafile = f"data/{filename}.txt"
        if path.exists(datafile):
            data = np.genfromtxt(datafile)
        else:
            n, n_tries = 5000, 20
            data = gen_data(F, p, n_nodes=n, n_tries=n_tries, datafile=datafile)

        plot_gaussianmix(data, means, variances, p, filename="pictures/{filename}.pdf")
