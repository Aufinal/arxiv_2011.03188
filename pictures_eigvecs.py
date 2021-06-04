"""
    Module `picture_eigvecs.py`.
    Numerical validation for the eigenvector correlation formulas in the DSBM.

    References:
    [1]: Submitted paper
"""

from os import path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.sparse.linalg import eigs

from generator import fast_sbm

rc("text", usetex=True)
rc("font", family="serif")


def true_eigenvectors(eta, n_nodes):
    """
    Computes the theoretical eigenvectors of the path-DSBM with parameter eta. See [1]
    for details and computations.

    Input:
        - eta: array_like, parameter for the path-DSBM.
        - n_nodes: number of nodes in the graph

    Output:
        - array of shape (2, n_nodes, eta.size), such that result[i, :, j] contains the
        i-th eigenvector associated with eta[j].
    """

    if not isinstance(eta, np.ndarray):
        eta = np.array(eta)

    m = len(eta)
    eigv = np.zeros((2, n_nodes, m))
    half = int(n_nodes / 2)
    eigv[0, :half, :] = np.sqrt(eta)
    eigv[0, half:, :] = np.sqrt(1 - eta)
    eigv[1, :half, :] = np.sqrt(eta)
    eigv[1, half:, :] = -np.sqrt(1 - eta)

    result = np.sqrt(2 / n_nodes) * eigv
    return result


def theoretical_overlaps(eta, d=10):
    """
    Returns the theoretical scalar products between the informative eigenvectors of a
    path-DSBM random graph and its expectation matrix. See [1] for details.

    Input:
        - eta: array_like, asymmetry parameters for the DSBM
        - d: degree parameter

    Output:
        - array of shape (2, 2, eta.size) such that result[i, j, k] is the theoretical
        scalar product between the i-th eigenvector of the graph and the j-th one of its
        expected adjacency matrix, for asymmetry parameter eta[k].
    """

    if not isinstance(eta, np.ndarray):
        eta = np.array(eta)

    theta = 2 * np.sqrt(eta * (1 - eta))
    mu = d * (1 + np.array([1, -1])[:, None] * theta) / 4
    x = d / (mu * mu)
    denominator = 4 - x + theta * theta * x
    numerator = 4 - 2 * x + x * x * (1 - theta * theta) / 4
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=mu > np.sqrt(mu[0, :]),
    )
    b = np.array([[0, 1], [1, 0]])
    mult = np.eye(2)[:, :, None] + b[:, :, None] * (2 * eta - 1)
    return mult * np.sqrt(correlations[:, None, :])


def empirical_overlaps(n_nodes, eta_list, n_samples, d, filename=None):
    """
    Computes the scalar products between the eigenvectors a path-DSBM random graph and
    their expected values.

    Input:
        - n_nodes: number of nodes in the graph
        - eta_list: list of asymmetry parameters eta to consider.
        - n_samples: number of graphs generated for each value of eta
        - d: degree parameter for the path-DSBM
        - filename: if provided, the data will be saved to disk

    Output:
        - mean_overlap: array of shape (2, 2, n_eta) containing the mean of the
        `n_samples` scalar products between the expected eigenvectors and the empirical
        ones.
        - std_overlap: idem, but for the standard deviation.

    If saved to disk, the saved matrix will be of shape (2, 2, 2, n_eta), with data[0]
    containing the mean overlaps and data[1] the standard deviations.
    """
    n_eta = eta_list.size
    true_vecs = true_eigenvectors(eta_list, n_nodes)
    overlap = np.zeros((2, 2, n_eta, n_samples))

    for (i, eta) in enumerate(eta_list):
        print(f"Step {i+1}/{n_eta} --- eta = {eta:.3f}", end="\r")

        for sample in range(n_samples):
            F = d * np.array([[1 / 2, eta], [1 - eta, 1 / 2]])
            A, _ = fast_sbm(n_nodes, F)

            _, eigenvectors = eigs(A, k=2)

            overlap[:, :, i, sample] = np.abs(true_vecs[:, :, i].dot(eigenvectors).T)

    mean_overlap = np.mean(overlap, axis=3)
    std_overlap = np.std(overlap, axis=3)

    if filename is not None:
        data = np.stack((mean_overlap.ravel(), std_overlap.ravel()))
        np.savetxt(filename, data)

    return mean_overlap, std_overlap


def gen_plot(
    x,
    means,
    stds,
    theories,
    bg_colors=None,
    fg_colors=None,
    xlabel=None,
    legends=None,
    filename=None,
):
    """
    A somewhat generic function to plot the empirical mean/std of a quantity against its
    predicted theory. Multiple such plots can be superimposed on the same figure.

    Input:
        - x: array_like, common x axis
        - means: list of arrays, one for each plot
        - stds: idem
        - theories: idem
        - bg_colors: colors to plot theory and stdev ; one for each plot
        - fg_colors: colors to plot the mean quantity; one for each plot
        - xlabel: label for the x axis
        - legends: one legend per plot
        - filename: if provided, saves the picture to the disk

    Output: None, but shows the generated figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel(xlabel, fontsize=30)
    ax.tick_params(axis="both", labelsize=20)

    for (k, (mean, std, theory, bg, fg, legend)) in enumerate(
        zip(means, stds, theories, bg_colors, fg_colors, legends)
    ):
        errsup = mean + std
        errinf = mean - std

        ax.fill_between(x, errinf, errsup, color=bg, alpha=0.2)
        ax.plot(
            x,
            mean,
            "-",
            markersize=1,
            marker="o",
            color=fg,
            label=legend,
            zorder=10,
        )
        ax.legend(loc=0, fontsize=20)

        ax.plot(x, theory, color=bg, linewidth=5)

    # if filename is not None:
    #     plt.savefig(filename)

    plt.show()


# ------------
# Actual picture generation
# ------------

if __name__ == "__main__":

    n_eta = 50
    eta_list = np.linspace(0.5, 1, n_eta)
    d = 20
    datafile = "data/eigen_overlaps.txt"

    if path.exists(datafile):
        data = np.genfromtxt(datafile)
        mean_overlap = data[0].reshape((2, 2, n_eta))
        std_overlap = data[1].reshape((2, 2, n_eta))
    else:
        mean_overlap, std_overlap = empirical_overlaps(
            500, eta_list, 20, d, filename=datafile
        )

    theoretical_overlap = theoretical_overlaps(eta_list, d)
    legend = r"$|\langle u_{}, \varphi_{} \rangle |$"

    gen_plot(
        eta_list,
        mean_overlap[[0, 1], [0, 1], :],
        std_overlap[[0, 1], [0, 1], :],
        theoretical_overlap[[0, 1], [0, 1], :],
        xlabel=r"$\eta$",
        bg_colors=["tomato", "cornflowerblue"],
        fg_colors=["brown", "mediumblue"],
        legends=[legend.format(1, 1), legend.format(2, 2)],
        filename="pictures/eigenvectors_corr1.pdf",
    )

    gen_plot(
        eta_list,
        mean_overlap[[0, 1], [1, 0], :],
        std_overlap[[0, 1], [1, 0], :],
        theoretical_overlap[[0, 1], [1, 0], :],
        xlabel=r"$\eta$",
        bg_colors=["tomato", "cornflowerblue"],
        fg_colors=["brown", "mediumblue"],
        legends=[legend.format(1, 2), legend.format(2, 1)],
        filename="pictures/eigenvectors_corr1.pdf",
    )
