"""
    Module `thresholds.py`.
    Computing and plotting various quantities associated to convergence thresholds in
    the directed SBM.

    References:
    [1]: Submitted paper
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

rc("text", usetex=True)
rc("font", family="serif")


def compute_eigenvalues(eta, s, r):
    """
    Computes the eigenvalues for a pathwise DSBM with the given parameters.

    Input:
        - eta: array_like, asymmetry parameter
        - s: float, degree parameter
        - r: number of blocks

    Output:
        - array of shape (r, eta.size), such that result[i, j] is the i-th eigenvalue
        for the parameter eta[j].
    """
    theta = 2 * np.sqrt(eta * (1 - eta))
    k = np.arange(r)
    cosines = np.cos((1 + k) * np.pi / (r + 1))
    eigenvalues = 1 / 2 + theta * cosines[:, None]

    return s * eigenvalues


def compute_threshold(eta, r):
    """
    In a pathwise SBM with given asymmetry and number of blocks, computes the threshold
    in s such that all eigenvalues of the model are informative.

    Input:
        - eta: array_like, asymmetry parameter
        - r: number of blocks

    Output:
        - array of shape (eta.size,), such that result[i] is the threshold for the
        parameter eta[i].
    """
    eigenvalues = compute_eigenvalues(eta, 1, r)
    return eigenvalues[0] / np.amin(eigenvalues ** 2, axis=0)


def plot_thresholds(r_list, n_eta=202, filename=None):
    """
    Plots the thresholds returned by `compute_thresholds`, for given values of r and
    eta between 0.5 and 1.

    Input:
        - r_list: list of number of blocks to consider
        - n_eta: number of datapoints for plot. They are regularly spaced between 0.5
        and 1 (both inclusive).
        - filename: if given, saves figure to disk.

    Output: none, but shows generated plot. The plot consists of one subplot per value
    of r, arranged side by side.
    """
    eta = np.linspace(0.5, 1, n_eta)
    fig, ax = plt.subplots(ncols=len(r_list), figsize=(15, 3))

    for i, r in enumerate(r_list):
        ax[i].set_yscale("log")
        ax[i].set_title(r"$r = {}$".format(2 ** (i + 1)))
        ax[i].set_xticks([0.5, 1])

        threshold = compute_threshold(eta, 2 ** (i + 1))
        label = r"$\eta \mapsto s(\eta, r)$"
        ax[i].plot(
            eta,
            threshold,
            color="darkred",
            linewidth=1,
            label=label if i == 0 else None,
        )
        ax[i].fill_between(eta, threshold, 0, color="tomato")

    ax[0].legend(loc=0)
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.show()


def plot_abs_eigenvalues(s, r, n_eta=202, filename=None):
    """
    Plots the absolute value of the expected eigenvalues in a pathwise DSBM with given
    parameters, as well as the threshold for informative eigenvalues.

    Input:
        - s: degree parameter
        - r: number of blocks
        - n_eta: number of datapoints for plot. They are regularly spaced between 0.5
        and 1 (both inclusive).
        - filename: if given, saves figure to disk.

    Output: none, but shows the generated plot.
    """
    eta = np.linspace(0.5, 1, n_eta)
    abs_eig = np.abs(compute_eigenvalues(eta, s, r))
    limit = np.sqrt(abs_eig[0])

    fig, ax = plt.subplots()
    for i in range(r):
        ax.plot(
            eta,
            abs_eig[i],
            linewidth=0.9,
            color="0.5",
            label="modulus of eigenvalues" if i == 0 else None,
        )

    ax.plot(eta, limit, linewidth=2, color="k", label=r"$\sqrt{\nu_1}$")
    ax.legend(loc=0)
    ax.set_xlabel(r"$\eta$")

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.show()


def etamin(s):
    """
    Computes the value of eta above which both eigenvalues are informative in the
    2-block pathwise DSBM.

    Input:
        - s: array_like, degree parameter
    """
    x = 1 + 2 * (1 - np.sqrt(2 * s + 1)) / s
    return 1 / 2 + np.sqrt(1 - x ** 2) / 2


def plot_etamin(s_min, s_max, spec_s, n_s=202, filename=None):
    """
    Plots the function defined in `etamin`, along with a few special values.

    Input:
        - s_min, s_max: bound for x axis of plot
        - spec_s: special values of s to highlight
        - n_s: number of datapoints for plot. They are regularly spaced between s_min
        and s_max (both inclusive).
        - filename: if given, saves figure to disk.

    Output: none, but shows generated plot. The plot consists of the line plot of
    `etamin`, plus the special values of s and their images.
    """
    s_list = np.linspace(s_min, s_max, n_s)
    fig, ax = plt.subplots()

    # Plotting special values :
    spec_values = list(etamin(np.array(spec_s)))
    for (x, y) in zip(spec_s, spec_values):
        ax.plot((x, x), (0.8, y), linewidth=0.5, c="0.2")
        ax.plot((2, x), (y, y), linewidth=0.5, c="0.2")

    ax.plot(s_list, etamin(s_list), linewidth=2, color="k", label=r"$\eta(s)$")
    ax.legend(loc=0)
    ax.set_xlabel(r"degree $s$")
    ax.set_ylabel(r"$\eta$")
    ax.set_xticks(spec_s)
    ax.set_yticks(spec_values)

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)

    plt.show()


# ------------
# Actual picture generation
# ------------

if __name__ == "__main__":

    r_list = 2 ** np.arange(1, 6)
    plot_thresholds(r_list, filename="pictures/thresholds.pdf")

    plot_abs_eigenvalues(10, 30, filename="pictures/abs_eigenvalues.pdf")

    plot_etamin(2, 100, [10, 25, 50, 75, 100], filename="pictures/eta_s.pdf")
