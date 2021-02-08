"""
Code for the generation of the histograms of the entries of the eigenvectors.

"""
from os.path import exists

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import eigs
from scipy.stats import norm

from generator import fast_sbm

n = 1000
n_tries = 5


def model_params(F, p):
    """
    Input:
        - F : connectivity matrix
        - p : proportion of vertices in each cluster

    Output:
        - r0: number of informative eigenvalues
        - means: theoretical means for gaussian mixture
        - variances : theoretical variances
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

    return r0, means, variances


def gen_data(F, p, r0=None, n=1000, n_tries=20, datafile=None):
    """
    Required:
        - F : connectivity matrix
        - p : proportion of vertices in each cluster

    Optional:
        - r0 : number of informative eigenvalues. Default is p.size, i.e. all eigenvalues are considered.
        - n : size of the generated graph. Default is 1000
        - n_tries : number of graphs generated. Default is 20.
        - datafile : filename to extract information from/save to.

    Output:
        - tot: (n * n_tries, r_0) matrix containing the aggregated eigenvectors.
        Each eigenvectors has norm sqrt(n).
    """

    if r0 is None:
        r0 = p.size

    if datafile is not None and exists(datafile):
        tot = np.genfromtxt(datafile)

        # if tot.shape != (n * n_tries, r0):
        # raise ValueError(f"Datafile does not contain a ({n * n_tries}, {r0}) matrix.")

    else:

        for i in range(n_tries):
            G = fast_sbm(n, F, left_proportions=p, right_proportions=p)
            if i == 0:
                actual_size = G.shape[0]
                tot = np.zeros((actual_size * n_tries, r0))
            vals, v = eigs(G, r0, which="LM", return_eigenvectors=True)
            for k in range(r0):
                if v[0, k] < 0:
                    v[:, k] = -v[:, k]
            tot[i * actual_size : (i + 1) * actual_size, :] = np.sqrt(n) * v.real

        if datafile is not None:
            np.savetxt(datafile, tot)

    return tot


F = np.array([[48, 6], [12, 24]])
p = np.array((1 / 3, 2 / 3))

r0, means, variances = model_params(F, p)
print(r0)
n, n_tries = 5000, 10

tot = gen_data(F, p, r0=r0, n=n, n_tries=n_tries, datafile="data/histograms_two.txt")


def plot_fluctuations(fname):

    from matplotlib import rc
    from matplotlib.ticker import FormatStrFormatter

    rc("text", usetex=True)
    rc("font", family="serif")

    data = np.loadtxt(fname)
    colors = ["lightpink", "skyblue", "lightpink"]
    edgecolors = ["hotpink", "steelblue"]
    fsize = 16
    histnames = [f"Entries of $u_{i+1}$" for i in range(r0)]

    fig, ax = plt.subplots(
        ncols=r0, sharey=True, gridspec_kw={"wspace": 0.1}, figsize=(11, 2)
    )

    for i in range(r0):
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        ax[i].hist(
            data[:, i],
            bins=100,
            density=True,
            color=colors[i],
            histtype="stepfilled",
            alpha=1,
            ec=edgecolors[i],
        )
        # ax[i].set_title(histnames[i], fontsize=fsize)
        ax[i].set_yticks([])
        x_min, x_max = ax[i].get_xlim()
        x = np.arange(x_min, x_max, 0.01)
        y_mat = (
            1
            / np.sqrt(2 * np.pi * variances[:, i])
            * np.exp(-np.square(x[:, np.newaxis] - means[:, i]) / (2 * variances[:, i]))
        )
        y = y_mat.dot(p)
        ax[i].plot(x, y, color="k", lw=2, alpha=0.7)
        ax[i].xaxis.set_major_formatter(FormatStrFormatter("%0.1f"))
    fsize2 = 16

    # ax[1].text(1,0,'$p_1\mathcal{N}(\mu_{2,1},\sigma_{2,1}^2)+p_2\mathcal{N}(\mu_{2,2},\sigma_{2,2}^2)$', fontsize=fsize2)

    # ax[0].text(2,0.5,'$p_1\mathcal{N}(\mu_{i,1},\sigma_{i,1}^2)+p_2\mathcal{N}(\mu_{i,2},\sigma_{i,2}^2)$', fontsize=fsize2)
    # ax[0].legend(loc=0, fontsize = fsize)
    plt.savefig("pictures/fluctuations_two.pdf", bbox_inches="tight")
    ax[0].set_zorder(100)

    plt.show()


plot_fluctuations("data/histograms_two.txt")


def plot_fluctuations_one(fname):

    from matplotlib import rc
    from matplotlib.ticker import FormatStrFormatter

    rc("text", usetex=True)
    rc("font", family="serif")

    data = np.loadtxt(fname)
    colors = ["lightpink", "skyblue", "lightpink"]
    edgecolors = ["hotpink", "steelblue"]
    fsize = 16
    histnames = [f"Entries of $u_{i+1}$" for i in range(r0)]

    fig, ax = plt.subplots(figsize=(11, 2))

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.hist(
        data,
        bins=100,
        density=True,
        color=colors[0],
        histtype="stepfilled",
        alpha=1,
        ec=edgecolors[0],
    )
    # ax.set_title(histnames[0], fontsize=fsize)
    ax.set_yticks([])
    x_min, x_max = ax.get_xlim()
    x = np.arange(x_min, x_max, 0.01)
    y_mat = (
        1
        / np.sqrt(2 * np.pi * variances[:, 0])
        * np.exp(-np.square(x[:, np.newaxis] - means[:, 0]) / (2 * variances[:, 0]))
    )
    y = y_mat.dot(p)
    ax.plot(x, y, color="k", lw=2, alpha=0.7)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%0.1f"))
    fsize2 = 16

    # ax[1].text(1,0,'$p_1\mathcal{N}(\mu_{2,1},\sigma_{2,1}^2)+p_2\mathcal{N}(\mu_{2,2},\sigma_{2,2}^2)$', fontsize=fsize2)

    # ax[0].text(2,0.5,'$p_1\mathcal{N}(\mu_{i,1},\sigma_{i,1}^2)+p_2\mathcal{N}(\mu_{i,2},\sigma_{i,2}^2)$', fontsize=fsize2)
    # ax[0].legend(loc=0, fontsize = fsize)
    plt.savefig("pictures/fluctuations_sparse_one.pdf", bbox_inches="tight")

    plt.show()


# plot_fluctuations_one("data/histograms_one.txt")
