#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

from clustering import clustering_score
from generator import create_connectivity_matrix, fast_sbm
from matrices import hermitian_adjacency, hermitian_product, skew_adjacency

n = 1000
n_samples = 10
n_clusters = 2

p = 10  # mean apriori degree
eta = 0.9
etal = np.linspace(0.5, 1, 50)

scores_adj = []
scores_double = []
scores_herm = []
scores_skew = []
scores_hermsum = []

for e in etal:

    print("Trials for eta = {}".format(e))
    F, eigvals_of_F = create_connectivity_matrix(p / 2, p * e, p * (1 - e), p / 2)
    score_local_adj = []
    score_local_double = []
    score_local_skew = []
    score_local_herm = []
    score_local_hermsum = []

    for t in range(n_samples):
        print("--------------- sample {}".format(t))
        A = fast_sbm(n, F=F)

        score_local_adj.append(clustering_score(A, n_clusters=n_clusters))
        score_local_double.append(
            clustering_score(A, n_clusters=n_clusters, double_sided=True)
        )
        score_local_skew.append(
            clustering_score(A, preprocessing=skew_adjacency, n_clusters=n_clusters)
        )
        score_local_herm.append(
            clustering_score(
                A, preprocessing=hermitian_adjacency, n_clusters=n_clusters
            )
        )
        score_local_hermsum.append(
            clustering_score(A, preprocessing=hermitian_product, n_clusters=n_clusters)
        )

    scores_adj.append(score_local_adj)
    scores_double.append(score_local_double)
    scores_herm.append(score_local_herm)
    scores_hermsum.append(score_local_hermsum)
    scores_skew.append(score_local_skew)

namelist = ["adj", "double sided", "skew", "herm", "hermsum"]
scorelist = [scores_adj, scores_double, scores_skew, scores_herm, scores_hermsum]
meanlist = [0 for _ in range(len(scorelist))]


fig, ax = plt.subplots(figsize=(10, 10))

for i in range(len(scorelist)):
    x = np.array(scorelist[i])
    np.savetxt("data/" + namelist[i] + "scores_{}.txt".format(n), x)
    meanlist[i] = np.mean(x, axis=1)
    ax.plot(etal, meanlist[i], label=namelist[i])

ax.legend(loc=2)
ax.set_title("Adjusted Rand Index")
plt.savefig("ARI_{}.png".format(n))
plt.show()
