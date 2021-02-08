import numpy as np
from matplotlib import pyplot as plt

from clustering import compute_score, kmeans_embedding, kmeans_on_vector_data
from generator import fast_sbm

N = 50
p = 5
n = 4000
eta_list = np.linspace(0.5, 1, num=N + 1)
eigen_scores = []
deg_scores = []

for eta in eta_list:
    print(eta)
    F = p * np.array([[1 / 2, eta], [1 - eta, 1 / 2]])
    eigen = 0
    deg = 0
    for _ in range(10):
        G = fast_sbm(n, F)
        eigen_embedding = kmeans_embedding(G, n_eigen=1, double_sided=True)
        deg_embedding = np.column_stack([G * np.ones(n), G.T * np.ones(n)])
        eigen_labels = kmeans_on_vector_data(eigen_embedding, n_clusters=2)
        deg_labels = kmeans_on_vector_data(deg_embedding, n_clusters=2)
        eigen += compute_score(eigen_labels)
        deg += compute_score(deg_labels)

    eigen_scores.append(eigen / 10)
    deg_scores.append(deg / 10)

plt.plot(eta_list, eigen_scores, color="red")
plt.plot(eta_list, deg_scores, color="blue")
plt.savefig("deg_clustering.png")
plt.show()
