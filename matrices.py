import numpy as np
from scipy.sparse import coo_matrix, diags


def skew_adjacency(A):
    """A -> i(A - A*)"""
    data, row, col = A.data, A.row, A.col
    new_data = np.concatenate([data, -data], axis=0)

    new_row = np.concatenate([row, col], axis=0)
    new_col = np.concatenate([col, row], axis=0)
    return 1j * coo_matrix((new_data, (new_row, new_col)))


def skew_laplacian(A, rank=4):
    """method in the paper by Laenen and Sun
    returns I - D^{-1/2}AD^{-1/2} where A is the hermitian adjacency with r-th root of unity"""

    N = A.shape[0]
    dplus = np.array(
        [1 / np.sqrt(x) if x != 0 else 0 for x in A.dot(np.ones(N))]
    )  # degrees
    dmoins = np.array(
        [1 / np.sqrt(x) if x != 0 else 0 for x in (A.T).dot(np.ones(N))]
    )  # degrees
    d = dplus + dmoins
    D = diags(d, shape=(N, N))
    id = diags(np.ones(N), shape=(N, N))

    l = np.ceil(2 * np.pi * rank)
    omega = np.exp(2 * 1j * np.pi / l)

    data, row, col = A.data, A.row, A.col
    new_data = np.concatenate([omega * data, np.conj(omega) * data], axis=0)
    new_row = np.concatenate([row, col], axis=0)
    new_col = np.concatenate([col, row], axis=0)

    H = coo_matrix((new_data, (new_row, new_col)), shape=(N, N))
    return id - D @ H @ D


def hermitian_adjacency(A):
    """A -> AA*"""
    B = A.transpose()
    return A @ B  # all hail Python 3


def hermitian_product(A):
    """A -> AA* + A*A"""
    B = A.transpose()
    return A @ B + B @ A


# def test():
#     from generator import fast_sbm
#     A = fast_sbm(100)
#     print(hermitian_adjacency(A))
#     print(hermitian_product(A))
#     print(skew_adjacency(A))
