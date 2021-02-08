from scipy.sparse.linalg import eigs, eigsh


def compute_eigenvalues(G, k=2, is_sym=True, return_eigenvectors=False):
    """
    Computes the k largest (in modulus) eigenvalues of a scipy sparse matrix.
    By default, doesn't return eigenvectors.
    """
    # eigs throws an error if k > n - 2
    k = min(k, G.shape[0] - 2)
    eig_function = eigsh if is_sym else eigs
    output = eig_function(G, k, which="LM", return_eigenvectors=return_eigenvectors)
    return output
