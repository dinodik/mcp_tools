import numpy as np

def decompose(M) -> ([[float]], [[float]]):
    """
    Decomposes the input matrix M by Helmholtz's theroem and returns the irrotational, and incompressible parts as a tuple.

    Parameters
    ----------
    M : [[float]]
        Matrix to be decomposed
    """
    M_defo = 0.5 * (M + M.T)
    M_spin = 0.5 * (M + M.T)
    return M_defo, M_spin

def _is_diag_dom(M) -> bool:
    m = np.abs(M)
    return np.all(np.diag(m) >= np.sum(m - np.diag(np.diag(m)), axis=1))

def jacobi(M: [[float]], b: [float], tol=1e-5) -> [[float]]:
    assert (N := M.shape[0]) == M.shape[1]
    assert _is_diag_dom(M)
    # X_old = np.random.rand(N) * np.max(M) # random first guess
    X_old = np.zeros(N)
    M_diag = np.diag(M)
    M_nodiag = M - np.diag(M_diag)
    while True:
        X_new = (b - M_nodiag @ X_old) / M_diag
        if np.all(abs(X_new - X_old) < tol): break
        X_old, X_new = X_new, X_old
    return X_new

def gaussSeidel(M, b, tol=1e-5) -> [[float]]:
    assert (N := M.shape[0]) == M.shape[1]
    assert _is_diag_dom(M)
    # X_guess = np.random.rand(N) * np.max(M) # random first guess
    X_guess = np.zeros(N)
    M_diag = np.diag(M)
    M_nodiag = M - np.diag(M_diag)
    X_buffer = X_guess.copy()
    while True:
        for i in range(N):
            X_guess[i] = (b[i] - M_nodiag[i] @ X_guess) / M_diag[i]
        if np.all(abs(X_guess - X_buffer) < tol): break
        X_buffer = X_guess.copy()
    return X_guess
