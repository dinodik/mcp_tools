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


def matMinor(mat, row, col):
    rows, cols = len(mat), len(mat[0])
    return [[mat[i][j] for j in range(cols) if j != col] for i in range(rows) if i != row]

def determinant(mat):
    assert (N := len(mat)) == len(mat[0])
    if N == 2:
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    return sum([(-1) ** j * mat[0][j] * determinant(matMinor(mat, 0, j)) for j in range(N)])

def solveSystemByCramer(mat, p):
    assert (N := len(mat)) == len(mat[0]) and len(p[0]) == 1
    assert (det := determinant(mat)) != 0
    replaceColumn = lambda M, q, col: [[M[i][j] if j != col else q[i][0] for j in range(N)] for i in range(N)]
    return [[determinant(replaceColumn(mat, p, j)) / det] for j in range(N)]
