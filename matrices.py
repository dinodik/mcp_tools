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
    """
    Helper function that checks if a given matrix is diagonally dominant.

    Parameters
    ----------
    M : [[float]]
        Matrix to be checked
    """
    m = np.abs(M)
    return np.all(np.diag(m) >= np.sum(m - np.diag(np.diag(m)), axis=1))

def jacobi(M: [[float]], b: [float], tol=1e-5) -> [[float]]:
    """
    Solves a system of equations `M @ x = b` iteratively with an relative crude error estimate of `tol`.
    
    Parameters
    ----------
    M : [[float]]
        Matrix defining the system of equation coefficients
    b : [float]
        Vector defining the RHS of the system of equations
    tol : float
        Minimum relative crude error between guesses before iterations finish

    Returns
    -------
    X : [float]
        Solution vector of the system
    """
    assert (N := M.shape[0]) == M.shape[1]
    assert _is_diag_dom(M)
    X_old = np.zeros(N) # first guess
    M_diag = np.diag(M) # vector of diagonal
    M_nodiag = M - np.diag(M_diag) # diagonal removed
    while True:
        X_new = (b - M_nodiag @ X_old) / M_diag
        if np.all(abs((X_new - X_old)/X_new) < tol): break
        X_old, X_new = X_new, X_old
    return X_new

def gaussSeidel(M, b, relax=1, tol=1e-5) -> [[float]]:
    """
    Solves a system of equations `M @ x = b` iteratively with an relative crude error estimate of `tol`. Different to Jacobi method by using new guesses immediately after they are calculated. However, this implementation is slow than the Jacobi one despite theory as Numpy vectorising becomes more difficult. This method also allows for a relaxation factor `relax`.
    
    Parameters
    ----------
    M : [[float]]
        Matrix defining the system of equation coefficients
    b : [float]
        Vector defining the RHS of the system of equations
    relax : float
        Relaxation factor to reduce iteration count when greater than 1. Max of 2
    tol : float
        Minimum relative crude error between guesses before iterations finish

    Returns
    -------
    X : [float]
        Solution vector of the system
    """
    assert (N := M.shape[0]) == M.shape[1]
    assert _is_diag_dom(M)
    X_guess = np.zeros(N) # first guess
    M_diag = np.diag(M) # vector of diagonal
    M_nodiag = M - np.diag(M_diag) # diagonal removed
    X_buffer = X_guess.copy() # only for crude error, not used in calculations
    while True:
        for i in range(N):
            X_guess[i] = (b[i] - M_nodiag[i] @ X_guess) / M_diag[i]
            X_guess[i] = relax * X_guess[i] + (1 - relax) * X_buffer[i]
        if np.all(abs((X_guess - X_buffer)/X_guess) < tol): break
        X_buffer = X_guess.copy()
    return X_guess
