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
