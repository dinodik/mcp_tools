import numpy as np

def _lagrangianPolynomial(j, x, xn) -> float:
    """
    Helper function for lagrInterp that returns the `j`th lagrangian polynomial evaluated at `x`.

    Parameters
    ----------
    j : int
        order of polynimial
    x : float
        point of evaluation
    xn : [float]
        array of known nodes
    """
    assert j < len(xn)
    prod = 1
    for k, xi in enumerate(xn):
        if k != j:
            prod *= (x - xi) / (xn[j] - xi)
    return prod

def lagrange(X, xn, yn) -> [float]:
    """
    Predicts values of f(`X`) given known points at `xn` and `yn` using Lagrangian interpolation.

    Parameters
    ----------
    X : [float]
        array of points to be interpolated
    xn : [float]
        array of x-values of known nodes
    yn : [float]
        array of y-values of known nodes
    """
    assert (N := len(xn)) == len(yn)
    interp = lambda x: sum([yn[j] * _lagrangianPolynomial(j, x, xn) for j in range(N)])
    return list(map(interp, X))

def _newtDivDiffFast(xn, yn, n) -> float:
    """
    Dynamic solution to computing the `n`th Newton divided difference given known nodes at `xn` and `yn`.

    Parameters
    ----------
    xn : [float]
        array of x-values of known nodes
    yn : [float]
        array of y-values of known nodes
    n : int
        order of divided difference
    """
    N = n + 1
    table = np.empty([N, N])
    table[:,0] = yn[:N]
    for i in range(1, N):
        for j in range(N - i):
            table[j, i] = (table[j+1, i-1] - table[j, i-1]) / (xn[j+i] - xn[j])
    return table[0][n]

def _product(arr) -> float:
    """
    Product of all elements in a list of floats.

    Parameters
    ----------
    arr : [float]
        array of floats to be multiplied together
    """
    prod = 1
    for n in arr:
        prod *= n
    return prod

def newton(X, xn, yn) -> [float]:
    """
    Predicts values of f(`X`) given known points at `xn` and `yn` using Newton interpolation.

    Parameters
    ----------
    X : [float]
        array of points to be interpolated
    xn : [float]
        array of x-values of known nodes
    yn : [float]
        array of y-values of known nodes
    """
    assert (N := len(xn)) == len(yn)
    interp = lambda x: yn[0] + sum([_newtDivDiffFast(xn, yn, i) * product([x - xn[j]for j in range(0, i)]) for i in range(1, N)])
    return list(map(interp, X))

def _splineCoeff(xn, yn, va, vb) -> [float]:
    """
    Helper function for `spline` that returns the cubic polynomials' coefficients. Gradient at boundaries are clamped with at `va` and `vb`.

    Parameters
    ----------
    xn : [float]
        array of x-values of known nodes
    yn : [float]
        array of y-values of known nodes
    va : float
        gradient at xn[0]
    vb : float
        gradient at xn[-1]
    """
    N = len(xn)
    h = [0] + [xn[i] - xn[i-1] for i in range(1, N)] ## steps
    D = [0] + [yn[i] - yn[i-1] for i in range(1, N)]
    A = np.zeros([N, N]) ## Av = B
    B = np.zeros([N])
    ## set clamped boundaries
    A[0][0] = 1
    B[0] = va
    A[-1][-1] = 1
    B[-1] = vb
    for i in range(1, N - 1):
        A[i][i-1]   = 1 / h[i]
        A[i][i]     = 2 * (1 / h[i] + 1 / h[i+1])
        A[i][i+1]   = 1 / h[i+1]
        B[i] = (3 * D[i+1]) / h[i+1]**2 + (3 * D[i]) / h[i]**2
    v = np.linalg.inv(A) @ B ## gradients

    coeff = np.empty([N-1, 4]) # a, b, c, d for cubic spline
    for i in range(N-1):
        coeff[i][0] = yn[i]
        coeff[i][1] = v[i]
        coeff[i][2] = (3 * D[i+1]) / h[i+1]**2 - (v[i+1] + 2 * v[i]) / h[i+1]
        coeff[i][3] = (v[i+1] + v[i]) / h[i+1]**2 - (2 * D[i+1]) / h[i+1]**3

    return coeff

def spline(X, xn, yn, va, vb) -> [float]:
    """
    Predicts values of f(`X`) given known points at `xn` and `yn` using cubic spline interpolation. Gradient at boundaries are clamped at `va` and `vb`.

    Parameters
    ----------
    X : [float]
        array of values to be interpolated
    xn : [float]
        array of x-values of known nodes
    yn : [float]
        array of y-values of known nodes
    va : float
        gradient at xn[0]
    vb : float
        gradient at xn[-1]
    """
    N = len(xn)
    coeff = _splineCoeff(xn, yn, va, vb)
    ys = np.empty_like(X)
    for i in range(N - 1):
        if i == 0: idx = np.where(X < xn[i+1])
        elif i == N - 2: idx = np.where(X >= xn[i])
        else: idx = np.where((X >= xn[i]) & (X < xn[i+1]))
        for j in idx[0]:
            prod = np.ones([4])
            for k in range(1, N + 1):
                prod[k:] *= (X[j] - xn[i])
            ys[j] = coeff[i] @ prod
    return ys
