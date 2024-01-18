import numpy as np

# TODO: shooting method?

def finiteDiff(ODE, a, b, BC_a, BC_b, BC_C, N) -> ([float], [float]):
    """
    Compute `y(x)` given a 2nd order ODE and boundary conditions at `x = a, b`. The form of the ODE is of `y'' + f(x)y' + g(x)y = p(x)`. The nature of the boundaries are given by array `C` and values `BC_a` and `BC_b`.

    Parameters
    ----------
    ODE : function
        given an array of `x` values, return the coefficients that define the ODE defined by `f(x)`, `g(x)`, and `p(x)`. Takes the form ODE(X: [float]) -> (F: [float], G: [float], P: [float]).
        Example for `y'' + 2xy' + 2 = cos(3x)`:
            `ODE = lambda X: (2*X, 2*np.ones_like(X), np.cos(3X))`
    a : float
        left boundary position of BVP, `x = a`
    b : float
        right boundary position of BVP, `x = b`
    BC_a : float
        left boundary value of BVP, see C for application
    BC_b : float
        right boundary value of BVP, see C for application
    BC_C : [float]
        an array of 4 coefficients that define the following system:
            `(x = a): BC_C[0] * y'(a) + BC_C[1] * y(a) = BC_a`
            `(x = b): BC_C[2] * y'(b) + BC_C[3] * y(b) = BC_b`
        to define the boundary conditions using linear combinations of y' and y.
        Example for `y(a) = 0; y'(b) = 2y(b) - 1`:
            `BC_C = [0, 1, 1, -2]`
            `BC_a = 0`
            `BC_b = -1`
    N : int
        number of subintervals between `x = a` and `x = b` for finite difference

    Returns
    -------
    X : [float]
        equally spaced `N + 1` x-value nodes between `x = a` and `x = b`
    Y : [float]
        solution to ODE in the form of `N + 1` y-value nodes corresponding to `X`
    """
    X = np.linspace(a, b, N+1)
    h = (b - a) / N
    F, G, P_int = ODE(X[1:-1]) # P_int for internal p(x) values
    ## coefficients for finite difference solver at internal nodes
    A = 1/h**2 - F/(2*h)
    B = G - 2/h**2
    C = 1/h**2 + F/(2*h)

    ## mat @ Y = P, initialise mat and P
    mat = np.zeros((N+1, N+1))
    P = np.zeros(N+1)

    ## internal
    for i, coeffs in enumerate(zip(A, B, C)):
        mat[i+1, i:i+3] = np.array(coeffs)
    P[1:-1] = P_int

    ## boundary using forward and backward difference
    mat[0, 0] = BC_C[1] - BC_C[0]/h
    mat[0, 1] = BC_C[0]/h
    mat[-1, -2] = -BC_C[2]/h
    mat[-1, -1] = BC_C[2]/h + BC_C[3]
    P[0]  = BC_a
    P[-1] = BC_b

    Y = np.linalg.inv(mat) @ P
    return X, Y
