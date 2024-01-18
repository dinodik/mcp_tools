import numpy as np

def trapezium(X, Y) -> float:
    """
    Computes the integral of curve given by `X` and `Y` using the trapezium rule.

    Parameters
    ----------
    X : [float]
        array of x-values of curve
    Y : [float]
        array of y-values of curve
    """
    assert (N := len(X)) == len(Y)
    I = 0
    for i in range(N - 1):
        dx = X[i+1] - X[i]
        I += dx * (Y[i+1] + Y[i]) / 2
    return I

def simpson(X, Y) -> float:
    """
    Computes the integral of curve given by `X` and `Y` using Simpson's 1/3 rule. Note that there must be an even number of intervals (i.e. odd number of given points).

    Parameters
    ----------
    X : [float]
        array of x-values of curve
    Y : [float]
        array of y-values of curve
    """
    assert len(X) % 2 == 1
    return (X[1] - X[0]) / 3 * (Y[0] + Y[-1] + 4*np.sum(Y[1:-1:2]) + 2*np.sum(Y[2:-2:2]))

def adaptiveSimpson(func, a, b, tol) -> (float, int):
    """
    Computes the integral of `func` from `a` to `b` to a tolerance of `tol` using Simpson's 1/3 rule and crude error approximation.

    Parameters
    ----------
    func : function
        function to be integrated in the form func(x) -> y
    a : float
        lower limit of integration
    b : float
        upper limit of integration
    tol : float
        required tolerance
    """
    M = 1
    I_m = np.inf
    while True:
        M *= 2
        X = np.linspace(a, b, M+1)
        I_2m = simpson(X, func(X))
        if abs((I_2m - I_m) / 15) < tol:
            break
        I_m = I_2m
    return I_2m, M + 1

_tg = [
    np.array([0]),
    np.array([1/np.sqrt(3), -1/np.sqrt(3)]),
    np.array([0, np.sqrt(3/5), -np.sqrt(3/5)]),
    np.array([np.sqrt(3/7-2/7*np.sqrt(6/5)), -np.sqrt(3/7-2/7*np.sqrt(6/5)), np.sqrt(3/7+2/7*np.sqrt(6/5)), -np.sqrt(3/7+2/7*np.sqrt(6/5))]),
    np.array([0, 1/3*np.sqrt(5-2*np.sqrt(10/7)), -1/3*np.sqrt(5-2*np.sqrt(10/7)), 1/3*np.sqrt(5+2*np.sqrt(10/7)), -1/3*np.sqrt(5+2*np.sqrt(10/7))])
]

_wg = [
    np.array([2]),
    np.array([1, 1]),
    np.array([8/9, 5/9, 5/9]),
    np.array([(18+np.sqrt(30))/36, (18+np.sqrt(30))/36, (18-np.sqrt(30))/36, (18-np.sqrt(30))/36]),
    np.array([128/225, (322+13*np.sqrt(70))/900, (322+13*np.sqrt(70))/900, (322-13*np.sqrt(70))/900, (322-13*np.sqrt(70))/900])
]

def quadrature(func, a, b, n) -> float:
    """
    Computes the integral of curve given by `X` and `Y` using Gauss quadratures. Note the degree of precision is given as `2n-1` and is limited to 9 (i.e. `n=5`).

    Parameters
    ----------
    func : function
        function to be integrated in the form func(x) -> y
    a : float
        lower limit of integration
    b : float
        upper limit of integration
    n : int
        number of quadrature nodes
    """
    t = (a*(1-_tg[n-1]) + b*(_tg[n-1]+1)) / 2
    return (b - a) / 2 * np.dot(func(t), _wg[n-1])
