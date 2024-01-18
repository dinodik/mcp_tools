from scipy.special import binom

def forward(X, Y, k, n):
    """
    Computes the `kth` derivative of curve given by `X` and `Y` at `X[n]` using forward differences.

    Parameters
    ----------
    X : [float]
        array of x-values of curve
    Y : [float]
        array of y-values of curve
    k : int
        order of derivative
    n : int
        index of `X` where derivative is evaluated
    """
    h = X[1] - X[0]
    return sum([(-1)**i * binom(k, i) * Y[n + k-i] for i in range(k+1)]) / h**k
