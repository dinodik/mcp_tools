from scipy.special import binom

def forward(X, Y, k, n):
    h = X[1] - X[0]
    return sum([(-1)**i * binom(k, i) * Y[n + k-i] for i in range(k+1)]) / h**k