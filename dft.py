def freqDFT(N: int, dt: float):
    """
    Returns appropriate frequency bins for a FT of `N` sample points with sample interval `dt`.
    Example: 
    ```
    > freqDFT(8, 0.05)
    array([0., 2.5, 5., 7.5, -10., 7.5, 5., -2.5])
    ```
    The above sampling frequency is `1/0.05 = 20Hz`, so an appropriate frequency bin is from -10Hz to 10Hz. However, since DFT evaluates the FT from 0Hz to 20Hz (i.e., k in `[0,N)`), the frequency bin array has to rolled to begin at zero. Note that the two ranges of frequencies are identical as the FT is N-periodic, i.e., the transform at 12.5Hz is equal to the transform at -7.5Hz. Alternatively, you can imagine that frequencies above 10Hz (Nyquist frequency) are aliased and appear as negative 'ghost frequencies' that are slower. However, these 'ghost frequencies' are part of the whole transform and are necessary to losslessly reconstruct the signal using IDFT.
    """
    freqs = np.arange(-(N//2), N//2 + N%2) / (N * dt) # in Hz
    return np.roll(freqs, -(N//2)) # rotate so it matches up with DFT output, i.e. 0Hz at FT[0]

def DFT(Y):
    """
    Returns the discrete Fourier transform of a given signal `Y`. Axis-0 is the frequency in units rev/sample where the `k`-th index is the DFT evaluated at `k/N` rev/sample where `1/N` is the base frequency. *Note that the result is not normalised.*

    Parameters
    ----------
    Y : [float]
        Input signal
    """
    N = len(Y)
    n = np.arange(N)
    k = n.reshape((N, 1))
    mat = np.exp(-2j * np.pi * k * n / N) # k*n is a matrix
    return mat @ Y

def IDFT(FT):
    """
    Returns the signal that produces the given data `FT` in frequency space. *Note that the result is not normalised.*

    Parameters
    ----------
    FT : [float]
        Input data in frequency space
    """
    N = len(FT)
    n = np.arange(N)
    k = n.reshape((N, 1))
    mat = np.exp(2j * np.pi * k * n / N)
    return mat @ FT / N
