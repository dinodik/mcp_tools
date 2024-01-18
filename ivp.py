import numpy as np

# TODO Vectorise FwEulerN

def FwEuler(dydt, t0, y0, t_end, dt) -> ([float], [float]):
    """
    Compute `y(t)` given `dydt` and initial condition `y0` at `t0` using forward Euler stepping.

    Parameters
    ----------
    dydt : function
        expression for the derivative of `y` in the form dydt(t, y) -> float
    t0 : float
        initial time
    y0 : float
        initial value
    t_end : float
        end time
    dt : float
        time step

    Returns
    -------
    (T, Y) : ([float], [float])
        array of time values and array of y(t) values
    """
    T = np.arange(t0, t_end + dt, dt)
    Y = np.empty_like(T)
    Y[0] = y0
    for i in range(T.shape[0] - 1):
        Y[i+1] = Y[i] + dt * dydt(T[i], Y[i])
    return T, Y

def ODERK4(dydt, t0, y0, t_end, dt) -> ([float], [float]):
    """
    Compute `y(t)` given `dydt` and initial condition `y0` at `t0` using RK4.

    Parameters
    ----------
    dydt : function
        expression for the derivative of `y` in the form dydt(t, y) -> float
    t0 : float
        initial time
    y0 : float
        initial value
    t_end : float
        end time
    dt : float
        time step

    Returns
    -------
    (T, Y) : ([float], [float])
        array of time values and array of y(t) values
    """
    T = np.arange(t0, t_end + dt, dt)
    Y = np.empty_like(T)
    Y[0] = y0
    for i, t in enumerate(T[:-1]):
        y = Y[i]
        k1 = dt * dydt(t, y)
        k2 = dt * dydt(t + dt/2, y + k1/2)
        k3 = dt * dydt(t + dt/2, y + k2/2)
        k4 = dt * dydt(t + dt, y + k3)
        Y[i+1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    return T, Y

def _mapF(funcs, *arr) -> [float]:
    """
    Map array of functions to array of outputs by calling each function with the provided arguments in `*arr`.

    Parameters
    ----------
    funcs : [function]
        array of functions
    *arr
        arguments to be passed into functions, note that each function receives all arguments
    """
    return np.array([funcs[i](*arr) for i in range(len(funcs))])

def FwEulerN(dYdt, t0, Y0, t_end, dt) -> ([float], [[float]]):
    """
    Compute `Y(t)` given `N` derivative expressions `dYdt` and `N` initial conditions `Y0` at `t0` using forward Euler stepping.

    Parameters
    dYdt : [function]
        array of derivative expressions, one for each `y(t)` in the form dydt(t, y) -> float
    t0 : float
        initial time
    Y0 : [float]
        array of initial values, in the same order as `dYdt`
    t_end : float
        end time
    dt : float
        time step

    Returns
    -------
    (T, Y) : ([float], [[float]])
        array of time values and 2D array of y(t) values where each row corresponds to a y(t)
    """
    assert (N := len(dYdt)) == len(Y0)
    T = np.arange(t0, t_end + dt, dt)
    Y = np.empty((N, T.shape[0]))
    Y[:,0] = Y0
    for i in range(T.shape[0]-1):
        Y[:,i+1] = Y[:,i] + dt * _mapF(dYdt, T[i], *Y[:,i])
    return T, Y
