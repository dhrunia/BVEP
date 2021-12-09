from numba import jit
import numpy as np

@jit(nopython=True)
def logit(val):
    return np.math.log(val) - np.math.log(1.0 - val)

@jit(nopython=True)
def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-val))

@jit(nopython=True)
def inv_logit_bounded(val, lb, ub):
    return lb + (ub - lb) * sigmoid(val)

@jit(nopython=True)
def logit_bounded(val, lb, ub):
    return logit((val - lb)/(ub - lb))

# @jit(nopython=True)
def bound_norm(val, lb, ub):
    return (val - lb) / (ub - lb)

# @jit(nopython=True)
def inv_bound_norm(val, lb, ub):
    return lb + (ub - lb) * val