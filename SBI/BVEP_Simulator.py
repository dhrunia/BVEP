#!/usr/bin/env python3
"""
@author: meysamhashemi  INS Marseille

"""
import numpy as np
from numba import jit


def func_2DepileptorEqs(y, eta, K, I, SC):
    num_nodes = int(SC.shape[0])
    x = np.empty(num_nodes)
    z = np.empty(num_nodes)
    F = np.empty(2*num_nodes)
    for i in np.r_[0:num_nodes]:
        x[i] = y[2*i]
        z[i] = y[2*i+1]
    for i in np.r_[0:num_nodes]:
        gx = 0
        for j in np.r_[0:num_nodes]:
            gx = gx+SC[i, j]*(x[j]-x[i])
        F[2*i] = -x[i]**3-2*x[i]**2-z[i]+I
        F[2*i+1] = 4*(x[i]-eta[i])-z[i]-K*gx
    return F


@jit(nopython=True)
def epileptor2D_sde_fn(y, eta, tau, K, SC):
    nn = SC.shape[0]
    x = y[0:nn]
    z = y[nn:2*nn]
    I1 = 3.1
    gx = np.sum(K * SC * (np.expand_dims(x, axis=0) -
                np.expand_dims(x, axis=1)), axis=1)
    dx = 1.0 - np.power(x, 3) - 2 * np.power(x, 2) - z + I1
    dz = (1.0/tau)*(4*(x - eta) - z - gx)
    return np.concatenate((dx, dz)).astype(np.float32)


@jit(nopython=True)
def Integrator_Euler(y_init, nt, dt, sigma, eta, tau, K, SC):
    nn = SC.shape[0]
    y_out = np.zeros((2*nn, nt), dtype=np.float32)
    y_next = y_init
    for i in np.arange(nt):
        k1 = epileptor2D_sde_fn(y_next, eta, tau, K, SC)
        y_next = y_next + dt * k1 + np.sqrt(dt) * \
            sigma * np.random.randn(2*nn).astype(np.float32)
        y_out[:, i] = y_next.astype(np.float32)

    return y_out


@jit(nopython=True)
def VEP2Dmodel(params, constants, SC):

    nn = SC.shape[0]

    # parameters
    eta = params[0:nn]
    y_init = params[nn:3*nn]
    K = params[3*nn]
    tau = params[3*nn+1]

    # fixed parameters
    sigma = np.float32(constants[0])
    dt = np.float32(constants[1])
    nt = int(constants[2])

    y_euler = Integrator_Euler(y_init, nt, dt, sigma, eta, tau, K, SC)

    return y_euler[0:nn, :]


def coupling(X, K, SC):
    nn = X.shape[0]
    gx = np.dot(X[:, np.newaxis], np.ones((1, nn))) - \
        np.dot(np.ones((nn, 1)), X[:, np.newaxis].T)
    return K*np.sum(SC*(gx), axis=0).T


def VEP2D_forwardmodel(params, constants, init_conditions, dt, ts, SC, seed=None):

    nt = ts.shape[0]
    nn = SC.shape[0]

    # parameters
    eta = params[0:nn]
    K = params[-1]

    # fixed parameters
    tau,  sigma = constants[0], constants[1], constants[2]
    I1 = 3.1

    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    # simulation from initial point
    x = np.zeros((nn, nt))  # fast voltage
    z = np.zeros((nn, nt))  # slow voltage

    # initial conditions
    x_init, z_init = init_conditions[0], init_conditions[1]
    x[:, 0] = x_init
    z[:, 0] = z_init

    # integrate SDE
    for t in range(0, nt-1):
        dx = 1.0 - x[:, t]*x[:, t]*x[:, t] - 2.0*x[:, t]*x[:, t] - z[:, t] + I1
        dz = (1/tau)*(4*(x[:, t] - eta[:]) -
                      z[:, t] - coupling(x[:, t], K, SC.T))
        x[:, t+1] = x[:, t] + dt*dx + \
            np.sqrt(dt) * sigma * np.random.randn()
        z[:, t+1] = z[:, t] + dt*dz + \
            np.sqrt(dt) * sigma * np.random.randn()

    return np.array(x).reshape(-1)

