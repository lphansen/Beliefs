"""
This module provides auxiliary functions to facilitate computations in other modules.

"""
import numpy as np
from numba import njit


@njit
def stationary_prob(P):
    """
    Computes stationary distribution.
    
    Parameters
    ----------
    P : (n, n) ndarray
        Transition probability matrix.
    
    Returns
    -------
    π : (n, ) ndarray
        Stationary distribution.

    """
    A = P.T - np.eye(P.shape[0])
    A[-1] = np.ones(P.shape[0])
    B = np.zeros(P.shape[0])
    B[-1] = 1.
    π = np.linalg.solve(A, B)
    return π


def chernoff_entropy(P, P_tilde, grid_size):
    """
    Measure the common decay rate of the Chernoff bounds on type 1 and type 2 errors.
    
    Parameters
    ----------
    P, P_tilde : (n, n) ndarray
        Probability transition matrices.
    grid_size : int
        Grid size of the [0,1] interval for searching optimal s.

    Returns
    -------
    decay_rate : float
        The common decay rate of the Chernoff bounds on type 1 and type 2 errors.
    optimal_s : float
        The optimizing s.

    """
    # Match coefficients
    R = P
    Q = P_tilde

    radius_list = []
    for i in range(grid_size+1):
        s = i/grid_size
        Hs_RQ = (R**s) * (Q**(1-s))
        radius = np.absolute(np.linalg.eig(Hs_RQ)[0]).max()
        radius_list.append(radius)

    decay_rate = 1-np.array(radius_list).min()
    optimal_s = np.argmin(radius_list)/grid_size

    return decay_rate, optimal_s


@njit
def construct_transition_matrix(f, g, z0, z1, ϵ, ξ, λ, e, P_z, P_z_tilde):
    """
    Construct empirical transition matrix.
    
    """
    n_X = f.shape[0]
    n_Z = z0.shape[1]
    Z_states = np.zeros(z0.shape[0])
    for i in range(z0.shape[0]):
        for j in range(z0.shape[1]):
            if z0[i,j] == True:
                Z_states[i] = j
    Z_states_next = np.zeros(z1.shape[0])
    for i in range(z1.shape[0]):
        for j in range(z1.shape[1]):
            if z1[i,j] == True:
                Z_states_next[i] = j
    Z_0_all = np.sum(Z_states == 0)
    Z_1_all = np.sum(Z_states == 1)
    Z_2_all = np.sum(Z_states == 2)
    Z_all = [Z_0_all, Z_1_all, Z_2_all]
    data = []
    for i in range(z0.shape[0]):
        temp = (Z_states[i], Z_states_next[i], i)
        data.append(temp)

    # Transition matrix, order: (original X, Z state 0), (original X, Z state 1), (original X, Z state 2)
    P       = np.zeros((n_X*n_Z, n_X*n_Z))
    P_tilde = np.zeros((n_X*n_Z, n_X*n_Z))

    for i in range(P.shape[0]):
        Z_state = int(i/n_X)
        X_state = i%n_X
        for j in range(P.shape[0]):
            Z_state_next = int(j/n_X)
            X_state_next = j%n_X
            if (Z_state, Z_state_next, X_state_next) in data:
                P[i, j] = 1. / Z_all[Z_state]
                N = 1./ϵ * np.exp(-g[X_state_next]/ξ\
                                  +f[X_state_next]@λ)\
                    * e[Z_state_next] / e[Z_state]
                P_tilde[i, j] = P[i, j]*N
    
    return P, P_tilde
    
    
    