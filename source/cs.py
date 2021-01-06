"""
This module implements the popular M-H sampler.

"""
import numpy as np
from numba import njit
from numba import prange


@njit
def mh_sampler(ϕ, α0, W_inv, args, step=1., T=1000, seed=1):
    r"""
    Implements M-H algorithm using a random walk chain. Assume
    :math:`$\theta^{(i)} = \theta^{(i-1)} + \epsilon \delta$` where $\epsilon$
    follows standard multivariate normal distribution and $\delta$ is a scalar.
    
    The target posterior distribution is proportional to
    :math:`\exp(\phi n L_n(\theta))\Pi(\theta)`. Additionally, we assume the
    prior :math:`\Pi` is flat.

    Parameters
    ----------
    ϕ : float
        Parameter in posterior distribution.
    α0 : (num_var, ) ndarray
        Initialization of α.
    W_inv : (num_var, num_var) ndarray
        Inverse of optimal weight matrix.
    args : tuple
        Model parameters and data, including ξ, f, g, z1, z0, z0_float.
    step : float
        Step size in the random walk.
    T : int
        Size of samples.
    seed : int
        Random seed.

    Returns
    -------
    samples : (T, num_var) ndarray
        Samples drawn from the posterior distribution.

    """
    np.random.seed(seed)
    samples = np.zeros((T, α0.shape[0]))
    ϵs = np.zeros((T, α0.shape[0]))
    accepted_num = 0
    for i in prange(α0.shape[0]):
        ϵs[:, i] = np.random.standard_normal(T) * step

    for t in prange(T):
        α1 = α0 + ϵs[t]
        L1 = L(α1, W_inv, args)
        L0 = L(α0, W_inv, args)
        RHS = np.exp((L1-L0)*args[1].shape[0]*ϕ)
        if np.random.rand() < RHS:
            accepted_num += 1
            α0 = α1.copy()
        samples[t] = α0
    
    accept_rate = accepted_num / T
    return samples, accept_rate


@njit(parallel=True)
def draw_flat_prior(bounds, n, seed=1):
    """
    Draws samples from flat prior.
    
    Parameters
    ----------
    bounds : tuple of tuple of floats
        Lower and upper bounds for each parameters.
    n : int
        Numebr of samples we want to draw.

    Returns
    -------
    res : (n, num_var) ndarray
        Samples.

    """
    np.random.seed(seed)
    num_var = len(bounds)
    res = np.zeros((n, num_var))
    for i in prange(n):
        for j in prange(num_var):
            temp = bounds[j]
            res[i, j] = np.random.uniform(temp[0], temp[1])
    return res


@njit(parallel=True)
def L(α, W_inv, args):
    """
    Criterion function using estimated optimal weight matrix.
    We transform conditional moments into unconditional moments first.

    Parameters
    ----------
    α : (n_α, ) ndarray
        Model parameters. n_α = n_f*n_states + n_states
    W_inv : (n_α, n_α) ndarray
        Inverse of estimated optimal weight matrix.
    args : tuple
        Model paraetmers and data, including ξ, f, g, z1, z0, z0_float, which are
        float, (T, n_f) ndarray, (T, ) ndarray, (T, n_states) ndarray,
        (T, n_states) ndarray, (T, n_states) ndarray respectively.

    Returns
    -------
    res : float
        Criterion function evaluated at α, given observed data and fixed ξ, θ.

    """
    ξ, f, g, z1, z0, z0_float = args
    T, n_f = f.shape
    n_states = z1.shape[1]
    μ = α[0]
    λ = α[1: 1+n_f*n_states]
    v = np.hstack((np.zeros(1), α[1+n_f*n_states:]))
    λ = λ.reshape(n_states, -1)
    kron_prod = np.zeros((T, n_states * (n_f + 1)))
    for t in prange(T):
        kron_prod[t] = np.kron(z0_float[t], 
                               ρ(f[t].copy(), λ[z0[t]][0], g[t], v[z1[t]], v[z0[t]], μ, ξ))
    mean = np.sum(kron_prod, axis=0).reshape(kron_prod.shape[1], -1) / T
    res = -0.5*mean.T@W_inv@mean
    return res[0, 0]


@njit
def estimate_optimal_W_inv(α, args):
    """
    Estimate optimal weight matrix.
    
    """
    ξ, f, g, z1, z0, z0_float = args
    T, n_f = f.shape
    n_states = z1.shape[1]
    μ = α[0]
    λ = α[1: 1+n_f*n_states]
    v = np.hstack((np.zeros(1), α[1+n_f*n_states:]))
    λ = λ.reshape(n_states, -1)
    kron_prod = np.zeros((T, n_states * (n_f + 1)))
    for t in prange(T):
        kron_prod[t] = np.kron(z0_float[t], 
                               ρ(f[t].copy(), λ[z0[t]][0], g[t], v[z1[t]], v[z0[t]], μ, ξ))
    mean = np.sum(kron_prod, axis=0).reshape(kron_prod.shape[1], -1) / T
    W_hat = (1./T*kron_prod.T@kron_prod) - mean@mean.T
    W_inv = np.linalg.inv(W_hat)
    return W_inv


@njit
def ρ(f, λ, g, v1, v0, μ, ξ):
    """
    Functions that define conditional moment condition.

    Prameters
    ---------
    f, λ : (n_f, ) ndarrays
        f(X1) and λ(Z0)
    g, v1, v0 : floats
        g(X1), v(Z1) and v(Z0)
    μ, ξ : floats
        Model parameters.

    Returns
    -------
    res : (n_f+1, ) ndarray
        Observed ρ.

    """
    temp = np.exp(-(g+v1-v0)/ξ + λ@f)
    res_1 = f * temp
    res_2 = temp - np.exp(-μ/ξ)
    res = np.append(res_1, res_2)
    return res
