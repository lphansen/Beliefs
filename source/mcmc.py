"""
This module implements the popular M-H sampler.

"""
import numpy as np
from numba import njit
from numba import prange


@njit(parallel=True)
def metropolis_hastings(π, x0, δ=1., T=1000, burn_in=1000, seed=1):
    r"""
    Implements M-H algorithm using a random walk chain. Assume
    :math:`$x^{(i)} = x^{(i-1)} + \epsilon \delta$` where $\epsilon$
    follows standard multivariate normal distribution and $\delta$ is a scalar.

    Parameters
    ----------
    π : callable
        (N, ) ndarray -> float
        The desired joint distribution that we draw samples from.
    x0 : (N, ) ndarray
        Initialization of x.
    δ : float
        Step size in the random walk.
    T : int
        Size of samples.
    burn_in : int
        Number of samples that we throw away.
    seed : int
        Random seed.

    Returns
    -------
    samples : (T, N) ndarray
        Samples drawn from the distribution π.

    """
    np.random.seed(seed)
    periods = T + burn_in
    samples = np.zeros((periods, x0.shape[0]))
    ϵs = np.zeros((periods, x0.shape[0]))
    for i in prange(x0.shape[0]):
        ϵs[:, i] = np.random.standard_normal(periods) * δ

    for i in prange(periods):
        x1 = x0 + ϵs[i]
        if np.random.rand() < π(x1) / π(x0):
            x0 = x1.copy()
        samples[i] = x0

    return samples[burn_in:]
