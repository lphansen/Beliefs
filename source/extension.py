"""
This module implements the extension of the solver, such as risk premia and
volatility calculation.

"""
import numpy as np
from source.solver import solve, find_ξ


def bound_ratio(find_ξ_args, g1, g2, ζ, lower=True, result_type=0):
    """
    This function computes the bound on ratios of two gs at a given ζ.
    
    Parameters
    ----------
    find_ξ_args : tuple
        Arguments (except for g in solver_args and min_RE) to be passed into find_ξ,
        including (solver_args, pct, initial_guess, interval, tol, max_iter).
        If pct == 0., then the function will use ξ = 100.
    g1 : (n,) ndarray
        The g1 in g = g1 - ζ*g2.
    g2 : (n,) ndarray
        The g2 in g = g1 - ζ*g2.
    ζ : float
        The ζ in g = g1 - ζ*g2.
    lower : bool
        If True, it will compute the lower bound of the ratio.
        If False, it will compute the upper bound of the ratio.
    result_type : int
        If 0, the function will compute the log difference of the two moments.
        If 1, the function will compute the ratio of the two moments.
        If 2, the function will subtract the log of the first moment by the second moment.

    Returns
    -------
    ratio : float
        Unconditional ratio of two moments. 
    ratio_cond : (n_states,) ndarray
        Conditional ratios of two moments.
    ratio_empirical : float
        Empirical unconditional ratio of two moments.
    ratio_cond_empirical : (n_states,) ndarray
        Empirical conditional ratios of two moments.

    """
    if lower:
        g = g1 - ζ*g2
    else:
        g = -(g1 - ζ*g2)
    # f, z0, z1, tol, max_iter
    f = find_ξ_args[0][0]
    z0 = find_ξ_args[0][1]
    z1 = find_ξ_args[0][2]
    n_states = z1.shape[1]
    solver_tol = find_ξ_args[0][3]
    solver_max_iter = find_ξ_args[0][4]
    solver_args = (f, g, z0, z1, solver_tol, solver_max_iter)
    result = solve(f, g, z0, z1, 100., solver_tol, solver_max_iter)
    if find_ξ_args[1] != 0:
        ξ = find_ξ(solver_args, result['RE'], find_ξ_args[1], find_ξ_args[2],
                   find_ξ_args[3], find_ξ_args[4], find_ξ_args[5])
        result = solve(f, g, z0, z1, ξ, solver_tol, solver_max_iter)

    # Calculate ratio, empirical
    # Term 1
    moment_cond_g1 = np.zeros(n_states)
    for state in range(n_states):
        moment_cond_g1[state] = np.mean(g1[z0[:, state]])
    moment_g1 = moment_cond_g1@result['π']

    # Term 2
    moment_cond_g2 = np.zeros(n_states)
    for state in range(n_states):
        moment_cond_g2[state] = np.mean(g2[z0[:, state]])
    moment_g2 = moment_cond_g2@result['π']

    # Calculate ratio, distorted
    # Term 1
    moment_bound_cond_g1 = np.zeros(n_states)
    for state in range(n_states):
        moment_bound_cond_g1[state] = np.mean(result['N'][z0[:, state]]*g1[z0[:, state]])
    moment_bound_g1 = moment_bound_cond_g1@result['π_tilde']

    # Term 2
    moment_bound_cond_g2 = np.zeros(n_states)
    for state in range(n_states):
        moment_bound_cond_g2[state] = np.mean(result['N'][z0[:, state]]*g2[z0[:, state]])
    moment_bound_g2 = moment_bound_cond_g2@result['π_tilde']

    # Combine term 1 and term 2
    if result_type == 0:
        ratio_empirical = np.log(moment_g1) - np.log(moment_g2)
        ratio_empirical_cond = np.log(moment_cond_g1) - np.log(moment_cond_g2)        
        ratio_bound = np.log(moment_bound_g1) - np.log(moment_bound_g2)
        ratio_bound_cond = np.log(moment_bound_cond_g1) - np.log(moment_bound_cond_g2)
    elif result_type == 1:
        ratio_empirical = moment_g1 / moment_g2
        ratio_empirical_cond = moment_cond_g1 / moment_cond_g2        
        ratio_bound = moment_bound_g1 / moment_bound_g2
        ratio_bound_cond = moment_bound_cond_g1 / moment_bound_cond_g2
    elif result_type == 2:
        ratio_empirical = np.log(moment_g1) - moment_g2
        ratio_empirical_cond = np.log(moment_cond_g1) - moment_cond_g2        
        ratio_bound = np.log(moment_bound_g1) - moment_bound_g2
        ratio_bound_cond = np.log(moment_bound_cond_g1) - moment_bound_cond_g2    
    else:
        raise ValueError('Invalid result_typle. It should be 0, 1 or 2.')

    return ratio_bound, ratio_bound_cond, ratio_empirical, ratio_empirical_cond
