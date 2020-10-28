"""
This module provides the solution to the minimization problem in the notebook.

"""
import numpy as np
from scipy.optimize import minimize
from numba import njit
from source.utilities import stationary_prob


def solve(f, g, z0, z1, ξ, quadratic=False, tol=1e-9, max_iter=1000):
    """
    This function implements the iteration scheme in the Computational
    Strategy section in the Belief_Notebook. It assumes a n-state Markov
    process for Z.

    Parameters
    ----------
    f : (n, n_f) ndarray
        f that satisfies E[N_1f(X_1)|Z_0] = 0.
    g : (n, 1) ndarray
        The function to be bounded.
    z0 : (n, n_states) ndarray
        Today's state vector.
        For example, (1,0,0) is state 1, (0,1,0) is state 2.
    z1 : (n, n_states) ndarray
        Tomorrow's state vector.
        For example, (1,0,0) is state 1, (0,1,0) is state 2.
    ξ : float
        Coefficient on relative entropy constraint.
    n_states : int
        Number of states.
    quadratic : bool
        If True, use quadratic divergence;
        If False, use relative entropy.
    tol : float
        Tolerance parameter for the iteration.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    res : OptimizeResult
        Class that stores the results of the optimization.

    """
    n_states = z0.shape[1]
    error = 1.
    count = 0
    n_fos = int(f.shape[1]/n_states) # Number of constraints on each state

    if quadratic:
        v = np.zeros(n_states)
        v_μ = np.zeros(n_states)  # v+μ
        λ_1 = np.zeros(f.shape[1])
        λ_2 = np.zeros(n_states)        
        while error > tol and count < max_iter:
            for state in range(n_states):
                v_μ[state], λ_1[state*n_fos: (state+1)*n_fos], λ_2[state]\
                = _minimize_objective_quadratic(f, g, z0, z1, state, n_fos, v, ξ, tol, max_iter)
            v_old = v
            μ = v_μ[0]
            v = v_μ - v_μ[0]
            error = np.max(np.abs(v - v_old))
            count += 1
        N = -1./ξ*(g+z1@v+f@λ_1+z0@λ_2) + 0.5
        N[N<0]=0
    else:
        v = np.zeros(n_states)
        e = np.ones(n_states)
        λ = np.zeros(f.shape[1])        
        while error > tol and count < max_iter:
            for state in range(n_states):
                v[state], λ[state*n_fos: (state+1)*n_fos]\
                = _minimize_objective(f, g, z0, z1, state, n_fos, e, ξ, tol, max_iter)
            e_old = e
            ϵ = v[0]
            e = v/v[0]
            error = np.max(np.abs(e - e_old))
            count += 1
        N = 1./ϵ * np.exp(-g/ξ+f@λ) * (z1@e) / (z0@e)
        v = - ξ * np.log(e)
        μ = - ξ * np.log(ϵ)        

    if count == max_iter:
        print('Warning: maximal iterations reached.')

    # Empirical transition matrix and stationary distribution
    P = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            P[i, j] = np.mean(z1[z0[:, i]][:, j])
    π = stationary_prob(P)

    # Distorted transition matrix and stationary distribution
    P_tilde = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            P_tilde[i, j] = np.mean(N[z0[:, i]] * z1[z0[:, i]][:, j])
    π_tilde = stationary_prob(P_tilde)

    # Conditional and unconditional relative entropy
    RE_cond = np.zeros(n_states)
    for state in range(n_states):
        RE_cond[state] = np.mean(N[z0[:, state]] * np.log(N[z0[:, state]]))
    RE = RE_cond @ π_tilde

    # Conditional and unconditional moment bounds for g
    moment_bound_cond = np.zeros(n_states)
    for state in range(n_states):
        moment_bound_cond[state] = np.mean(N[z0[:, state]]* g[z0[:, state]])
    moment_bound = moment_bound_cond @ π_tilde

    # Conditional and unconditional empirical moment for g
    moment_empirical_cond = np.zeros(n_states)
    for state in range(n_states):
        moment_empirical_cond[state] = np.mean(g[z0[:, state]])
    moment_empirical = np.mean(g)

    res = OptimizeResult({'ϵ':ϵ,
                          'λ':λ,
                          'count':count,
                          'ξ':ξ,
                          'v':v,
                          'RE_cond':RE_cond,
                          'RE':RE,
                          'P':P,
                          'π':π,
                          'P_tilde':P_tilde,
                          'π_tilde':π_tilde,
                          'moment_bound':moment_bound,
                          'moment_bound_cond':moment_bound_cond,
                          'moment_empirical_cond':moment_empirical_cond,
                          'moment_empirical':moment_empirical,
                          'N':N})

    if quadratic:
        div_cond = np.zeros(n_states)
        for state in range(n_states):
            div_cond[state] = np.mean(N[z0[:, state]]**2 - N[z0[:, state]]) * 0.5
        div = div_cond @ π_tilde
        res['QD'] = div
        res['QD_cond'] = div_cond
    else:
        res['e'] = e
        res['μ'] = μ

    return res


def find_ξ(solver_args, min_div, pct, initial_guess=1., interval=(0, 100.), tol=1e-5, max_iter=100):
    """
    This function finds the ξ that leads to x% increase to the
    relative entropy or quadratic divergence compared to the minimum.

    Parameters
    ----------
    solver_args : tuple
        Arguments (except for ξ) to be passed into the solver, including
        (f, g, z0, z1, quadratic, tol, max_iter)
    min_div : float
        Minimum divergence.
    pct : float
        Percent increase to the relative entropy.
        i.e. pct=0.2 means 20% increase to the entropy.
    initial_guess : float
        Initial guess for ξ.
    interval : tuple of ints
        Interval of ξ that we search over.
    tol : float
        Tolerance parameter on ξ.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    ξ : float
        The ξ that corresponds to x% increase to the entropy.
    
    """
    error = -1.
    count = 0
    ξ = initial_guess
    lower_bound, upper_bound = interval
    f, g, z0, z1, quadratic, solver_tol, solver_max_iter = solver_args
    while np.abs(error) > tol and count < max_iter:
        result = solve(f, g, z0, z1, ξ, quadratic, solver_tol, solver_max_iter)
        if quadratic:
            div = result['QD']
        else:
            div = result['RE']
        error = div/min_div - pct - 1
        if np.abs(error) < tol or lower_bound == upper_bound:
            break
        if error < 0.:
            upper_bound = ξ
            ξ = (lower_bound + ξ)/2.
        else:
            lower_bound = ξ
            ξ = (ξ + upper_bound)/2.
        count += 1
    if count == max_iter:
        print('Warning: maximal iterations reached. Error = %s' % (RE/min_RE - pct - 1))
    if lower_bound == upper_bound:
        print('Warning: lower bound is equal to upper bound. Please reset tolerance level.')
    return ξ


@njit
def _objective(λ, f, g, z0, z1_float, state, n_fos, e, ξ):
    """
    The objective function.
    
    """
    selector = z0[:, state]
    term_1 = -g[selector]/ξ
    term_2 = f[:, state*n_fos: (state+1)*n_fos][selector]@λ
    term_3 = np.log(z1_float[selector]@e)
    x = term_1 + term_2 + term_3
    # Use "max trick" to improve accuracy
    a = x.max()
    return np.log(np.mean(np.exp(x-a))) + a    


@njit
def _objective_gradient(λ, f, g, z0, z1_float,
                        state, n_fos, e, ξ):
    """
    Gradient of the objective function.
    
    """
    selector = z0[:,state]
    temp1 = -g[selector]/ξ + f[:,state*n_fos:(state+1)*n_fos][selector]@λ + np.log(z1_float[selector]@e)
    temp2 = f[:,state*n_fos:(state+1)*n_fos][selector]*(np.exp(temp1.reshape((len(temp1),1)))/np.mean(np.exp(temp1)))
    temp3 = np.empty(temp2.shape[1])
    for i in range(temp2.shape[1]):
        temp3[i] = np.mean(temp2[:,i])
    return temp3         


@njit
def _objective_quadratic(λ, f, g, z0, z1_float, state, n_fos, e, ξ):
    """
    The objective function with quadratic specification of divergence.
    
    """
    λ_1 = λ[:-1]
    λ_2 = λ[-1]
    selector = z0[:, state]
    term_1 = g[selector]
    term_2 = z1[selector]@v
    term_3 = f[:, state*n_fos:(state+1)*n_fos][selector]@λ_1
    term_4 = λ_2
    x = (term_1+term_2+term_3+term_4)/(-ξ) + 0.5
    x[x<0] = 0
    result = np.mean(x**2)*(ξ/2.) + λ_2
    return result
    

def _minimize_objective(f, g, z0, z1, state, n_fos, e, ξ, tol, max_iter):
    """
    Use scipy.minimize (L-BFGS-B, BFGS or CG) to solve the minimization problem.

    """
    z1_float = z1.astype(float)
    
    for method in ['L-BFGS-B','BFGS','CG']:
        model = minimize(_objective, 
                         np.ones(n_fos),
                         args = (f, g, z0, z1_float, state, n_fos, e, ξ),
                         method = method,
                         jac = _objective_gradient,
                         tol = tol,
                         options = {'maxiter': max_iter})
        if model.success:
            break
    if model.success == False:
        print("---Warning: the convex solver fails when ξ = %s, tolerance = %s--- " % (ξ, tol))
        print(model.message)

    v = np.exp(model.fun)
    λ = model.x
    return v, λ


def _minimize_objective_quadratic(f, g, z0, z1, state, n_fos, v, ξ, tol, max_iter):
    """
    Use scipy.minimize (L-BFGS-B, BFGS or CG) to solve the minimization problem.

    """
    z1_float = z1.astype(float)
    
    for method in ['L-BFGS-B','BFGS','CG']:
        model = minimize(_objective_quadratic, 
                         np.ones(n_fos),
                         args = (f, g, z0, z1_float, state, n_fos, v, ξ),
                         method = method,
                         tol = tol,
                         options = {'maxiter': max_iter})
        if model.success:
            break
    if model.success == False:
        print("---Warning: the convex solver fails when ξ = %s, tolerance = %s--- " % (ξ, tol))
        print(model.message)

    # Calculate v+μ, λ_1 and λ_2 (here λ_1 is of dimension self.n_f)
    v_μ = - model.fun
    λ_1 = model.x[:-1]
    λ_2 = model.x[-1]
    return v_μ, λ_1, λ_2


class OptimizeResult(dict):
    """ Represents the optimization result.
    Attributes
    ----------
    ξ : float
        The coefficient of relative entropy constraint that users provide.
    ϵ, μ : float
        Model solution.
    e, v : (n_states,) ndarray
        Model solution.
    λ : (n_f,) ndarray
        Model solution.
    count : int
        Number of iterations.
    N : (n,) ndarray
        One-period change of measure.
    RE_cond : (n_states,) ndarray
        Implied conditional relative entropy.
    RE : float
        Unconditional relative entropy.
    QD_cond : (n_states,) ndarray
        Implied conditional quadratic divergence.
    QD : float
        Unconditional quadratic divergence.
    P : (n_states, n_states) ndarray
        Empirical transition probabilities.
    P_tilde : (n_states, n_states) ndarray
        Distorted transition probabilities.
    π : (n_states,) ndarray
        Empirical stationary probabilities.
    π_tilde : (n_states,) ndarray
        Distorted stationary probabilities.
    moment_empirical : float
        Empirical moment of g.
    moment_bound : float
        Moment bound on g.
    moment_empirical_cond : (n_states,) ndarray
        Empirical conditional moment of g.
    moment_bound_cond : (n_states,) ndarray
        Conditional moment bound on g.

    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())