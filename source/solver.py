import numpy as np
from scipy.optimize import minimize


def optimize(f, g, indicator_z0, indicator_z1, ξ, n_states, tol=1e-9, max_iter=1000):
    """
    This function implements the iteration scheme in the Computational
    Strategy section in the Belief_Notebook. It assumes a n-state Markov
    process for Z.
    
    Parameters
    ----------
    f : (N, n_f) ndarray
        f that satisfies E[N_1f(X_1)|Z_0] = 0.
    g : (N, 1) ndarray
        The function to be bounded.
    indicator_z0 : (N, n_states) ndarray
        Indicator function of Z_0 for each sample point.
    indicator_z1 : (N, n_states) ndarray
        Indicator function of Z_1 for each sample point.
    ξ : float
        Coefficient on relative entropy constraint.
    n_states : int
        Number of states.
    tol : float
        Tolerance parameter for the iteration.
    max_iter : int
        Maximum number of iterations.
    
    
    Returns
    -------        
    res : OptimizeResult
        Class that stores the results of the optimization.
    """
    error = 1.
    count = 0
    e = np.ones(n_states)
    v = np.zeros(n_states)
    λ = np.zeros(f.shape[1])
    n_fos = int(f.shape[1]/n_states)
    
    while error > tol:
        for state in range(n_states):
            v[state], λ[state*n_fos: (state+1)*n_fos]\
            = _minimize_objective(f, g, indicator_z0, indicator_z1, state, n_fos, e, ξ, tol, max_iter)
        e_old = e
        ϵ = v[0]
        e = v/v[0]
        error = np.max(np.abs(e - e_old))
        count += 1

    # N_1 and E[N_1 | state k]
    N = 1./ϵ * np.exp(-g/ξ+f@λ) * (indicator_z1@e) / (indicator_z0@e)
    E_N_cond = np.zeros(n_states)
    for state in range(n_states):
        E_N_cond[state] = np.mean(N[indicator_z0[:, state]])
    
    # Empirical transition matrix and stationary distribution
    P = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            P[i, j] = np.mean(indicator_z1[indicator_z0[:, i]][:, j]) 
    π = _stationary_prob(P)

    # Distorted transition matrix and stationary distribution
    P_tilde = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            P_tilde[i, j] = np.mean(N[indicator_z0[:, i]] * indicator_z1[indicator_z0[:, i]][:, j])
    π_tilde = _stationary_prob(P_tilde)
    
    # Conditional and unconditional relative entropy
    RE_cond = np.zeros(n_states)
    for state in range(n_states):
        RE_cond[state] = np.mean(N[indicator_z0[:, state]] * np.log(N[indicator_z0[:, state]]))
    RE = RE_cond @ π_tilde
    
    # Conditional and unconditional moment bounds for g
    moment_bound_cond = np.zeros(n_states)
    for state in range(n_states):
        moment_bound_cond[state] = np.mean(N[indicator_z0[:, state]]* g[indicator_z0[:, state]])
    moment_bound = moment_bound_cond @ π_tilde

    # Conditional and unconditional empirical moment for g
    moment_empirical_cond = np.zeros(n_states)
    for state in range(n_states):
        moment_empirical_cond[state] = np.mean(g[indicator_z0[:, state]])
    moment_empirical = np.mean(g)

    # Calculate v
    v_0 = -ξ * np.log(e)    
    
    res = OptimizeResult({'ϵ':ϵ,
                          'e':e,
                          'λ':λ,
                          'count':count,
                          'ξ':ξ,
                          'μ':μ,
                          'v_0':v_0,
                          'RE_cond':RE_cond,
                          'RE':RE,
                          'E_N_cond':E_N_cond,
                          'P':P,
                          'π':π,
                          'P_tilde':P_tilde,
                          'π_tilde':π_tilde,
                          'moment_bound':moment_bound,
                          'moment_bound_cond':moment_bound_cond,
                          'moment_empirical_cond':moment_empirical_cond,
                          'moment_empirical':moment_empirical,
                          'N':N})
    return res


@jit
def _objective(λ, f, g, indicaotor_z0, indicator_z1_float,
               state, n_fos, e, ξ):
    """
    The objective function.
    
    """
    selector = indicaotor_z0[:,state]
    term_1 = -g[selector]/ξ
    term_2 = f[:,state*n_fos: (state+1)*n_fos][selector]@λ
    term_3 = np.log(pd_indicator_float[selector]@e)
    x = term_1 + term_2 + term_3
    # Use "max trick" to improve accuracy
    a = x.max()
    return np.log(np.mean(np.exp(x-a))) + a    


@jit
def _objective_gradient(λ, f, g, indicaotor_z0, indicator_z1_float,
                        state, n_fos, e, ξ):
    """
    Gradient of the objective function.
    
    """
    selector = pd_lag_indicator[:,state]
    temp1 = -g[selector]/ξ + f[:,state*n_fos:(state+1)*n_fos][selector]@λ + np.log(pd_indicator_float[selector]@e)
    temp2 = f[:,state*n_fos:(state+1)*n_fos][selector]*(np.exp(temp1.reshape((len(temp1),1)))/np.mean(np.exp(temp1)))
    temp3 = np.empty(temp2.shape[1])
    for i in range(temp2.shape[1]):
        temp3[i] = np.mean(temp2[:,i])
    return temp3         


def _minimize_objective(f, g, indicator_z0, indicator_z1, state, n_fos, e, ξ, tol, max_iter):
    """
    Use scipy.minimize (L-BFGS-B, BFGS or CG) to solve the minimization problem.

    """
    indicator_z1_float = indicator_z1.astype(float)

    for method in ['L-BFGS-B','BFGS','CG']:
        model = minimize(_objective, 
                         np.ones(n_fos),
                         args = (f, g, indicaotor_z0, indicator_z1_float, state, n_fos, e, ξ)
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


def _stationary_prob(P):
    A = P.T - np.eye(P.shape[0])
    A[-1] = np.ones(P.shape[0])
    B = np.zeros(P.shape[0])
    B[-1] = 1.
    π = np.linalg.solve(A, B)
    return π


class OptimizeResult(dict):
    """ Represents the optimization result.
    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.
    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
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