"""
This module produces plots and tables for the notebook.

"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets, interact
from source.preprocessing import preprocess_data
from source.solver import solve, find_ξ
from source.extension import bound_ratio


def figure_1():
    # Figure 1: bounds on expected log market return
    # Below we load presolved ξs but users can use find_ξ to resolve them.    
    f, log_Rw, z0, z1, Rf, Rm, SMB, HML = preprocess_data(n_states=3)
    g = log_Rw

    result_min = solve(f, g, z0, z1, ξ=10., quadratic=False,
                       tol=1e-9, max_iter=1000)

    ξs_lower = [100, 0.29255, 0.20511, 0.16625, 0.14302, 0.12713, 0.11539]
    ξs_upper = [100, 0.29898, 0.21142, 0.17248, 0.14918, 0.13323, 0.12141]
    result_lower_list = []
    result_upper_list = []

    for i in range(0,7):
        temp = solve(f, g, z0, z1, ξ=ξs_lower[i],
                     quadratic=False, tol=1e-9, max_iter=1000)
        result_lower_list.append(temp)
        temp = solve(f, -g, z0, z1, ξ=ξs_upper[i],
                     quadratic=False, tol=1e-9, max_iter=1000)
        result_upper_list.append(temp)

    def f1(percent):
        box_chart(result_min,
                  result_lower_list[int(percent/5)],
                  result_upper_list[int(percent/5)])
    res = interact(f1, percent=widgets.IntSlider(min=0, max=30, step=5, value=20))
    return res


def figure_2():
    # Figure 2: bounds on risk premia
    f, log_Rw, z0, z1, Rf, Rm, SMB, HML = preprocess_data(n_states=3)
    g1 = Rm
    g2 = Rf
    solver_args = (f, z0, z1, False, 1e-9, 1000)
    risk_premia_min, risk_premia_cond_min, risk_premia_empirical, risk_premia_cond_empirical\
        = bound_ratio(find_ξ_args=(solver_args, 0., None, None, None, None),
                         g1=g1, g2=g2, ζ=1.)
    result_risk_min = {'moment_empirical':risk_premia_empirical,
                       'moment_empirical_cond':risk_premia_cond_empirical,
                       'moment_bound':risk_premia_min,
                       'moment_bound_cond':risk_premia_cond_min}

    # Below we load presolved ζs but users can plot the objective function over ζ to find the optimal ζs. 
    # ζs that correspond to 0%-30% higher min RE (risk premia), step size=5%
    ζs_lower = [1., 1.007, 1.006, 1.006, 1.006, 1.006, 1.008]
    ζs_upper = [1., 1.007, 1.007, 1.008, 1.007, 1.008, 1.006]
    result_risk_lower_list = []
    result_risk_upper_list = []
    for i in range(0,7):
        # lower bound
        risk_premia_lower, risk_premia_cond_lower, _ ,_\
            = bound_ratio(find_ξ_args=(solver_args, i*0.05, 1., (0., 100.), 1e-5, 100),
                             g1=g1, g2=g2, ζ=ζs_lower[i], lower=True)
        result_risk_lower = {'moment_bound':risk_premia_lower,
                             'moment_bound_cond':risk_premia_cond_lower}
        result_risk_lower_list.append(result_risk_lower)
        # upper bound
        risk_premia_upper, risk_premia_cond_upper, _ ,_\
            = bound_ratio(find_ξ_args=(solver_args, i*0.05, 1., (0., 100.), 1e-5, 100),
                             g1=g1, g2=g2, ζ=ζs_upper[i], lower=False)    
        result_risk_upper = {'moment_bound':-risk_premia_upper,
                             'moment_bound_cond':-risk_premia_cond_upper}
        result_risk_upper_list.append(result_risk_upper)

    def f2(percent):
        box_chart(result_risk_min,
                  result_risk_lower_list[int(percent/5)],
                  result_risk_upper_list[int(percent/5)])
    
    res = interact(f2, percent=widgets.IntSlider(min=0, max=30, step=5, value=20))
    
    return res


def table_1(n_states=3, quadratic=False):
    f, log_Rw, z0, z1, Rf, Rm, SMB, HML = preprocess_data(n_states)
    g = log_Rw # Set g to be log return on wealth

    result_min = solve(f, g, z0, z1, ξ=100., quadratic=quadratic,
                       tol=1e-9, max_iter=1000)

    # Table 1: transition matrix and stationary probability
    print('Table 1: Empirical and distorted transition probabilities')
    print('-------------------------------------------------------')
    print('               empirical             min divergence')
    print('-------------------------------------------------------')
    print('Transition Matrix:')
    for state in np.arange(1, n_states+1):
        print(f"           {np.round(result_min['P'][state-1],2)}         {np.round(result_min['P_tilde'][state-1],2)}")
    print('')
    print('Stationary Probability:')
    print(f"           {np.round(result_min['π'],2)}         {np.round(result_min['π_tilde'],2)}")
    print('-------------------------------------------------------')


def table_2():
    # Table 2: bounds on log expected return and generalized volatility
    f, log_Rw, z0, z1, Rf, Rm, SMB, HML = preprocess_data(n_states=3)

    # 1) Calculate bounds on expected return
    # Minimum divergence case
    result_min = solve(f, Rm, z0, z1, ξ=100., quadratic=False,
                       tol=1e-9, max_iter=1000)

    # 20% higher divergence case, lower bound problem
    ξ_20_lower = find_ξ(solver_args=(f, Rm, z0, z1, False, 1e-9, 1000),
                        min_div=result_min['RE'], pct=0.2, initial_guess=1.,
                        interval=(0, 100.), tol=1e-5, max_iter=100)
    result_lower = solve(f=f, g=Rm, z0=z0, z1=z1, ξ=ξ_20_lower,
                         quadratic=False, tol=1e-9, max_iter=1000)

    # 20% higher divergence case, upper bound problem
    ξ_20_upper = find_ξ(solver_args=(f, -Rm, z0, z1, False, 1e-9, 1000),
                        min_div=result_min['RE'], pct=0.2, initial_guess=1.,
                        interval=(0, 100.), tol=1e-5, max_iter=100)
    result_upper = solve(f=f, g=-Rm, z0=z0, z1=z1, ξ=ξ_20_upper,
                         quadratic=False, tol=1e-9, max_iter=1000)

    # 2) Calculate bounds on generalized volatility
    # Below we load presolved ζs but users can plot the objective function over ζ to find the optimal ζs. 
    g1 = Rm
    g2 = log_Rw
    solver_args = (f, z0, z1, False, 1e-9, 1000)
    vol_min, vol_cond_min, vol_empirical, vol_cond_empirical\
        = bound_ratio(find_ξ_args=(solver_args, 0., None, None, None, None),
                      g1=g1, g2=g2, ζ=1., lower=True, result_type=2)
    vol_lower, vol_cond_lower, _, _\
        = bound_ratio(find_ξ_args=(solver_args, 0.2, 1., (0., 10.), 1e-5, 100),
                         g1=g1, g2=g2, ζ=1.008, lower=True, result_type=2)
    vol_upper, vol_cond_upper, _, _\
        = bound_ratio(find_ξ_args=(solver_args, 0.2, 1., (0., 10.), 1e-5, 100),
                         g1=g1, g2=g2, ζ=1.009, lower=False, result_type=2)

    print('Table 2: Log expected market return and generalized volatility')
    print('-------------------------------------------------------------------------------')
    print('conditioning      logE           logE          logE - Elog     logE - Elog')
    print('                empirical       imputed         empirical        imputed')
    print('                             (lower, upper)                   (lower,upper)')
    print('-------------------------------------------------------------------------------')
    print('low D/P           %s           %s             %s            %s' \
          % (np.round(np.log(result_min['moment_empirical_cond'][0])*400,2),
             np.round(np.log(result_min['moment_bound_cond'][0])*400,2),
             np.round(vol_cond_empirical[0]*400,2),
             np.round(vol_cond_min[0]*400,2)))
    print('                              (%s,%s)                      (%s,%s)' \
          % (np.round(np.log(result_lower['moment_bound_cond'][0])*400,2),
             np.round(np.log(-result_upper['moment_bound_cond'][0])*400,2),
             np.round(vol_cond_lower[0]*400,2),np.round(vol_cond_upper[0]*400,2)))
    print('mid D/P           %s            %s             %s            %s' \
          % (np.round(np.log(result_min['moment_empirical_cond'][1])*400,2),
             np.round(np.log(result_min['moment_bound_cond'][1])*400,2),
             np.round(vol_cond_empirical[1]*400,2),np.round(vol_cond_min[1]*400,2)))
    print('                              (%s,%s)                      (%s,%s)' \
          % (np.round(np.log(result_lower['moment_bound_cond'][1])*400,2),
             np.round(np.log(-result_upper['moment_bound_cond'][1])*400,2),
             np.round(vol_cond_lower[1]*400,2),np.round(vol_cond_upper[1]*400,2)))
    print('high D/P          %s          %s             %s            %s' \
          % (np.round(np.log(result_min['moment_empirical_cond'][2])*400,2),
             np.round(np.log(result_min['moment_bound_cond'][2])*400,2),
             np.round(vol_cond_empirical[2]*400,2),np.round(vol_cond_min[2]*400,2)))
    print('                              (%s,%s)                      (%s,%s)' \
          % (np.round(np.log(result_lower['moment_bound_cond'][2])*400,2),
             np.round(np.log(-result_upper['moment_bound_cond'][2])*400,2),
             np.round(vol_cond_lower[2]*400,2),np.round(vol_cond_upper[2]*400,2)))
    print('unconditional     %s           %s             %s            %s' \
          % (np.round(np.log(result_min['moment_empirical'])*400,2),
             np.round(np.log(result_min['moment_bound'])*400,2),
             np.round(vol_empirical*400,2),np.round(vol_min*400,2)))
    print('                              (%s,%s)                      (%s,%s)' \
          % (np.round(np.log(result_lower['moment_bound'])*400,2),
             np.round(np.log(-result_upper['moment_bound'])*400,2),
             np.round(vol_lower*400,2),np.round(vol_upper*400,2)))
    print('-------------------------------------------------------------------------------')
    print('Note 1: here we use n_states = 3 and a relative entropy divergence.')
    print('Note 2: the numbers in the parentheses impose a divergence constraint')
    print('        that is 20 percent higher than the minimum.')

    
def table_3():
    # Table 3: Expected log market return bounds
    f, log_Rw, z0, z1, Rf, Rm, SMB, HML = preprocess_data(n_states=3)
    g = log_Rw # Set g to be log return on wealth

    # 1) Relative entropy specification
    result_min_RE = solve(f, g, z0, z1, ξ=100., quadratic=False,
                       tol=1e-9, max_iter=1000)
    ξ_20_lower_RE = find_ξ(solver_args=(f, g, z0, z1, False, 1e-9, 1000),
                        min_div=result_min_RE['RE'], pct=0.2, initial_guess=1.,
                        interval=(0, 10.), tol=1e-5, max_iter=100)
    result_lower_RE = solve(f, g, z0, z1, ξ=ξ_20_lower_RE, quadratic=False,
                         tol=1e-9, max_iter=1000)
    ξ_20_upper_RE = find_ξ(solver_args=(f, -g, z0, z1, False, 1e-9, 1000),
                        min_div=result_min_RE['RE'], pct=0.2, initial_guess=1.,
                        interval=(0, 10.), tol=1e-5, max_iter=100)
    result_upper_RE = solve(f, -g, z0, z1, ξ=ξ_20_upper_RE, quadratic=False,
                         tol=1e-9, max_iter=1000)

    # 2) Quadratic specification
    result_min_QD = solve(f, g, z0, z1, ξ=10., quadratic=True,
                       tol=1e-9, max_iter=1000)
    ξ_20_lower_QD = find_ξ(solver_args=(f, g, z0, z1, True, 1e-9, 1000),
                        min_div=result_min_QD['QD'], pct=0.2, initial_guess=1.,
                        interval=(0, 10.), tol=1e-4, max_iter=100)
    result_lower_QD = solve(f, g, z0, z1, ξ=ξ_20_lower_QD, quadratic=True,
                         tol=1e-9, max_iter=1000)
    ξ_20_upper_QD = find_ξ(solver_args=(f, -g, z0, z1, True, 1e-9, 1000),
                        min_div=result_min_QD['QD'], pct=0.2, initial_guess=1.,
                        interval=(0, 10.), tol=1e-4, max_iter=100)
    result_upper_QD = solve(f, -g, z0, z1, ξ=ξ_20_upper_QD, quadratic=True,
                         tol=1e-9, max_iter=1000)

    print('Table 3: Expected log market return bounds')
    print('-----------------------------------------------------------------------')
    print('conditioning    empirical   relative entropy     quadratic divergence')
    print('                             (lower, upper)         (lower,upper)')
    print('-----------------------------------------------------------------------')
    print('low D/P           %s         (%s,%s)           (%s,%s) ' \
          % (np.round(result_min_RE['moment_empirical_cond'][0]*400,2),
             np.round(result_lower_RE['moment_bound_cond'][0]*400,2),
             np.round(-result_upper_RE['moment_bound_cond'][0]*400,2),
             np.round(result_lower_QD['moment_bound_cond'][0]*400,2),
             np.round(-result_upper_QD['moment_bound_cond'][0]*400,2)))
    print('mid D/P           %s         (%s,%s)           (%s,%s)    ' \
          % (np.round(result_min_RE['moment_empirical_cond'][1]*400,2),
             np.round(result_lower_RE['moment_bound_cond'][1]*400,2),
             np.round(-result_upper_RE['moment_bound_cond'][1]*400,2),         
             np.round(result_lower_QD['moment_bound_cond'][1]*400,2),
             np.round(-result_upper_QD['moment_bound_cond'][1]*400,2)))
    print('high D/P          %s        (%s,%s)           (%s,%s)  ' \
          % (np.round(result_min_RE['moment_empirical_cond'][2]*400,2),
             np.round(result_lower_RE['moment_bound_cond'][2]*400,2),
             np.round(-result_upper_RE['moment_bound_cond'][2]*400,2),         
             np.round(result_lower_QD['moment_bound_cond'][2]*400,2),
             np.round(-result_upper_QD['moment_bound_cond'][2]*400,2)))
    print('unconditional     %s         (%s,%s)           (%s,%s)   ' \
          % (np.round(result_min_RE['moment_empirical']*400,2),
             np.round(result_lower_RE['moment_bound']*400,2),
             np.round(-result_upper_RE['moment_bound']*400,2),         
             np.round(result_lower_QD['moment_bound']*400,2),
             np.round(-result_upper_QD['moment_bound']*400,2)))
    print('-----------------------------------------------------------------------')
    print('Note 1: here we use n_states = 3.')
    print('Note 2: the numbers in the parentheses impose a divergence constraint')
    print('        that is 20 percent higher than the minimum.')


def table_4():
    # Table 4: Proportional risk compensations
    f, log_Rw, z0, z1, Rf, Rm, SMB, HML = preprocess_data(n_states=3)

    # 1) Calculate bounds on risk premia of market return
    # Below we load presolved ζs but users can plot the objective function over ζ to find the optimal ζs. 
    g1 = Rm
    g2 = Rf
    solver_args = (f, z0, z1, False, 1e-9, 1000)
    risk_Rm_min, risk_Rm_cond_min, risk_Rm_empirical, risk_Rm_cond_empirical\
        = bound_ratio(find_ξ_args=(solver_args, 0., None, None, None, None),
                      g1=g1, g2=g2, ζ=1., lower=True)
    risk_Rm_lower, risk_Rm_cond_lower, _, _\
        = bound_ratio(find_ξ_args=(solver_args, 0.2, 1., (0., 10.), 1e-5, 100),
                         g1=g1, g2=g2, ζ=1.006, lower=True)
    risk_Rm_upper, risk_Rm_cond_upper, _, _\
        = bound_ratio(find_ξ_args=(solver_args, 0.2, 1., (0., 10.), 1e-5, 100),
                         g1=g1, g2=g2, ζ=1.007, lower=False)

    # 2) Calculate bounds on risk premia of SMB return
    # Below we load presolved ζs but users can plot the objective function over ζ to find the optimal ζs. 
    g1 = SMB
    g2 = Rf
    solver_args = (f, z0, z1, False, 1e-9, 1000)
    risk_SMB_min, risk_SMB_cond_min, risk_SMB_empirical, risk_SMB_cond_empirical\
        = bound_ratio(find_ξ_args=(solver_args, 0., None, None, None, None),
                      g1=g1, g2=g2, ζ=1., lower=True)
    risk_SMB_lower, risk_SMB_cond_lower, _, _\
        = bound_ratio(find_ξ_args=(solver_args, 0.2, 1., (0., 10.), 1e-5, 100),
                         g1=g1, g2=g2, ζ=1.001, lower=True)
    risk_SMB_upper, risk_SMB_cond_upper, _, _\
        = bound_ratio(find_ξ_args=(solver_args, 0.2, 1., (0., 10.), 1e-5, 100),
                         g1=g1, g2=g2, ζ=1.002, lower=False)

    # 3) Calculate bounds on risk premia of HML return
    # Below we load presolved ζs but users can plot the objective function over ζ to find the optimal ζs. 
    g1 = HML
    g2 = Rf
    solver_args = (f, z0, z1, False, 1e-9, 1000)
    risk_HML_min, risk_HML_cond_min, risk_HML_empirical, risk_HML_cond_empirical\
        = bound_ratio(find_ξ_args=(solver_args, 0., None, None, None, None),
                      g1=g1, g2=g2, ζ=1., lower=True)
    risk_HML_lower, risk_HML_cond_lower, _, _\
        = bound_ratio(find_ξ_args=(solver_args, 0.2, 1., (0., 10.), 1e-5, 100),
                         g1=g1, g2=g2, ζ=0.999, lower=True)
    risk_HML_upper, risk_HML_cond_upper, _, _\
        = bound_ratio(find_ξ_args=(solver_args, 0.2, 1., (0., 10.), 1e-5, 100),
                         g1=g1, g2=g2, ζ=1., lower=False)
    
    print('Table 4: Proportional risk compensations')
    print('------------------------------------------------------------------------')
    print('conditioning      market return     small minus big     high minus low')
    print('                  (lower, upper)     (lower, upper)     (lower, upper)')
    print('                    empirical          empirical           empirical')
    print('------------------------------------------------------------------------')
    print('low D/P           (%s,%s)         (%s,%s)       (%s,%s)'  \
          % (np.round(risk_Rm_cond_lower[0]*400,2),
             np.round(risk_Rm_cond_upper[0]*400,2),
             np.round(risk_SMB_cond_lower[0]*400,2),
             np.round(risk_SMB_cond_upper[0]*400,2),
             np.round(risk_HML_cond_lower[0]*400,2),
             np.round(risk_HML_cond_upper[0]*400,2)))
    print('                      %s                %s               %s' \
          % (np.round(risk_Rm_cond_empirical[0]*400,2),
             np.round(risk_SMB_cond_empirical[0]*400,2),
             np.round(risk_HML_cond_empirical[0]*400,2)))
    print('mid D/P           (%s,%s)         (%s,%s)       (%s,%s)'  \
          % (np.round(risk_Rm_cond_lower[1]*400,2),
             np.round(risk_Rm_cond_upper[1]*400,2),
             np.round(risk_SMB_cond_lower[1]*400,2),
             np.round(risk_SMB_cond_upper[1]*400,2),
             np.round(risk_HML_cond_lower[1]*400,2),
             np.round(risk_HML_cond_upper[1]*400,2)))
    print('                      %s                %s               %s' \
          % (np.round(risk_Rm_cond_empirical[1]*400,2),
             np.round(risk_SMB_cond_empirical[1]*400,2),
             np.round(risk_HML_cond_empirical[1]*400,2)))
    print('high D/P          (%s,%s)         (%s,%s)       (%s,%s)'  \
          % (np.round(risk_Rm_cond_lower[2]*400,2),
             np.round(risk_Rm_cond_upper[2]*400,2),
             np.round(risk_SMB_cond_lower[2]*400,2),
             np.round(risk_SMB_cond_upper[2]*400,2),
             np.round(risk_HML_cond_lower[2]*400,2),
             np.round(risk_HML_cond_upper[2]*400,2)))
    print('                     %s                %s               %s' \
          % (np.round(risk_Rm_cond_empirical[2]*400,2),
             np.round(risk_SMB_cond_empirical[2]*400,2),
             np.round(risk_HML_cond_empirical[2]*400,2)))
    print('unconditional     (%s,%s)         (%s,%s)       (%s,%s)'  \
          % (np.round(risk_Rm_lower*400,2),
             np.round(risk_Rm_upper*400,2),
             np.round(risk_SMB_lower*400,2),
             np.round(risk_SMB_upper*400,2),
             np.round(risk_HML_lower*400,2),
             np.round(risk_HML_upper*400,2)))
    print('                      %s                %s               %s' \
          % (np.round(risk_Rm_empirical*400,2),
             np.round(risk_SMB_empirical*400,2),
             np.round(risk_HML_empirical*400,2)))
    print('------------------------------------------------------------------------')
    print('Note 1: here we use n_states = 3 and a relative entropy divergence.')
    print('Note 2: the numbers in the parentheses impose a divergence constraint')
    print('        that is 20 percent higher than the minimum.')
    

def table_5(n_states=3):
    # Table 5: Comparison of transition matrices
    f, log_Rw, z0, z1, Rf, Rm, SMB, HML = preprocess_data(n_states)
    g = log_Rw # Set g to be log return on wealth

    result_re = solve(f, g, z0, z1, ξ=10., quadratic=False,
                       tol=1e-9, max_iter=1000)
    result_qd = solve(f, g, z0, z1, ξ=10., quadratic=True,
                       tol=1e-9, max_iter=1000)

    print('Table 5: Transition probabilities and stationary probabilities')
    print('-------------------------------------------------------')
    print('          relative entropy        quadratic divergence')
    print('-------------------------------------------------------')
    print('Transition Matrix:')
    for state in np.arange(1, n_states+1):
        print(f"           {np.round(result_re['P'][state-1],2)}         {np.round(result_qd['P_tilde'][state-1],2)}")
    print('')
    print('Stationary Probability:')
    print(f"           {np.round(result_re['π'],2)}         {np.round(result_qd['π_tilde'],2)}")
    print('-------------------------------------------------------')


def objective_vs_ξ(n_states):
    """
    An illustration of how the optimized μ and ϵ change with ξ.

    """
    f, log_Rw, z0, z1, Rf, Rm, SMB, HML = preprocess_data(n_states)    
    ξ_grid = np.arange(.01,1.01,.005)
    results_lower = [None]*len(ξ_grid)

    for i in range(len(ξ_grid)):
        ξ = ξ_grid[i]
        temp = solve(f=f, g=log_Rw, z0=z0, z1=z1, ξ=ξ,
                     quadratic=False, tol=1e-9, max_iter=1000)
        results_lower[i] = temp

    μs_lower = np.array([result['μ'] for result in results_lower])
    ϵs_lower = np.array([result['ϵ'] for result in results_lower])

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=μs_lower, name='μ', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=ϵs_lower, name='ϵ', line=dict(color='green')),
        row=1, col=2
    )
    fig.update_layout(height=400, width=1000, title_text="Minimized μ (left) and ϵ (right)", showlegend = False)
    fig.update_xaxes(rangemode="tozero",title_text='ξ')
    fig.update_yaxes(rangemode="tozero")
    
    fig['layout']['xaxis'+str(int(1))].update(range = (0.,1.))
    fig['layout']['xaxis'+str(int(2))].update(range = (0.,1.))

    fig.show()
    
    
def entropy_moment_bounds(n_states):
    f, log_Rw, z0, z1, Rf, Rm, SMB, HML = preprocess_data(n_states)
    ξ_grid = np.arange(.01,1.01,.01)
    results_lower = [None]*len(ξ_grid)
    results_upper = [None]*len(ξ_grid)

    for i in range(len(ξ_grid)):
        ξ = ξ_grid[i]
        temp = solve(f=f, g=log_Rw, z0=z0, z1=z1, ξ=ξ,
                     quadratic=False, tol=1e-9, max_iter=1000)        
        results_lower[i] = temp
        temp = solve(f=f, g=-log_Rw, z0=z0, z1=z1, ξ=ξ,
                     quadratic=False, tol=1e-9, max_iter=1000)          
        results_upper[i] = temp

    REs_lower = np.array([result['RE'] for result in results_lower])
    moment_bounds_cond_lower = np.array([result['moment_bound_cond'] for result in results_lower])
    moment_bounds_cond_upper = np.array([-result['moment_bound_cond'] for result in results_upper])
    moment_bounds_lower = np.array([result['moment_bound'] for result in results_lower])
    moment_bounds_upper = np.array([-result['moment_bound'] for result in results_upper])
    moment_cond = np.array([result['moment_empirical_cond'] for result in results_lower])
    moment = np.array([result['moment_empirical'] for result in results_lower])

    # Plots for RE and E[Mg(X)]
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=np.ones_like(ξ_grid)*REs_lower[-1]*1.2, name='1.2x min RE', line=dict(color='black',dash='dash')),
        row=1, col=1
    )    
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=REs_lower, name='lower bound', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_lower, name='lower bound', line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_upper, name='upper bound', line=dict(color='red')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment, name='E[g(X)]',line=dict(dash='dash',color='orange')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_cond_lower[:,0], name='lower bound', visible=False, line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_cond_upper[:,0], name='upper bound', visible=False, line=dict(color='red')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_cond[:,0], name='E[g(X)|1]', visible=False, line=dict(dash='dash',color='orange')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_cond_lower[:,1], name='lower bound', visible=False, line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_cond_upper[:,1], name='upper bound', visible=False, line=dict(color='red')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_cond[:,1], name='E[g(X)|2]', visible=False, line=dict(dash='dash',color='orange')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_cond_lower[:,2], name='lower bound', visible=False, line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_cond_upper[:,2], name='upper bound', visible=False, line=dict(color='red')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_cond[:,2], name='E[g(X)|3]', visible=False, line=dict(dash='dash',color='orange')),
        row=1, col=2
    )
    fig.update_layout(height=400, width=1000, title_text="Relative entropy (left) and moment bounds (right)", showlegend = False)
    fig.update_xaxes(rangemode="tozero",title_text='ξ')
    fig.update_yaxes(rangemode="tozero")

    fig['layout']['xaxis'+str(int(1))].update(range = (0.,1.))
    fig['layout']['yaxis'+str(int(1))].update(range = (0.,0.06))
    fig['layout']['xaxis'+str(int(2))].update(range = (0.,1.))
    fig['layout']['yaxis'+str(int(2))].update(range = (-0.01,0.04))

    # Add button
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="up",
                active=0,
                x=1.2,
                y=0.9,
                buttons=list([
                    dict(label="Unconditional",
                         method="update",
                         args=[{"visible": [True]*5 + [False]*15 }]),
                    dict(label="State 1",
                         method="update",
                         args=[{"visible": [True]*2 + [False]*3 + [True]*3 + [False]*6}]),
                    dict(label="State 2",
                         method="update",
                         args=[{"visible": [True]*2 + [False]*6 + [True]*3 + [False]*3}]),
                    dict(label="State 3",
                         method="update",
                         args=[{"visible": [True]*2 + [False]*9 + [True]*3}]),
                ]),
            )
        ])

    fig.show()


def box_chart(result_min, result_lower, result_upper, save=False):
    conditioning = ['low D/P', 'middle D/P', 'high D/P', 'unconditional']
    min_entropy_implied = np.array([result_min['moment_bound_cond'][0]*400,
                                    result_min['moment_bound_cond'][1]*400,
                                    result_min['moment_bound_cond'][2]*400,
                                    result_min['moment_bound']*400])
    empirical_average = np.array([result_min['moment_empirical_cond'][0]*400,
                                  result_min['moment_empirical_cond'][1]*400,
                                  result_min['moment_empirical_cond'][2]*400,
                                  result_min['moment_empirical']*400])
    # Each number is repeated once in order to form a desirable shape of boxes
    low_bound_20 = np.array([result_lower['moment_bound_cond'][0]*400,
                             result_lower['moment_bound_cond'][0]*400,
                             -result_upper['moment_bound_cond'][0]*400,
                             -result_upper['moment_bound_cond'][0]*400]) 
    middle_bound_20 = np.array([result_lower['moment_bound_cond'][1]*400,
                                result_lower['moment_bound_cond'][1]*400,
                                -result_upper['moment_bound_cond'][1]*400,
                                -result_upper['moment_bound_cond'][1]*400])
    high_bound_20 = np.array([result_lower['moment_bound_cond'][2]*400,
                              result_lower['moment_bound_cond'][2]*400,
                              -result_upper['moment_bound_cond'][2]*400,
                              -result_upper['moment_bound_cond'][2]*400])
    unconditional_bound_20 = np.array([result_lower['moment_bound']*400,
                                       result_lower['moment_bound']*400,
                                       -result_upper['moment_bound']*400,
                                       -result_upper['moment_bound']*400])

    fig, ax = plt.subplots()
    bplot = ax.boxplot(np.vstack((low_bound_20,
                                  middle_bound_20,
                                  high_bound_20,
                                  unconditional_bound_20)).T,
                       usermedians=min_entropy_implied, # Override the automatically calculated median
                       labels=conditioning,
                       widths=0.4, 
                       medianprops=dict(linestyle='-.',
                                        linewidth=0,
                                        color='black'),
                       patch_artist=True)

    for box in bplot['boxes']:
        box.set(facecolor='salmon')

    splot = ax.scatter(x=np.arange(1, 5),
                       y=empirical_average,
                       c='black')

    ax.axvline(x=3.5,
               color='black',
               linestyle='--',
               linewidth = 1)

    ax.set_ylim(0.,14.5)

    plt.show()

    if save:
        fig.savefig("plot.pdf")


def print_results(result_lower, result_upper, quadratic=False):
    n_states = result_lower['P'].shape[0]
    # Print iteration information
    print("--- Iteration information ---")
    print("Number of iterations (lower bound problem): %s" % (result_lower['count']))
    print("Number of iterations (upper bound problem): %s" % (result_upper['count']))

    # Print converged parameter results
    print("\n")
    print("--- Converged values for the lower bound problem ---")
    if quadratic:
        print("μ: %s" % np.round(result_lower['μ'],2))
        print("v: %s" % np.round(result_lower['v'],2))
    else:
        print("ϵ: %s" % np.round(result_lower['ϵ'],2))
        print("e: %s" % np.round(result_lower['e'],2))
    print("λ: %s" % np.round(result_lower['λ'],2))

    print(" ")
    print("--- Converged values for the upper bound problem ---")
    if quadratic:
        print("μ: %s" % np.round(result_upper['μ'],2))
        print("v: %s" % np.round(result_upper['v'],2))
    else:
        print("ϵ: %s" % np.round(result_upper['ϵ'],2))
        print("e: %s" % np.round(result_upper['e'],2))
    print("λ: %s" % np.round(result_upper['λ'],2))

    # Print transition probability matrix under the original empirical probability
    print("\n")
    print("--- Transition Probability Matrix (Original) ---")
    print(np.round(result_lower['P'],2))

    # Print transition probability matrix under distorted probability, lower bound
    print(" ")
    print("--- Transition Probability Matrix (Distorted, lower bound problem) ---")
    print(np.round(result_lower['P_tilde'],2))

    # Print transition probability matrix under distorted probability, upper bound
    print(" ")
    print("--- Transition Probability Matrix (Distorted, upper bound problem) ---")
    print(np.round(result_upper['P_tilde'],2))

    # Print stationary distribution under the original empirical probability
    print("\n")
    print("--- Stationary Distribution (Original) ---")
    print(np.round(result_lower['π'],2))

    # Print stationary distribution under distorted probability, lower bound
    print(" ")
    print("--- Stationary Distribution (Distorted, lower bound problem) ---")
    print(np.round(result_lower['π_tilde'],2))

    # Print stationary distribution under distorted probability, upper bound
    print(" ")
    print("--- Stationary Distribution (Distorted, upper bound problem) ---")
    print(np.round(result_upper['π_tilde'],2))

    # Print relative entropy
    print("\n")
    print("--- Relative Entropy (lower bound problem) ---")
    for state in np.arange(1, n_states+1):
        print(f"E[NlogN|state {state}] = {np.round(result_lower['RE_cond'][state-1],4)}")
    print("E[NlogN]         = %s " % np.round(result_lower['RE'],4))

    # Print relative entropy
    print(" ")
    print("--- Relative Entropy (Upper bound problem) ---")
    for state in np.arange(1, n_states+1):
        print(f"E[NlogN|state {state}] = {np.round(result_upper['RE_cond'][state-1],4)}")
    print("E[NlogN]         = %s " % np.round(result_upper['RE'],4))
    
    if quadratic:
        # Print quadratic divergence
        print("\n")
        print("--- Quadratic Divergence (lower bound problem) ---")
        for state in np.arange(1, n_states+1):
            print(f"state {state} : {np.round(result_lower['QD_cond'][state-1],4)}")
        print("Unconditional         = %s " % np.round(result_lower['QD'],4))

        # Print quadratic divergence
        print(" ")
        print("--- Quadratic Divergence (Upper bound problem) ---")
        for state in np.arange(1, n_states+1):
            print(f"state {state} : {np.round(result_upper['QD_cond'][state-1],4)}")
        print("Unconditional         = %s " % np.round(result_upper['QD'],4))        

    # Print conditional moment & bounds
    print("\n")
    print("--- Moment (Empirical, annualized, %) ---")
    for state in np.arange(1, n_states+1):
        print(f"E[g(X)|state {state}] = {np.round(result_lower['moment_empirical_cond'][state-1]*400,2)}")
    print("E[g(X)]  = %s " % (np.round(result_lower['moment_empirical']*400,2)))
    print(" ")
    print("--- Moment (Lower bound, annualized, %) ---")
    for state in np.arange(1, n_states+1):
        print(f"E[Ng(X)|state {state}] = {np.round(result_lower['moment_bound_cond'][state-1]*400,2)}")
    print("E[Ng(X)] = %s " % (np.round(result_lower['moment_bound']*400,2)))
    print(" ")
    print("--- Moment (Upper bound, annualized, %) ---")
    for state in np.arange(1, n_states+1):
        print(f"E[Ng(X)|state {state}] = {np.round(-result_upper['moment_bound_cond'][state-1]*400,2)}")
    print("E[Ng(X)] = %s " % (np.round(-result_upper['moment_bound']*400,2)))
    print("\n")