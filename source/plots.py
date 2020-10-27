"""
This module produces plots and tables for the notebook.

"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from source.preprocessing import preprocess_data
from source.solver import solve


def objective_vs_ξ(n_states):
    """
    An illustration of how the optimized μ and ϵ change with ξ.

    """
    f, log_Rw, z0, z1, Rf, Rm, SMB, HML = preprocess_data(n_states)    
    ξ_grid = np.arange(.01,1.01,.005)
    results_lower = [None]*len(ξ_grid)

    for i in range(len(ξ_grid)):
        ξ = ξ_grid[i]
        temp = solve(f=f, g=log_Rw, z0=z0, z1=z1, 
                     ξ=ξ, tol=1e-9, max_iter=1000)
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
        temp = solve(f=f, g=log_Rw, z0=z0, z1=z1, 
                     ξ=ξ, tol=1e-9, max_iter=1000)        
        results_lower[i] = temp
        temp = solve(f=f, g=-log_Rw, z0=z0, z1=z1, 
                     ξ=ξ, tol=1e-9, max_iter=1000)          
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


def print_results(result_lower, result_upper):
    n_states = result_lower['P'].shape[0]
    # Print iteration information
    print("--- Iteration information ---")
    print("Number of iterations (lower bound problem): %s" % (result_lower['count']))
    print("Number of iterations (upper bound problem): %s" % (result_upper['count']))

    # Print converged parameter results
    print("\n")
    print("--- Converged values for the lower bound problem ---")
    print("ϵ: %s" % np.round(result_lower['ϵ'],2))
    print("e: %s" % np.round(result_lower['e'],2))
    print("λ: %s" % np.round(result_lower['λ'],2))

    print(" ")
    print("--- Converged values for the upper bound problem ---")
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