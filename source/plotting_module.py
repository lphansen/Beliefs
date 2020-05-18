import ipywidgets as widgets
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import interact, interactive, fixed, interact_manual
from source.utilities import *

def objective_vs_ξ(n_states):
    # Solve the minimization problems over a grid of ξ
    tol = 2e-10
    max_iter = 1000
    
    # Initialize solver
    solver = InterDivConstraint(n_states,tol,max_iter)

    # Define g(X) = log Rw
    solver.g = solver.log_Rw
    
    # Grid for ξ
    ξ_grid = np.arange(.01,1.01,.005)

    results_lower = [None]*len(ξ_grid)
    
    for i in range(len(ξ_grid)):
        ξ = ξ_grid[i]
        temp = solver.iterate(ξ,lower=True)
        results_lower[i] = temp

    μs_lower = np.array([result['μ'] for result in results_lower])
    ϵs_lower = np.array([result['ϵ'] for result in results_lower])

    # Plots
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
#     fig['layout']['yaxis'+str(int(1))].update(range = (0.,0.06))
    fig['layout']['xaxis'+str(int(2))].update(range = (0.,1.))
#     fig['layout']['yaxis'+str(int(2))].update(range = (-0.01,0.04))

    fig.show()
    
    
def entropy_moment_bounds(n_states):
    # Solve the minimization problems over a grid of ξ
    tol = 2e-10
    max_iter = 1000
    
    # Initialize solver
    solver = InterDivConstraint(n_states,tol,max_iter)

    # Define g(X) = log Rw
    solver.g = solver.log_Rw
    
    # Grid for ξ
    ξ_grid = np.arange(.01,1.01,.01)

    results_lower = [None]*len(ξ_grid)
    results_upper = [None]*len(ξ_grid)
    
    for i in range(len(ξ_grid)):
        ξ = ξ_grid[i]
        temp = solver.iterate(ξ,lower=True)
        results_lower[i] = temp
        temp = solver.iterate(ξ,lower=False)
        results_upper[i] = temp

    REs_lower = np.array([result['RE'] for result in results_lower])
    moment_bounds_cond_lower = np.array([result['moment_bound_cond'] for result in results_lower])
    moment_bounds_cond_upper = np.array([result['moment_bound_cond'] for result in results_upper])
    moment_bounds_lower = np.array([result['moment_bound'] for result in results_lower])
    moment_bounds_upper = np.array([result['moment_bound'] for result in results_upper])
    moment_cond = np.array([result['moment_cond'] for result in results_lower])
    moment = np.array([result['moment'] for result in results_lower])

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
    
def box_chart(result_min,result_lower,result_upper,save=False,dpi=1200):
    conditioning = ['low D/P', 'middle D/P', 'high D/P', 'unconditional']
    min_entropy_implied = np.array([result_min['moment_bound_cond'][0]*400, result_min['moment_bound_cond'][1]*400, result_min['moment_bound_cond'][2]*400, result_min['moment_bound']*400])
    empirical_average = np.array([result_min['moment_cond'][0]*400, result_min['moment_cond'][1]*400, result_min['moment_cond'][2]*400, result_min['moment']*400])

    # Each number is repeated once in order to form a desirable shape of boxes
    low_bound_20 = np.array([result_lower['moment_bound_cond'][0]*400, result_lower['moment_bound_cond'][0]*400, result_upper['moment_bound_cond'][0]*400, result_upper['moment_bound_cond'][0]*400]) 
    middle_bound_20 = np.array([result_lower['moment_bound_cond'][1]*400, result_lower['moment_bound_cond'][1]*400, result_upper['moment_bound_cond'][1]*400, result_upper['moment_bound_cond'][1]*400])
    high_bound_20 = np.array([result_lower['moment_bound_cond'][2]*400,result_lower['moment_bound_cond'][2]*400,result_upper['moment_bound_cond'][2]*400,result_upper['moment_bound_cond'][2]*400])
    unconditional_bound_20 = np.array([result_lower['moment_bound']*400, result_lower['moment_bound']*400, result_upper['moment_bound']*400, result_upper['moment_bound']*400])

    fig, ax = plt.subplots()
    bplot = ax.boxplot(np.vstack((low_bound_20, middle_bound_20, high_bound_20, unconditional_bound_20)).T,
                       usermedians = min_entropy_implied, #override the automatically calculated median
                       labels = conditioning, widths = 0.4, 
                       medianprops = dict(linestyle='-.', linewidth = 0, color = 'black'),
                       patch_artist = True)

    for box in bplot['boxes']:
        box.set(facecolor = 'salmon')

    splot = ax.scatter(x = np.arange(1, 5), y = empirical_average, c = 'black')

    ax.axvline(x = 3.5, color = 'black', linestyle = '--', linewidth = 1)
    
    ax.set_ylim(0.,14.5)
    
    plt.show()
    
    if save:
        fig.savefig("box_20%.png",dpi=dpi)

        
# ξs that correspond to 0%-30% higher min RE (g(X)=log Rw), step size=5%
ξs_lower = np.array([100,
                     0.2925539016723633,
                     0.2051076889038086,
                     0.16624605655670166,
                     0.14301961660385132,
                     0.12713348865509033,
                     0.11538618803024292])
ξs_upper = np.array([100,
                     0.2989816665649414,
                     0.21141958236694336,
                     0.17248201370239258,
                     0.14917802810668945,
                     0.1332303285598755,
                     0.12140893936157227])

# ζs that correspond to 0%-30% higher min RE (risk premia), step size=5%
ζs_lower = np.array([1.,
                     1.007,
                     1.006,
                     1.006,
                     1.006,
                     1.006,
                     1.008])
ζs_upper = np.array([1.,
                     1.007,
                     1.007,
                     1.008,
                     1.007,
                     1.008,
                     1.006])
