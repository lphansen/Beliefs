import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utilities import *

def entropy_moment_bounds():
    # Solve the minimization problems over a grid of ξ
    tol = 2e-10
    max_iter = 1000
    
    # Initialize solver
    solver = InterDivConstraint(tol,max_iter)

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
    REs_upper = np.array([result['RE'] for result in results_upper])
    moment_bounds_cond_lower = np.array([result['moment_bound_cond'] for result in results_lower])
    moment_bounds_cond_upper = np.array([result['moment_bound_cond'] for result in results_upper])
    moment_bounds_lower = np.array([result['moment_bound'] for result in results_lower])
    moment_bounds_upper = np.array([result['moment_bound'] for result in results_upper])
    moment_cond = np.array([result['moment_cond'] for result in results_lower])
    moment = np.array([result['moment'] for result in results_lower])

    # Plots for RE and E[Mg(X)]
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=REs_lower, name='E[MlogM] for lower bound problem', line=dict(color='blue')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=ξ_grid, y=REs_upper, name='E[MlogM] for upper bound problem', line=dict(color='purple')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=np.ones_like(ξ_grid)*REs_lower[-1]*1.1, name='1.1x minimum RE', line=dict(color='black',dash='dash')),
        row=1, col=1
    )


    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_lower, name='E[Mg(X)], lower bound', line=dict(color='green')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_upper, name='E[Mg(X)], upper bound', line=dict(color='red')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment, name='E[g(X)]',line=dict(dash='dash',color='orange')),
        row=1, col=2
    )


    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_cond_lower[:,0], name='E[Mg(X)|state 1], lower bound', visible=False, line=dict(color='green')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_cond_upper[:,0], name='E[Mg(X)|state 1], upper bound', visible=False, line=dict(color='red')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_cond[:,0], name='E[g(X)|state 1]', visible=False, line=dict(dash='dash',color='orange')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_cond_lower[:,1], name='E[Mg(X)|state 2], lower bound', visible=False, line=dict(color='green')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_cond_upper[:,1], name='E[Mg(X)|state 2], upper bound', visible=False, line=dict(color='red')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_cond[:,1], name='E[g(X)|state 2]', visible=False, line=dict(dash='dash',color='orange')),
        row=1, col=2
    )


    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_cond_lower[:,2], name='E[Mg(X)|state 3], lower bound', visible=False, line=dict(color='green')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_bounds_cond_upper[:,2], name='E[Mg(X)|state 3], upper bound', visible=False, line=dict(color='red')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=ξ_grid, y=moment_cond[:,2], name='E[g(X)|state 3]', visible=False, line=dict(dash='dash',color='orange')),
        row=1, col=2
    )


    fig.update_layout(height=400, width=1000, title_text="Relative entropy (left) and moment bounds (right)", showlegend = False)
    fig.update_xaxes(rangemode="tozero",title_text='ξ')
    fig.update_yaxes(rangemode="tozero")

    # fig['layout']['xaxis'+str(int(1))].update(range = (0,0.2))
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
                         args=[{"visible": [True]*6 + [False]*15 }]),
                    dict(label="State 1",
                         method="update",
                         args=[{"visible": [True]*3 + [False]*3 + [True]*3 + [False]*6}]),
                    dict(label="State 2",
                         method="update",
                         args=[{"visible": [True]*3 + [False]*6 + [True]*3 + [False]*3}]),
                    dict(label="State 3",
                         method="update",
                         args=[{"visible": [True]*3 + [False]*9 + [True]*3}]),
                ]),
            )
        ])


    fig.show()
    
def box_chart():
    conditioning = ['low D/P', 'middle D/P', 'high D/P', 'unconditional']
    min_entropy_implied = np.array([2.18, 2.73, 4.80, 2.40])
    empirical_average = np.array([5.12, 3.54, 13.90, 7.54])

    # Each number is repeated once in order to form a desirable shape of boxes
    low_bound_20 = np.array([1.54, 1.54, 2.96, 2.96]) 
    middle_bound_20 = np.array([2.54, 2.54, 2.93, 2.93])
    high_bound_20 = np.array([4.54,4.54,5.10,5.10])
    unconditional_bound_20 = np.array([1.72, 1.72, 3.13, 3.13])

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

    plt.show()
    
    