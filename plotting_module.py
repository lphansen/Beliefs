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
        go.Scatter(x=ξ_grid, y=np.ones_like(ξ_grid)*REs_lower[-1]*1.2, name='1.2x min RE', line=dict(color='black',dash='dash')),
        row=1, col=1
    )    
    
    fig.add_trace(
        go.Scatter(x=ξ_grid, y=REs_lower, name='lower bound', line=dict(color='blue')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=ξ_grid, y=REs_upper, name='upper bound', line=dict(color='purple')),
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
    
def box_chart(result_min,result_lower,result_upper):
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
    

# ξs that correspond to 0%-25% higher min RE (g(X)=log Rw)
ξs_lower = np.array([100,
                     0.660308837890625,
                     0.465484619140625,
                     0.3792381286621094,
                     0.3277549743652344,
                     0.2925539016723633,
                     0.2665672302246094,
                     0.24636244773864746,
                     0.23006033897399902,
                     0.21654391288757324,
                     0.2051076889038086,
                     0.19526314735412598,
                     0.18666887283325195,
                     0.1790839433670044,
                     0.17232286930084229,
                     0.16624605655670166,
                     0.16074597835540771,
                     0.15573608875274658,
                     0.15114754438400269,
                     0.14692401885986328,
                     0.14301961660385132,
                     0.13939595222473145,
                     0.13602107763290405,
                     0.1328662633895874,
                     0.12991082668304443,
                     0.12713348865509033])
ξs_upper = np.array([100,
                     0.6666183471679688,
                     0.4721088409423828,
                     0.3856649398803711,
                     0.3341245651245117,
                     0.2989816665649414,
                     0.2729339599609375,
                     0.25270676612854004,
                     0.2364063262939453,
                     0.22286725044250488,
                     0.21141958236694336,
                     0.20155417919158936,
                     0.19294500350952148,
                     0.18534398078918457,
                     0.17857110500335693,
                     0.17248201370239258,
                     0.16696691513061523,
                     0.1619412899017334,
                     0.15733754634857178,
                     0.15309679508209229,
                     0.14917802810668945,
                     0.14554333686828613,
                     0.14215445518493652,
                     0.1389867067337036,
                     0.13601797819137573,
                     0.1332303285598755])