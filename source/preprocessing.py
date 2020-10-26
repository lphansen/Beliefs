import numpy as np
import pandas as pd


def preprocess_data():
    """
    Load datasets and initialize the solver.
    
    """        
    # Load data
    data = pd.read_csv('./data/UnitaryData.csv')
    pd_lag = np.array(data['d.p'])

    # Specify dimensions
    n_fos = 4
    n_states = 3

    # Calculate indicator based on today's pd ratio
    pd_lag_indicator_float = np.empty((data.shape[0], n_states))
    tercile = np.quantile(pd_lag, np.arange(n_states + 1)/n_states)
    for i in range(n_states):
        pd_lag_indicator_float[:,i] = (pd_lag >= tercile[i]) & (pd_lag <= tercile[i+1])
    pd_lag_indicator = pd_lag_indicator_float.astype(bool)

    # Calculate indicator for tomorrow's pd ratio
    pd_indicator = pd_lag_indicator[1:]
    pd_indicator_float = pd_indicator.astype(float)

    # Drop last row since we do not have tomorrow's pd ratio at that point
    pd_lag_indicator = pd_lag_indicator[:-1]
    X = np.array(data[['Rf','Rm-Rf','SMB','HML']])[:-1]
    f = np.empty((X.shape[0], X.shape[1] * n_states))
    for state in range(n_states):
        f[:,(n_fos * state):(n_fos * (state+1))] = X * pd_lag_indicator[:, state:(state+1)]
    log_Rw = np.array(data['log.RW'])[:-1]

    return f, log_Rw