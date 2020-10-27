"""
This module prepares data for the examples used in plots.py and the notebook.

"""
import numpy as np
import pandas as pd


def preprocess_data(n_states):
    """
    Clean data.
    
    Parameters
    ----------
    n_states : int
        Number of states for the d/p ratio. The division is based on
        percentile of the empirical d/p data.
    
    Returns
    -------
    f : (n, n_f) ndarray
        Moment conditions.
    log_Rw, Rf, Rm, SMB, HML : (n,) ndarrays
        Log return and gross returns.
    z0, z1 : (n, n_states) ndarrays
        Today's and tomorrow's state vectors.

    """
    # Load data
    data = pd.read_csv('./data/UnitaryData.csv')
    pd_lag = np.array(data['d.p'])

    # Specify dimensions
    n_fos = 4

    # Calculate indicator based on today's pd ratio
    z0_float = np.empty((data.shape[0], n_states))
    tercile = np.quantile(pd_lag, np.arange(n_states + 1)/n_states)
    for i in range(n_states):
        z0_float[:,i] = (pd_lag >= tercile[i]) & (pd_lag <= tercile[i+1])
    z0 = z0_float.astype(bool)

    # Calculate indicator for tomorrow's pd ratio
    z1 = z0[1:]

    # Drop last row since we do not have tomorrow's pd ratio at that point
    z0 = z0[:-1]
    x = np.array(data[['Rf','Rm-Rf','SMB','HML']])[:-1]
    f = np.empty((x.shape[0], x.shape[1] * n_states))
    for state in range(n_states):
        f[:,(n_fos * state):(n_fos * (state+1))] = x * z0[:, state:(state+1)]
    log_Rw = np.array(data['log.RW'])[:-1]
    
    # Compute risk free rate and excess returns
    Rf = (x[:,0]+1.)*np.exp(log_Rw)
    Rm = x[:,1]*np.exp(log_Rw) + Rf
    SMB = x[:,2]*np.exp(log_Rw) + Rf
    HML = x[:,3]*np.exp(log_Rw) + Rf

    return f, log_Rw, z0, z1, Rf, Rm, SMB, HML
