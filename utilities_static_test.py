# Load packages
import numpy as np
import pandas as pd
from numba import jit
from scipy.optimize import minimize


class StaDivConstraint:
    def __init__(self,tol=1e-8,max_iter=1000):
        """
        Load datasets and initialize the solver.
        """        
        # Load data
        data = pd.read_csv('UnitaryData.csv')
        pd_lag = np.array(data['d.p'])
        
        # Calculate terciles for pd ratio
        tercile_1 = np.quantile(pd_lag,1./3)
        tercile_2 = np.quantile(pd_lag,2./3)

        # Calculate indicator based on today's pd ratio
        pd_lag_indicator = np.array([pd_lag <= tercile_1,(pd_lag <= tercile_2) & (pd_lag > tercile_1),pd_lag > tercile_2]).T

        # Calculate indicator for tomorrow's pd ratio
        self.pd_indicator = pd_lag_indicator[1:]
        self.pd_indicator_float = self.pd_indicator.astype(float)
        
        # Drop last row since we do not have tomorrow's pd ratio at that point
        self.pd_lag_indicator = pd_lag_indicator[:-1]
        X = np.array(data[['Rf','Rm-Rf','SMB','HML']])[:-1]
        self.f = np.hstack((X * self.pd_lag_indicator[:,:1],X * self.pd_lag_indicator[:,1:2],X * self.pd_lag_indicator[:,2:3]))
        self.g = np.array(data['log.RW'])[:-1]        
        
        # Placeholder for state, k
        self.state = None
        self.k = None
        
        # Specify dimensions
        self.n_f = 4
        self.n_states = 3
        
        # Specify tolerance levels and maximum iterations for the convex solver
        self.tol = tol
        self.max_iter = max_iter
        
        
    def _objective(self,multipliers):
        """
        Objective function of the minimization problem in approach one.
        """    
        ξ = multipliers[0]
        λ_1 = multipliers[1:-1] 
        λ_2 = multipliers[-1] 
        
        if self.lower:
            term_1 = self.g
        else:
            term_1 = -self.g
        term_2 = self.f@λ_1
        term_3 = λ_2
        x = (term_1 + term_2 + term_3)/(-ξ) + 0.5
        x[x<0]=0
        result = np.mean(x**2)*(ξ/2.) + ξ*self.k + λ_2
        return result
        
        
    def solve(self,k,lower=True):
        """
        Use scipy.minimize (L-BFGS-B, SLSQP) to solve the minimization problem.
        """
        self.k = k
        self.lower = lower
        
        initial_point = np.ones(self.n_f*self.n_states+2)
        
        bounds = []
        for i in range(len(initial_point)):
            # Set first variable to be ξ
            if i==0:
                bounds.append((1e-16,None))
            else:
                bounds.append((None,None))
            
        for method in ['L-BFGS-B','SLSQP']:
            model = minimize(self._objective, 
                             initial_point,
                             method=method,
                             tol=self.tol,
                             bounds=bounds,
                             options={'maxiter': self.max_iter})
            if model.success:
                break
                
        if model.success == False:
            print("---Warning: the convex solver fails---")
            print(model.message)
            
        # Save optimization status
        result = {'result':-model.fun,
               'success':model.success,
               'message':model.message,
               'nit':model.nit,
               'ξ':model.x[0],
               'λ_1':model.x[1:-1],
               'λ_2':model.x[-1]}
        
        # Calculate M
        if self.lower:
            term_1 = self.g
            term_2 = self.f@result['λ_1']
            term_3 = result['λ_2']
            M = -1./result['ξ'] * (term_1+term_2+term_3) + 0.5
            M[M<0]=0
        else:
            term_1 = -self.g
            term_2 = self.f@result['λ_1']
            term_3 = result['λ_2']
            M = -1./result['ξ'] * (term_1+term_2+term_3) + 0.5
            M[M<0]=0

        # Calculate empirical probability
        π = np.zeros(self.n_states)
        for i in np.arange(1,self.n_states+1,1):
            π[i-1] = np.mean(self.pd_lag_indicator[:,i-1])

        # Calculate distorted probability
        π_tilde = np.zeros_like(π)
        for i in np.arange(1,self.n_states+1,1):
            π_tilde[i-1] = np.mean(M * self.pd_lag_indicator[:,i-1])

        # Calculate conditional/unconditional moment bound
        moment_bound_cond = []
        for i in np.arange(1,self.n_states+1,1):
            temp = np.mean(M*self.g*self.pd_lag_indicator[:,i-1]) / np.mean(M*self.pd_lag_indicator[:,i-1])
            moment_bound_cond.append(temp)
        moment_bound_cond = np.array(moment_bound_cond)
        moment_bound = np.mean(M*self.g) 

        # Calculate the original conditional/unconditional moment for g(X)
        # Original moment 
        moment_cond = []
        for i in np.arange(1,self.n_states+1,1):
            temp = np.mean(self.g[self.pd_lag_indicator[:,i-1]])
            moment_cond.append(temp)  
        moment = np.mean(self.g)

        return {'π':π,
                'π_tilde':π_tilde,
                'moment_bound_cond':moment_bound_cond,
                'moment_bound':moment_bound,
                'moment_cond':moment_cond,
                'moment':moment,
                'M':M}
