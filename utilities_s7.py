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
        
        
    def _objective_approach_one(self,multipliers):
        """
        Objective function of the minimization problem in approach one.
        """    
        ξ = multipliers[0]
        λ = multipliers[1:] 
        
        term_1 = self.g
        term_2 = self.f@λ
        x = (term_1 + term_2)/(-ξ)
        # Use "max trick" to improve accuracy
        a = x.max()
        # log_E_exp(x)
        log_E_exp_x = np.log(np.sum(np.exp(x-a))) + a - np.log(self.f.shape[0])
        return (log_E_exp_x+self.k)*ξ
    
    
    def _objective_approach_two(self,multipliers):
        """
        Objective function of the minimization problem in approach two.
        """  
        ξ = multipliers[0]
        λ = multipliers[1:]
        self.state = 1
        multipliers_1 = np.insert(multipliers[(self.state-1)*self.n_f+1:self.state*self.n_f+1],0,ξ,axis=0)
        temp_1 = self._objective_approach_three(multipliers_1)
        E_B_1 = np.mean(self.pd_lag_indicator[:,self.state-1])
        result = temp_1 * E_B_1
        for state in [2,3]:
            self.state = state
            multipliers_new = np.insert(multipliers[(self.state-1)*self.n_f+1:self.state*self.n_f+1],0,ξ,axis=0)
            temp = self._objective_approach_three(multipliers_new)
            E_B = np.mean(self.pd_lag_indicator[:,self.state-1])
            result += temp * E_B
        
        return result       
            
    
    def _objective_approach_three(self,multipliers):
        """
        Objective function of the minimization problem in approach three.
        """
        ξ = multipliers[0]
        λ = multipliers[1:]
        
        selector = self.pd_lag_indicator[:,self.state-1]
        term_1 = self.g[selector]
        term_2 = (self.f[:,(self.state-1)*self.n_f:self.state*self.n_f][selector])@λ
        x = (term_1 + term_2)/(-ξ)
        # Use "max trick" to improve accuracy
        a = x.max()
        # log_E_exp(x)
        log_E_exp_x = np.log(np.sum(np.exp(x-a))) + a - np.log(np.sum(selector))
        return (log_E_exp_x+self.k)*ξ
        
    
    def _objective_min_k_approach_one(self,λ):
        """
        Objective function for finding the minimum value of k in approach one.
        """    
        x = -self.f@λ
        # Use "max trick" to improve accuracy
        a = x.max()
        # log_E_exp(x)
        log_E_exp_x = np.log(np.sum(np.exp(x-a))) + a - np.log(self.f.shape[0])
        return log_E_exp_x
    
    
    def _objective_min_k_approach_three(self,λ):
        """
        Objective function for finding the minimum value of k in approach three.
        """
        selector = self.pd_lag_indicator[:,self.state-1]
        x = -(self.f[:,(self.state-1)*self.n_f:self.state*self.n_f][selector])@λ
        # Use "max trick" to improve accuracy
        a = x.max()
        # log_E_exp(x)
        log_E_exp_x = np.log(np.sum(np.exp(x-a))) + a - np.log(np.sum(selector))
        return log_E_exp_x
    
    
    def cal_min_k(self,approach=1,state=None):
        """
        Use scipy.minimize (L-BFGS-B, BFGS, CG) to solve the minimization problem.
        """
        if approach == 1:
            objective = self._objective_min_k_approach_one
            initial_point = np.ones(self.n_f*self.n_states)          
        elif approach == 2:
            objective = self._objective_min_k_approach_two
            initial_point = np.ones(self.n_f*self.n_states)
        elif approach == 3:
            if state not in [1,2,3]:
                raise Exception('Please correctly specify state in approach 3. It should be 1,2 or 3.')
            else:
                self.state = state
            objective = self._objective_min_k_approach_three
            initial_point = np.ones(self.n_f)
        else:
            raise Exception('Approach should be 1,2 or 3. The specified approach was: {}'.format(approach))
            
        for method in ['L-BFGS-B','BFGS','CG']:
            model = minimize(objective, 
                             initial_point,
                             method=method,
                             tol=self.tol,
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
               'λ':model.x}
        
        return result        
        
    def solve(self,k,approach=1,state=None):
        """
        Use scipy.minimize (L-BFGS-B, SLSQP) to solve the minimization problem.
        """
        self.k = k
        if approach == 1:
            objective = self._objective_approach_one
            initial_point = np.ones(self.n_f*self.n_states+1)          
        elif approach == 2:
            objective = self._objective_approach_two
            initial_point = np.ones(self.n_f*self.n_states+1)
        elif approach == 3:
            if state not in [1,2,3]:
                raise Exception('Please correctly specify state in approach 3. It should be 1,2 or 3.')
            else:
                self.state = state
            objective = self._objective_approach_three
            initial_point = np.ones(self.n_f+1)
        else:
            raise Exception('Approach should be 1,2 or 3. The specified approach was: {}'.format(approach))
            
        bounds = []
        for i in range(len(initial_point)):
            # Set first variable to be ξ
            if i==0:
                bounds.append((1e-16,None))
            else:
                bounds.append((None,None))
            
        for method in ['L-BFGS-B','SLSQP']:
            model = minimize(objective, 
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
               'λ':model.x[1:]}
        
        return result
        
        
    