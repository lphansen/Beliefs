# Load packages
import numpy as np
import pandas as pd
from numba import jit
from scipy.optimize import minimize


@jit
def _objective_numba(f,g,pd_lag_indicator,pd_indicator_float,state,n_f,e,ξ,λ):
    """
    Objective function of the minimization problem. Use numba.jit to boost computational performance.
    """        
    selector = pd_lag_indicator[:,state-1]
    term_1 = -g[selector]/ξ
    term_2 = f[:,(state-1)*n_f:state*n_f][selector]@λ
    term_3 = np.log(pd_indicator_float[selector]@e)
    x = term_1 + term_2 + term_3
    # Use "max trick" to improve accuracy
    a = x.max()
    # log_E_exp(x)
    return np.log(np.sum(np.exp(x-a))) + a    

@jit
def _objective_gradient_numba(f,g,pd_lag_indicator,pd_indicator_float,state,n_f,e,ξ,λ):
    """
    Gradient of the objective function. Use numba.jit to boost computational performance.
    """         
    selector = pd_lag_indicator[:,state-1]
    temp1 = -g[selector]/ξ + f[:,(state-1)*n_f:state*n_f][selector]@λ + np.log(pd_indicator_float[selector]@e)
    temp2 = f[:,(state-1)*n_f:state*n_f][selector]*(np.exp(temp1.reshape((len(temp1),1)))/np.mean(np.exp(temp1)))
    temp3 = np.empty(temp2.shape[1])
    for i in range(temp2.shape[1]):
        temp3[i] = np.mean(temp2[:,i])
    return temp3

class InterDivConstraint:
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
        
        # Placeholder for state, e, ϵ
        self.state = None
        self.e = None
        self.ϵ = None
        
        # Specify dimensions
        self.n_f = 4
        self.n_states = 3
        
        # Specify tolerance levels and maximum iterations for the convex solver
        self.tol = tol
        self.max_iter = max_iter
        
        
    def _objective(self,λ):
        """
        Objective function of the minimization problem.
        """
        if self.lower:
            return _objective_numba(self.f,self.g,self.pd_lag_indicator,self.pd_indicator_float,self.state,self.n_f,self.e,self.ξ,λ)
        else:
            return _objective_numba(self.f,-self.g,self.pd_lag_indicator,self.pd_indicator_float,self.state,self.n_f,self.e,self.ξ,λ)            
    
    def _objective_gradient(self,λ):
        """
        Gradient of the objective function.     
        """
        if self.lower:
            return _objective_gradient_numba(self.f,self.g,self.pd_lag_indicator,self.pd_indicator_float,self.state,self.n_f,self.e,self.ξ,λ)
        else:
            return _objective_gradient_numba(self.f,-self.g,self.pd_lag_indicator,self.pd_indicator_float,self.state,self.n_f,self.e,self.ξ,λ)            
    
    def _min_objective(self):
        """
        Use scipy.minimize (L-BFGS-B, BFGS or CG) to solve the minimization problem.
        """
        for method in ['L-BFGS-B','BFGS','CG']:
            model = minimize(self._objective, 
                             np.ones(self.n_f), 
                             method=method,
                             jac = self._objective_gradient,
                             tol=self.tol,
                             options={'maxiter': self.max_iter})
            if model.success:
                break
        if model.success == False:
            print("---Warning: the convex solver fails when ξ == %s---" % self.ξ)
            print(model.message)
            
        # Calculate v and λ (here λ is of dimension self.n_f)
        v = np.exp(model.fun)/np.sum(self.pd_lag_indicator[:,self.state-1])
        λ = model.x
        return v,λ
    

    def iterate(self,ξ,lower=True):
        """
        Iterate to get staitionary e and ϵ (eigenvector and eigenvalue) for the minimization problem. Here we fix ξ.
        Return a dictionary of variables that are of our interest. 
        """
        # Fix ξ
        self.ξ = ξ
        self.lower = lower
        
        # initial error
        error = 1.
        # count times
        count = 0

        while error > self.tol:
            if count == 0:
                # initial guess for e
                self.e = np.ones(self.n_states)
                # placeholder for v
                v = np.zeros(self.n_states)   
                # placeholder for λ
                λ = np.zeros(self.n_states*self.n_f)
            for k in np.arange(1,self.n_states+1,1):
                self.state = k
                v[self.state-1],λ[(self.state-1)*self.n_f:self.state*self.n_f] = self._min_objective()
            # update e and ϵ
            e_old = self.e
            self.ϵ = v[0]
            self.e = v/v[0]
            error = np.max(np.abs(self.e - e_old))
            count += 1
        
        # Calculate M and E[M|state k]
        if self.lower:
            M = 1./self.ϵ * np.exp(-self.g/self.ξ+self.f@λ) * (self.pd_indicator@self.e) / (self.pd_lag_indicator@self.e)
        else:
            M = 1./self.ϵ * np.exp(self.g/self.ξ+self.f@λ) * (self.pd_indicator@self.e) / (self.pd_lag_indicator@self.e)
        E_M_cond = []
        for i in np.arange(1,self.n_states+1,1):
            temp = np.mean(M[self.pd_lag_indicator[:,i-1]])
            E_M_cond.append(temp)
        E_M_cond = np.array(E_M_cond)
        
        # Calculate transition matrix and staionary distribution under distorted probability
        P_tilde = np.zeros((self.n_states,self.n_states))
        for i in np.arange(1,self.n_states+1,1):
            for j in np.arange(1,self.n_states+1,1):
                P_tilde[i-1,j-1] = np.mean(M[self.pd_lag_indicator[:,i-1]]*self.pd_indicator[self.pd_lag_indicator[:,i-1]][:,j-1]) 
        A = P_tilde.T - np.eye(self.n_states)
        A[-1] = np.ones(self.n_states)
        B = np.zeros(self.n_states)
        B[-1] = 1.
        π_tilde = np.linalg.solve(A, B)
        
        # Calculate transition matrix and stationary distribution under the original empirical probability
        P = np.zeros((self.n_states,self.n_states))
        for i in np.arange(1,self.n_states+1,1):
            for j in np.arange(1,self.n_states+1,1):
                P[i-1,j-1] = np.mean(self.pd_indicator[self.pd_lag_indicator[:,i-1]][:,j-1]) 
        A = P.T - np.eye(self.n_states)
        A[-1] = np.ones(self.n_states)
        B = np.zeros(self.n_states)
        B[-1] = 1.
        π = np.linalg.solve(A, B)
        
        # Calculate conditional/unconditional 
        RE_cond = []
        for i in np.arange(1,self.n_states+1,1):
            temp = np.mean(M[self.pd_lag_indicator[:,i-1]]*np.log(M[self.pd_lag_indicator[:,i-1]]))
            RE_cond.append(temp)
        RE_cond = np.array(RE_cond)
        RE = RE_cond @ π_tilde
        
        # Calculate μ and moment bound
        μ = - self.ξ * np.log(self.ϵ)
        moment_bound_check = μ - self.ξ*RE
        # Conditional moment bounds
        moment_bound_cond = []
        for i in np.arange(1,self.n_states+1,1):
            temp = np.mean(M[self.pd_lag_indicator[:,i-1]]*self.g[self.pd_lag_indicator[:,i-1]])
            moment_bound_cond.append(temp)
        moment_bound_cond = np.array(moment_bound_cond)
        moment_bound = moment_bound_cond @ π_tilde
        
        # Calculate the original conditional/unconditional moment for g(X)
        # Original moment 
        moment_cond = []
        for i in np.arange(1,self.n_states+1,1):
            temp = np.mean(self.g[self.pd_lag_indicator[:,i-1]])
            moment_cond.append(temp)  
        moment = np.mean(self.g)
        
        # Calculate v
        v_0 = -self.ξ * np.log(self.e)
        
        result = {'ϵ':self.ϵ,
                  'e':self.e,
                  'λ':λ,
                  'count':count,
                  'ξ':self.ξ,
                  'μ':μ,
                  'v_0':v_0,
                  'RE_cond':RE_cond,
                  'RE':RE,
                  'E_M_cond':E_M_cond,
                  'P':P,
                  'π':π,
                  'P_tilde':P_tilde,
                  'π_tilde':π_tilde,
                  'moment_bound':moment_bound,
                  'moment_bound_check':moment_bound_check,
                  'moment_bound_cond':moment_bound_cond,
                  'moment_cond':moment_cond,
                  'moment':moment,
                  'M':M}
        
        return result
    
    
    
    