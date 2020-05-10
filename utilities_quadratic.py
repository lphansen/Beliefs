import numpy as np
import pandas as pd
from numba import jit
from scipy.optimize import minimize



@jit
def _objective_numba(f,g,pd_lag_indicator,pd_indicator_float,state,n_f,v,ξ,λ):
    """
    Use -(μ + v_0) as the objective function. Use numba.jit to boost computational performance.
    """
    λ_1 = λ[:-1]
    λ_2 = λ[-1]
    selector = pd_lag_indicator[:,state-1]
    term_1 = g[selector]
    term_2 = pd_indicator_float[selector]@v
    term_3 = f[:,(state-1)*n_f:state*n_f][selector]@λ_1
    term_4 = λ_2
    x = (term_1+term_2+term_3+term_4)/(-ξ) + 0.5
    x[x<0] = 0
    result = np.mean(x**2)*(ξ/2.) + λ_2
    return result


'''
Solver for the intertemporal divergence problem. Here we use quadratic divergence as the measure of divergence.
'''
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
        self.X = np.array(data[['Rf','Rm-Rf','SMB','HML']])[:-1]
        self.f = np.hstack((self.X * self.pd_lag_indicator[:,:1],self.X * self.pd_lag_indicator[:,1:2],self.X * self.pd_lag_indicator[:,2:3]))
        self.log_Rw = np.array(data['log.RW'])[:-1] 
        
        # Placeholder for g, state, v, μ
        self.g = None
        self.state = None
        self.v = None
        self.μ = None
        
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
            return _objective_numba(self.f,self.g,self.pd_lag_indicator,self.pd_indicator_float,self.state,self.n_f,self.v,self.ξ,λ)
        else:
            return _objective_numba(self.f,-self.g,self.pd_lag_indicator,self.pd_indicator_float,self.state,self.n_f,self.v,self.ξ,λ)            
    
    def _min_objective(self):
        """
        Use scipy.minimize (L-BFGS-B, BFGS, CG or SLSQP) to solve the minimization problem.
        """
        for method in ['L-BFGS-B','BFGS','CG','SLSQP']:
#         for method in ['SLSQP']:
            model = minimize(self._objective, 
                             np.ones(self.n_f+1), 
                             method = method,
                             tol = self.tol,
                             options = {'maxiter': self.max_iter})
            if model.success:
                break
        if model.success == False:
            print("---Warning: desired error not achieved when ξ = %s, tolerance = %s--- " % (self.ξ,self.tol))
            print(model.message)
            
        # Calculate v+μ, λ_1 and λ_2 (here λ_1 is of dimension self.n_f)
        v_μ = - model.fun
        λ_1 = model.x[:-1]
        λ_2 = model.x[-1]
        return v_μ,λ_1,λ_2
    
    def iterate(self,ξ,lower=True):
        """
        Iterate to get staitionary e and ϵ (eigenvector and eigenvalue) for the minimization problem. Here we fix ξ.
        Return a dictionary of variables that are of our interest. 
        """
        # Check if self.g is defined or not
        if self.g is None:
            raise Exception("Sorry, please define self.g first!")            
        
        # Fix ξ
        self.ξ = ξ
        self.lower = lower
        
        # Initial error
        error = 1.
        # Count iteration times
        count = 0

        while error > self.tol:
            if count == 0:
                # Initial guess for v
                self.v = np.zeros(self.n_states)
                # Placeholder for v+μ
                v_μ = np.zeros(self.n_states)   
                # Placeholder for λ_1, λ_2
                λ_1 = np.zeros(self.n_states*self.n_f)
                λ_2 = np.zeros(self.n_states)
            for k in np.arange(1,self.n_states+1,1):
                self.state = k
                v_μ[self.state-1],λ_1[(self.state-1)*self.n_f:self.state*self.n_f],λ_2[self.state-1] = self._min_objective()
            # Update v and μ, fix v[0] = 0
            v_old = self.v
            self.μ = v_μ[0]
            self.v = v_μ - v_μ[0]
            error = np.max(np.abs(self.v - v_old))
            count += 1
        # Calculate N and E[N|state k]
        if self.lower:
            term_1 = self.g
        else:
            term_1 = -self.g
        term_2 = self.pd_indicator@self.v
        term_3 = self.f@λ_1
        term_4 = self.pd_lag_indicator@λ_2
        N = -1./self.ξ*(term_1+term_2+term_3+term_4) + 0.5
        N[N<0]=0
        E_N_cond = []
        for i in np.arange(1,self.n_states+1,1):
            temp = np.mean(N[self.pd_lag_indicator[:,i-1]])
            E_N_cond.append(temp)
        E_N_cond = np.array(E_N_cond)
        
        # Calculate transition matrix and staionary distribution under distorted probability
        P_tilde = np.zeros((self.n_states,self.n_states))
        for i in np.arange(1,self.n_states+1,1):
            for j in np.arange(1,self.n_states+1,1):
                P_tilde[i-1,j-1] = np.mean(N[self.pd_lag_indicator[:,i-1]]*self.pd_indicator[self.pd_lag_indicator[:,i-1]][:,j-1]) 
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
        
        # Conditional/unconditional quadratic divergence
        QD_cond = []
        for i in np.arange(1,self.n_states+1,1):
            temp = np.mean(N[self.pd_lag_indicator[:,i-1]]**2 - N[self.pd_lag_indicator[:,i-1]]) * 0.5
            QD_cond.append(temp)
        QD_cond = np.array(QD_cond)
        QD = QD_cond @ π_tilde        
        
        # Calculate unconditional moment bound using μ
        moment_bound_check = self.μ - self.ξ*QD   
        
        # Conditional/unconditional moment bounds
        moment_bound_cond = []
        for i in np.arange(1,self.n_states+1,1):
            temp = np.mean(N[self.pd_lag_indicator[:,i-1]]*self.g[self.pd_lag_indicator[:,i-1]])
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
        
        result = {'λ_1':λ_1,
                  'λ_2':λ_2,
                  'count':count,
                  'ξ':self.ξ,
                  'μ':self.μ,
                  'v':self.v,
                  'E_N_cond':E_N_cond,
                  'P':P,
                  'π':π,
                  'P_tilde':P_tilde,
                  'π_tilde':π_tilde,
                  'QD_cond':QD_cond,
                  'QD':QD,
                  'moment_bound_check':moment_bound_check,
                  'moment_bound':moment_bound,
                  'moment_bound_cond':moment_bound_cond,
                  'moment_cond':moment_cond,
                  'moment':moment,
                  'N':N}
        
        return result
    
    
    def find_ξ(self,x_min_QD,lower,tol=1e-7,max_iter=100):
        """
        This function will use bisection method to find the ξ that corresponds to x times the minimal QD.
        """
        # Get minimal QD
        result = self.iterate(10,lower)
        min_QD = result['QD']
        
        # Start iteration
        count = 0
        for i in range(max_iter):
            # Get QD at current choice of ξ
            if i == 0:
                ξ = 1.
                # Set lower/upper bounds for ξ
                lower_bound = 0.
                upper_bound = 10.
                
            result = self.iterate(ξ,lower)
            QD = result['QD']
            
            # Compare to the level we want
            error = QD/min_QD-x_min_QD
            if np.abs(error)<tol:
                break
            else:
                if error < 0.:
                    upper_bound = ξ
                    ξ = (lower_bound + ξ)/2.
                else:
                    lower_bound = ξ
                    ξ = (ξ + upper_bound)/2.
            
            count += 1
            if count == max_iter:
                print('Maximal iterations reached. Error = %s' % (QD/min_QD-x_min_QD))
        
        return ξ
        
        
        
