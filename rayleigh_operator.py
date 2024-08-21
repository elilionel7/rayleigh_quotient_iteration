# rayleigh_operator.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'SEEM_Chebyshev_master'))

import numpy as np
from operator_data import operator_data
from scipy.fftpack import dct as dct
from scipy.fftpack import idct as idct
from scipy.fftpack import dctn as dctn
from scipy.fftpack import idctn as idctn
from scipy.fftpack import dst as dst
from scipy.fftpack import idst as idst
from scipy.sparse.linalg import LinearOperator
import scipy


class RayleighOperator(operator_data):
    

    def qr_solve3(self,rhs1,l):
        
        rhs2 = np.zeros_like(self.gdata.b[:,0])
    
        rhs = np.hstack((rhs1,rhs2))
        print('i worked')
        rct = self.make_rct_matrix(l)
        print('i did not worked')
        cond = np.linalg.cond(rct)
        Q,R = np.linalg.qr(rct)
        u = scipy.linalg.solve_triangular(R.T,rhs,lower=True)
        u = np.dot(Q,u)
        u = np.reshape(u,(self.gdata.m,self.gdata.m))
        u = self.ker(u,l).flatten() 
        return u,cond
    
    def rayleigh_quotient_iteration(self,tol=1e-8, max_iter=100):
        
        rhs1 = np.ones_like(self.gdata.flag)[0] #pde right hand side
        rhs1 =  rhs1 / np.linalg.norm(rhs1)
        
        u = np.ones(self.gdata.m**2)  
        u = u / np.linalg.norm(u)  
        
        for iteration in range(max_iter):
            u_new, cond = self.qr_solve3(rhs1, self.l)
            
            
            u_new = u_new / np.linalg.norm(u_new) 
            u_new = u_new[self.gdata.flag]
            new_rhs1 = u_new

            if np.linalg.norm(new_rhs1 - rhs1) < tol:
                return u_new, cond, iteration + 1
            
            rhs1 = new_rhs1
        
        return rhs1, cond, max_iter

