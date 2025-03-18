# rayleigh_operator.py

import numpy as np
from scipy.fftpack import dct as dct
from scipy.fftpack import idct as idct
from scipy.fftpack import dctn as dctn
from scipy.fftpack import idctn as idctn
from scipy.fftpack import dst as dst
from scipy.fftpack import idst as idst
from scipy.sparse.linalg import LinearOperator
from scipy.interpolate import griddata

import scipy

class RayleighOperator:
    def __init__(self, gdata, l, precond=False):
        self.gdata = gdata
        self.l = l
        self.MM = LinearOperator((gdata.k + gdata.p, gdata.k + gdata.p), matvec=self.M)
        if precond:
            self.PP = LinearOperator((gdata.k + gdata.p, gdata.k + gdata.p), matvec=self.precond)

    def ker(self, w, l=None):
        if l is None:
            l = self.l
        w = np.reshape(w, (self.gdata.m, self.gdata.m))
        w = np.real(idctn(dctn(w) * (1 + self.gdata.fx**2 + self.gdata.fy**2)**-l)) / (2 * self.gdata.m)**2
        return w.flatten()
    def lap(self,w):
        w1 = np.copy(w)
        w1 = dct(w1,axis=0)*-self.gdata.fx
        w1 = np.roll(w1,-1,axis=0)
        w1 = idst(w1,axis=0)/(2*self.gdata.m)
        w1 = w1 * self.gdata.x1/(1-self.gdata.x1**2)**(3/2)
        w2 = np.copy(w)
        w2 = idct(dct(w2,axis=0)*-self.gdata.fx**2,axis=0)/(2*self.gdata.m)
        w2 = w2 * -1/(1-self.gdata.x1**2)
        w = w2
        w = w1 + w2
        return w

    def lapt(self,w):
        w1 = np.copy(w)
        w1 = w1 * self.gdata.x1/(1-self.gdata.x1**2)**(3/2)
        w1 = dst(w1,axis=0)/(2*(self.gdata.m))
        w1 = np.roll(w1,1,axis=0)
        w1 = idct(w1*-self.gdata.fx,axis=0)
        w2 = np.copy(w)
        w2 = w2 * -1/(1-self.gdata.x1**2)
        w2 = dct(w2,axis=0)/(2*(self.gdata.m))
        w2 = w2 * -self.gdata.fx**2
        w2 = idct(w2,axis=0)
        z = w1 + w2
        return z
    # def C(self, w):
    #     w_grid = np.reshape(w, (self.gdata.m, self.gdata.m))
    #     lap_w = np.reshape(self.lap(w_grid.flatten()), (self.gdata.m, self.gdata.m))
    #     lap_w += np.reshape(self.lap(w_grid.T.flatten()), (self.gdata.m, self.gdata.m)).T
    #     lap_w[~self.gdata.flag] = 0  # Dirichlet boundary
    #     boundary_eval = self.gdata.xx @ w.flatten()
    #     return np.hstack((lap_w[self.gdata.flag], boundary_eval))
    
    def C(self,w):
    #    Evaluate at b.
        b = self.gdata.xx.dot(w)
    #    Take Chebyshev derivative
        w = np.reshape(w,(self.gdata.m,self.gdata.m))
        z = self.lap(w) + np.transpose(self.lap(np.transpose(w)))
        z[~self.gdata.flag] = 0
        z = z[self.gdata.flag]
        return np.hstack((z,b))
    

    def Ct_shift(self, w, shift):
        z = np.zeros((self.gdata.m, self.gdata.m))
        z[self.gdata.flag] = w[:self.gdata.k]
        z = self.lapt(z) + self.lapt(z.T).T - shift * z
        z = z.flatten()
        z += self.gdata.xxT.dot(w[-self.gdata.p:])
        return z
    
    def Ct(self,w):
        z = np.zeros((self.gdata.m,self.gdata.m))
        z[self.gdata.flag] = w[:self.gdata.k]
        z = self.lapt(z) + self.lapt(z.T).T
        z = z.flatten()
        z += self.gdata.xxT.dot(w[-self.gdata.p:])
#        
        return z

    def precond(self, w):
        w1 = w[:self.gdata.k]
        if self.l == 3:
            w1 += self.gdata.FD(w1)
        elif self.l == 4:
            w1 += self.gdata.FD(2*w1 + self.gdata.FD(w1))
        elif self.l == 5:
            w1 += self.gdata.FD(3*w1 + self.gdata.FD(3*w1 + self.gdata.FD(w1)))
        w2 = scipy.linalg.lu_solve(self.gdata.B, w[-self.gdata.p:])
        return np.hstack((w1, w2))

    def M(self,w):
        w = self.Ct(w)
        w = self.ker(w)
        w = self.C(w)
        return w

    def make_rct_matrix_shift(self, l, shift):
        m = np.zeros((self.gdata.m**2, self.gdata.k + self.gdata.p))
        for i in range(self.gdata.k + self.gdata.p):
            z = np.zeros(self.gdata.k + self.gdata.p)
            z[i] = 1
            m[:, i] = self.ker(self.Ct_shift(z, shift), l).flatten()
        return m 

    def a_u(self, u):
        Au_full_grid = self.lap(u.reshape(self.gdata.x1.shape))
        return Au_full_grid.flatten()

    def rayleigh_quotient(self, u, Au):
        return np.dot(u, Au) / np.dot(u, u)
    
    def interpolate_solution(self, x, y, sols):
        x = x.flatten()
        y = y.flatten()
        values = sols.flatten()
        pnts = np.column_stack((x, y))

        def interp_func(xi, yi):
            xi = np.asarray(xi).flatten()
            yi = np.asarray(yi).flatten()
            zi = griddata(pnts, values, (xi, yi), method="cubic")
            return zi

        return interp_func


    def inner_product(self, u, v):
        eval_xi, eval_yi = self.gdata.eval_xi, self.gdata.eval_yi
        weights = self.gdata.weights
        u_interp_func = self.interpolate_solution(self.gdata.x1, self.gdata.x2, u * v)
        uv_eval = u_interp_func(eval_xi, eval_yi)
        return np.sum(weights * uv_eval)

    def orthogonalize(self, u, eigenfunctions):
        u_orth = u.copy()
        for v in eigenfunctions:
            proj = self.inner_product(u_orth, v) / self.inner_product(v, v)
            u_orth -= proj * v
        return u_orth / np.linalg.norm(u_orth)

    def qrSolve_shift(self, rhs, shift, l):
        rct_shifted = self.make_rct_matrix_shift(l, shift)
        Q, R = np.linalg.qr(rct_shifted)
        u_small = scipy.linalg.solve_triangular(R.T, rhs, lower=True)
        u_small = Q @ u_small
        u_full_grid = np.zeros((self.gdata.m, self.gdata.m))
        u_full_grid[self.gdata.flag] = u_small[:self.gdata.k]
        u = self.ker(u_full_grid, l).flatten()
        Au = self.a_u(u)
        residual = np.linalg.norm(Au - shift * u)
        print(f"Shift={shift:.4f}, Residual={residual:.2e}")
        return u
    
    def rq_int(self, u, Au):
        eval_xi, eval_yi = self.gdata.eval_xi, self.gdata.eval_yi
        weights = self.gdata.weights
        Au = Au.flatten()
        Au_interp_func = self.interpolate_solution(self.gdata.x1, self.gdata.x2, u * Au)
        Au_eval = Au_interp_func(eval_xi, eval_yi)
        u_interp_func = self.interpolate_solution(self.gdata.x1, self.gdata.x2, u * u)
        uu_eval = u_interp_func(eval_xi, eval_yi)
        numerator = np.sum(weights * Au_eval)
        denominator = np.sum(weights * uu_eval)
        return numerator / denominator

    def rq_int_iter_eig(self, l, u0=None, tol=1e-6, max_iter=100, eigenfunctions=None, mode=1):
        
        if eigenfunctions is None:
            eigenfunctions = []

        if u0 is None:
            u0 = np.random.rand(self.gdata.m**2)
            u0 /= np.linalg.norm(u0)

        u = u0.copy()
        Au = self.a_u(u)
        shift = self.rayleigh_quotient(u, Au)  
    
        rhs_i = u[:self.gdata.k]
        rhs_b = np.zeros(self.gdata.p)
        rhs = np.hstack((rhs_i, rhs_b))
        print(rhs.shape)
        
        # Iteration loop
        for iteration in range(1, max_iter + 1):
            
            u_new = self.qrSolve_shift(rhs, shift, l)
            norm_u_new = np.linalg.norm(u_new)
            if norm_u_new < 1e-14:
                raise ValueError("Solution vector is numerically zero.")
            u_new /= norm_u_new  

            if eigenfunctions:
                u_new = self.orthogonalize(u_new, eigenfunctions)

            Au_new = self.a_u(u_new)
            
            shift_new = self.rq_int(u_new, Au_new)  

            residual1 = np.linalg.norm(Au_new - shift_new * u_new)  #
            residual2 = np.linalg.norm(u_new - u)  

            print(f"Iteration {iteration}: Shift={shift_new:.8f}, Residual1={residual1:.2e}, Residual2={residual2:.2e}")

            if residual1 < 1e-2 and residual2 < tol:
                print(f"Converged after {iteration} iterations.")
                return u_new, shift_new, iteration

            u = u_new
            shift = shift_new  

        print("Max iterations reached without convergence.")
        return u, shift, iteration