# rayleigh_operator.py (merged and corrected)
import numpy as np
from scipy.fftpack import dct as dct
from scipy.fftpack import idct as idct
from scipy.fftpack import dctn as dctn
from scipy.fftpack import idctn as idctn
from scipy.fftpack import dst as dst
from scipy.fftpack import idst as idst
from scipy.sparse.linalg import LinearOperator
import scipy

class RayleighOperator:
    def __init__(self,gdata,l,precond=False):
        self.gdata = gdata
        self.l = l
        self.MM = LinearOperator((self.gdata.k+self.gdata.p,self.gdata.k+self.gdata.p),matvec = self.M)
        if precond == True:
            self.PP = LinearOperator((self.gdata.k+self.gdata.p,self.gdata.k+self.gdata.p),matvec = self.precond)

    def ker(self,w,l=None):
        if l == None:
            l = self.l
        w = np.reshape(w,(self.gdata.m,self.gdata.m))
        w = np.real(idctn(dctn(w)*(1+self.gdata.fx**2 + self.gdata.fy**2)**-l))/(2*self.gdata.m)**2
        return w.flatten()
    def lap(self, u):
        u_full = np.reshape(u, (self.gdata.m, self.gdata.m))
        ux = np.gradient(u_full, axis=0)
        uxx = np.gradient(ux, axis=0)
        uy = np.gradient(u_full, axis=1)
        uyy = np.gradient(uy, axis=1)
        return (uxx + uyy).flatten()
    
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
    
    def precond(self,w):
        w1 = w[:self.gdata.k]
        if self.l == 3:
            w1 = w1 + self.gdata.FD(w1)
        if self.l == 4:
            w1 = w1 + self.gdata.FD(2*w1 + self.gdata.FD(w1))
        if self.l == 5:
            w1 = w1 + self.gdata.FD(3*w1 + self.gdata.FD(3*w1 + self.gdata.FD(w1)))
        w2 = w[-self.gdata.p:]
        w2 = scipy.linalg.lu_solve(self.gdata.B,w2)
        return np.hstack((w1,w2))
    # def C(self, w):
    #         w_reshaped = np.reshape(w, (self.gdata.m, self.gdata.m))
    #         z_interior = self.lap(w_reshaped) + self.lap(w_reshaped.T).T
    #         z_boundary = self.gdata.xx @ w.flatten()  # Dirichlet: u=0 on boundary
    #         z_interior[~self.gdata.flag] = 0  # Enforce boundary condition
    #         return np.hstack((z_interior[self.gdata.flag], z_boundary))
    
    def C(self, w):
        w_reshaped = np.reshape(w, (self.gdata.m, self.gdata.m))
        z_interior = np.reshape(self.lap(w_reshaped.flatten()), (self.gdata.m, self.gdata.m))
        z_interior += np.reshape(self.lap(w_reshaped.T.flatten()), (self.gdata.m, self.gdata.m)).T
        
        z_interior[~self.gdata.flag] = 0  
        z_boundary = self.gdata.xx @ w.flatten()
        return np.hstack((z_interior[self.gdata.flag], z_boundary))

        
    def Ct(self,w):
        z = np.zeros((self.gdata.m,self.gdata.m))
        z[self.gdata.flag] = w[:self.gdata.k]
        z = self.lapt(z) + self.lapt(z.T).T 
        z = z.flatten()
        z += self.gdata.xxT.dot(w[-self.gdata.p:])
        return z
    
    def Ct_shift(self, w, shift): #update from last week meeting
        
        z = np.zeros((self.gdata.m, self.gdata.m))
        z[self.gdata.flag] = w[:self.gdata.k]
        
        z = self.lapt(z) + self.lapt(z.T).T - shift * z
        z = z.flatten()
        z += self.gdata.xxT.dot(w[-self.gdata.p:])
        return z
    
    def M(self,w):
        w = self.Ct(w)
        w = self.ker(w)
        w = self.C(w)
        return w
    
    def make_rct_matrix_shift(self, l, shift):
        m = np.zeros((self.gdata.m**2, self.gdata.k + self.gdata.p))
        for i in range(0, self.gdata.k + self.gdata.p):
            z = np.zeros(self.gdata.k + self.gdata.p)
            z[i] = 1
            z = self.ker(self.Ct_shift(z, shift), l).flatten()
            m[:, i] = z
        return m

    def a_u(self, u):
        Au_full_grid = self.lap(u.reshape(self.gdata.x1.shape))
        return Au_full_grid.flatten()

    def rayleigh_quotient(self, u, Au):
        numerator = np.dot(u, Au)
        denominator = np.dot(u, u)
        return numerator / denominator

    def inner_product(self, u, v):
        return np.dot(u, v)

    def orthogonalize(self, u, eigenfunctions):
        for v in eigenfunctions:
            u -= self.inner_product(u, v) / self.inner_product(v, v) * v
        return u / np.linalg.norm(u)

    
    def qrSolve_shift(self, rhs, shift, l):
        # Construct the shifted operator matrix explicitly
        rct_shifted = self.make_rct_matrix_shift(l, shift)

       
        Q, R = np.linalg.qr(rct_shifted)

      
        u = scipy.linalg.solve_triangular(R.T, rhs, lower=True)
        u = Q @ u

    
        u = np.reshape(u, (self.gdata.m, self.gdata.m))
        u = self.ker(u, l).flatten()

       
        Au = self.a_u(u)
        residual = np.linalg.norm(Au - shift * u)
        print(f"Shift={shift:.4f}, Residual={residual:.2e}")

        return u



    def rq_int_iter_eig(self, l, u0=None, tol=1e-6, max_iter=100, eigenfunctions=None):
        if eigenfunctions is None:
            eigenfunctions = []

        if u0 is None:
            u0 = np.random.rand(self.gdata.m * self.gdata.m)
            u0 /= np.linalg.norm(u0)

        u = u0.copy()
        Au = self.a_u(u)
        shift = self.rayleigh_quotient(u, Au)

        for iteration in range(1, max_iter + 1):
            # Pass the correct parameters explicitly
            u_new = self.qrSolve_shift(u, shift, l)
            norm_u_new = np.linalg.norm(u_new)

            if norm_u_new < 1e-14:
                raise ValueError("Solution vector became numerically zero, iteration stopped.")

            u_new /= norm_u_new

            if eigenfunctions:
                u_new = self.orthogonalize(u_new, eigenfunctions)

            Au_new = self.a_u(u_new)
            shift_new = self.rayleigh_quotient(u_new, Au_new)

            residual = np.linalg.norm(u_new - u)
            print(f"Iteration {iteration}: Shift = {shift_new:.8f}, Residual = {residual:.2e}")

            if residual < tol:
                print(f"Converged after {iteration} iterations.")
                return u_new, shift_new, iteration

            u = u_new
            shift = shift_new

        print("Maximum iterations reached without convergence.")
        return u, shift, iteration
