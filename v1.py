# rayleigh_operator.py
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "SEEM_Chebyshev_master"))

from collections import defaultdict
import pandas as pd
import numpy as np
from operator_data import operator_data
from scipy.fftpack import dct as dct
from scipy.fftpack import idct as idct
from scipy.fftpack import dctn as dctn
from scipy.fftpack import idctn as idctn
from scipy.fftpack import dst as dst
from scipy.fftpack import idst as idst
from scipy.sparse.linalg import LinearOperator
from scipy.integrate import simps
from scipy.interpolate import interp2d, RectBivariateSpline
from scipy.interpolate import griddata
from scipy.integrate import simps
from scipy.sparse.linalg import LinearOperator, gmres


import scipy


class RayleighOperator(operator_data):
    def __init__(self, gdata, p, precond=False):
        super().__init__(gdata, p, precond)

    def a_u(self, u):
        u_full = u.reshape(self.gdata.x1.shape)
        Au_full_grid = self.lap(u_full)
        return Au_full_grid.flatten()

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

    def inner_product(self, u, v):
        eval_xi, eval_yi = self.gdata.eval_xi, self.gdata.eval_yi
        weights = self.gdata.weights
        u_interp_func = self.interpolate_solution(self.gdata.x1, self.gdata.x2, u * v)
        uv_eval = u_interp_func(eval_xi, eval_yi)
        return np.sum(weights * uv_eval)

    def gram_schmidt(self, vectors):
        ortho_vectors = []
        for v in vectors:
            for u in ortho_vectors:
                v -= np.dot(v, u) / np.dot(u, u) * u
            ortho_vectors.append(v / np.linalg.norm(v))
        return ortho_vectors

    def orthogonalize(self, u, eigenfunctions):
        for v in eigenfunctions:
            numerator = self.inner_product(u, v)
            denominator = self.inner_product(v, v)
            c = numerator / denominator
            u = u - c * v
            u = self.gram_schmidt([u] + eigenfunctions)[0]
        return u

    def rq_int_iter_eig1(self, l, tol=1e-6, max_iter=100, eigenfunctions=None):
        if eigenfunctions is None:
            eigenfunctions = []
        u0 = np.ones(self.gdata.x1.size)
        if eigenfunctions:
            u0 = self.orthogonalize(u0, eigenfunctions)
            u0 /= np.linalg.norm(u0)
        x = self.gdata.x1.flatten()
        y = self.gdata.x2.flatten()
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        f_theta = 0.95 * np.ones_like(theta)
        shift_values = 1 - r / f_theta
       
        for iter in range(max_iter):
            u = self.qrSolve_shift(u0[: self.gdata.k], l, shift)
            u /= np.linalg.norm(u)
            if eigenfunctions:
                u = self.orthogonalize(u, eigenfunctions)
                u /= np.linalg.norm(u)
            Au_full_grid = self.a_u(u)
            lambdaU_new = self.rq_int(u, Au_full_grid)
            print(f"lambdaU_new: {lambdaU_new}")
            if np.abs(lambdaU_new - shift) < tol:
                print(f"Converged after {iter+1} iterations.")
                return u, lambdaU_new, iter + 1
            shift = lambdaU_new
            print(f"Updated shift: {shift}")
            u0 = u.copy()
        print("Maximum iterations reached without convergence.")
        return u, lambdaU_new, max_iter
    
    def rq_int_iter_eig(self, l, u0=None, shift=None, tol=1e-6, max_iter=100, eigenfunctions=None, alpha=0.8):
        if eigenfunctions is None:
            eigenfunctions = []
        if u0 is None:
            x = self.gdata.x1.flatten()
            y = self.gdata.x2.flatten()
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            f_theta = 0.95 * np.ones_like(theta)
            u0 = 1 - r / f_theta
            print(f"Initial guess u0 shape: {u0.shape}")
        else:
            u0 = u0.copy()
            print(f"Provided initial guess u0 shape: {u0.shape}")
        if eigenfunctions:
            u0 = self.orthogonalize(u0, eigenfunctions)
            u0 /= np.linalg.norm(u0)
        Au0 = self.a_u(u0)
        shift = self.rq_int(u0, Au0)
        print(f"Initial shift computed using Rayleigh quotient: {shift}")
        lambdaU_sequence = [shift]
        for iter in range(max_iter):
            u = self.qrSolve_shift(u0[: self.gdata.k], l, shift)
            u /= np.linalg.norm(u)
            if eigenfunctions:
                u = self.orthogonalize(u, eigenfunctions)
                u /= np.linalg.norm(u)
            Au_full_grid = self.a_u(u)
            lambdaU_new = self.rq_int(u, Au_full_grid)
            lambdaU_sequence.append(lambdaU_new)
            shift = alpha * lambdaU_new + (1 - alpha) * shift
            residual = abs(lambdaU_new - shift)
            print(f"Iteration {iter+1}: Lambda = {lambdaU_new}, Shift = {shift}, Residual = {residual}")
            if residual < tol:
                print(f"Converged after {iter+1} iterations.")
                return u, lambdaU_new, iter + 1
            u0 = u.copy()
        print("Maximum iterations reached without convergence.")
        return u, lambdaU_new, max_iter


# SEEM_Chebyshev_master/operator_data.py
import numpy as np
from scipy.fftpack import dct as dct
from scipy.fftpack import idct as idct
from scipy.fftpack import dctn as dctn
from scipy.fftpack import idctn as idctn
from scipy.fftpack import dst as dst
from scipy.fftpack import idst as idst
from scipy.sparse.linalg import LinearOperator
import scipy

class operator_data:
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
    def C(self,w):
        # Evaluate at b.
        b = self.gdata.xx.dot(w)
        # Take Chebyshev derivative
        w = np.reshape(w,(self.gdata.m,self.gdata.m))
        z = self.lap(w) + np.transpose(self.lap(np.transpose(w)))
        z = z[self.gdata.flag]
        return np.hstack((z,b))
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
    
    def Ct_shift1(self, w, shift):
        z = np.zeros((self.gdata.m, self.gdata.m))
        z[self.gdata.flag] = w[:self.gdata.k]
        lapt_z = self.lapt(z)
        lapt_z_T = self.lapt(z.T).T
        shift_z = shift.reshape(z.shape) * z
        z_new = lapt_z + lapt_z_T - shift_z
        z_new = z_new.flatten()
        z_new += self.gdata.xxT.dot(w[-self.gdata.p:])
        return z_new
    
    def M(self,w):
        w = self.Ct(w)
        w = self.ker(w)
        w = self.C(w)
        return w
    def transform(self,w):
        u = self.Ct(w)
        u = self.ker(u)
        return u
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
    def solve(self,interior,boundary,tol=1e-8):
        rhs1 = interior(self.gdata.x1[self.gdata.flag],self.gdata.x2[self.gdata.flag])
        rhs2 = boundary(self.gdata.b[:,0],self.gdata.b[:,1])
        rhs = np.hstack((rhs1,rhs2))
        u,it = PCG(self.MM,self.PP,rhs,tol)
        u = self.transform(u)
        return u,it
    def vec_solve(self,rhs,tol=1e-8):
        u,it = PCG(self.MM,self.PP,rhs,tol)
        u = self.transform(u)
        return u,it
    def make_m_matrix(self):
        m = np.zeros((self.gdata.k+self.gdata.p,self.gdata.k+self.gdata.p))
        for i in np.arange(self.gdata.k+self.gdata.p):
            z = np.zeros(self.gdata.k+self.gdata.p)
            z[i] = 1
            z = self.M(z)
            m[:,i] = z
        return m
    def qr_solve(self,interior,boundary,l):
        rhs1 = interior(self.gdata.x1[self.gdata.flag],self.gdata.x2[self.gdata.flag])
        rhs2 = boundary(self.gdata.b[:,0],self.gdata.b[:,1])
        
        rhs = np.hstack((rhs1,rhs2))
        rct = self.make_rct_matrix(l)
        cond = np.linalg.cond(rct)
        Q,R = np.linalg.qr(rct)
        u = scipy.linalg.solve_triangular(R.T,rhs,lower=True)
      

        u = np.dot(Q,u)
        u = np.reshape(u,(self.gdata.m,self.gdata.m))
        u = self.ker(u,l).flatten()

        return u,cond
   
    def qr_solve2(self,interior,boundary,l):
        rhs1 = interior(self.gdata.x1[self.gdata.flag],self.gdata.x2[self.gdata.flag])
        rhs2 = boundary(self.gdata.g_grid)
        rhs = np.hstack((rhs1,rhs2))
        rct = self.make_rct_matrix(l)
        cond = np.linalg.cond(rct)
        Q,R = np.linalg.qr(rct)
        u = scipy.linalg.solve_triangular(R.T,rhs,lower=True)
        u = np.dot(Q,u)
        u = np.reshape(u,(self.gdata.m,self.gdata.m))
        u = self.ker(u,l).flatten()
        return u,cond
    
    
    def qrSolve_shift(self, rhs1, l, shift):
        rct = self.make_rct_matrix_shift(l, shift) 
        rhs2 = np.zeros(self.gdata.p)  
        rhs = np.hstack((rhs1, rhs2)) 
        # cond = np.linalg.cond(rct)
        Q,R = np.linalg.qr(rct)
        u = scipy.linalg.solve_triangular(R.T,rhs,lower=True)
        u = np.dot(Q,u)
        u = np.reshape(u,(self.gdata.m,self.gdata.m))
        u = self.ker(u,l).flatten()
        return u

    
    def make_rct_matrix(self,l):
        m = np.zeros((self.gdata.m**2,self.gdata.k+self.gdata.p))
        for i in range(0,self.gdata.k+self.gdata.p):
            z = np.zeros(self.gdata.k+self.gdata.p)
            z[i] = 1
            z = self.ker(self.Ct(z),l).flatten()
            m[:,i] = z
        return m
  
    def make_rct_matrix_shift(self, l, shift):
        m = np.zeros((self.gdata.m**2, self.gdata.k + self.gdata.p))
        for i in range(0, self.gdata.k + self.gdata.p):
            z = np.zeros(self.gdata.k + self.gdata.p)
            z[i] = 1
            z = self.ker(self.Ct_shift(z, shift), l).flatten()
            m[:, i] = z
        return m
    
       
def PCG(MM,PP,rhs,tol=1e-8):
    tol = np.linalg.norm(rhs)*tol
    it = 0
    u = np.zeros_like(rhs)
    r = rhs
    z = PP*r
    p = z
    while np.sqrt(np.dot(r,r)) > tol and it < np.size(rhs):
        rz = np.dot(r,z)
        Mp = MM*p
        alpha = rz/np.dot(p,Mp)
        u = u + alpha*p
        r = r - alpha*Mp
        z = PP*r
        beta = np.dot(r,z)/rz
        p = z + beta*p
        it = it + 1
    return u,it



# exmp_int_vec.py:
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "SEEM_Chebyshev_master"))

import numpy as np
from rayleigh_operator import RayleighOperator
from grid_data2 import grid_data
from print_results import make_graph, make_chart
from make_graph_rq import make_graph_qr
from make_chart_rq import make_chart_qr


def main():
    print("Starting the Rayleigh Quotient Iteration...")

    exact_sol = lambda x, y: x**3 + y**3
    num_eigfuncs = 3

    results = dict()
    results["p"] = [1, 2, 3]  # Polynomial degrees
    results["pts"] = [20, 24, 30]  # Number of points
    results["L2"] = np.zeros((len(results["p"]), len(results["pts"]), num_eigfuncs))
    results["eigenvalues"] = np.zeros(
        (len(results["p"]), len(results["pts"]), num_eigfuncs)
    )
    results["iterations"] = np.zeros_like(results["eigenvalues"], dtype=int)
    results["orthogonality"] = np.zeros(
        (len(results["p"]), len(results["pts"]), num_eigfuncs, num_eigfuncs)
    )

    for l_idx, l in enumerate(results["p"]):
        for k_idx, pts in enumerate(results["pts"]):
            print(f"\nRunning for p={l} and pts={pts}...")

            # Grid and operator setup
            gdata = grid_data(
                pts,
                [lambda x: 0.95 * np.cos(x), lambda y: 0.95 * np.sin(y)],
                l,
                precond=False,
                order="spectral",
            )
            odata = RayleighOperator(gdata, l, precond=False)
            sol = exact_sol(gdata.x1, gdata.x2)

            eigfuncs = []
            eigvals = []

            for eigen_idx in range(num_eigfuncs):
                print(f"\nComputing eigenfunction {eigen_idx + 1}...")

                u, lambdaU, iterations = odata.rq_int_iter_eig(
                    l, eigenfunctions=eigfuncs
                )

                u_interior = u[: gdata.k]
                sol_interior = sol[gdata.flag]

                l2_norm = np.linalg.norm(u_interior - sol_interior) / np.linalg.norm(
                    sol_interior
                )

                eigfuncs.append(u)
                eigvals.append(lambdaU)
                results["eigenvalues"][l_idx, k_idx, eigen_idx] = lambdaU
                results["iterations"][l_idx, k_idx, eigen_idx] = iterations
                results["L2"][l_idx, k_idx] = l2_norm

                print(
                    f"L2 Norm for Eigenfunction {eigen_idx + 1}, p={l}, pts={pts}: {l2_norm}"
                )
                print(f"Eigenvalue {eigen_idx + 1} for p={l}, pts={pts}: {lambdaU}")
                print(f"Iterations: {iterations}")

            for i in range(len(eigfuncs)):
                for j in range(len(eigfuncs)):
                    inner_prod = odata.inner_product(eigfuncs[i], eigfuncs[j])
                    results["orthogonality"][l_idx, k_idx, i, j] = inner_prod
                    if i != j:
                        print(
                            f"Inner product between eigenfunction {i + 1} and {j + 1}: {inner_prod}"
                        )

    print("\nRayleigh Quotient Iteration completed.")

if __name__ == "__main__":
    main()