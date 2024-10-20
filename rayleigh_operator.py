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


    def a_u(self, u, l):
        # u_full_grid = np.zeros(self.gdata.m**2)
        # u_full_grid[self.gdata.flag.flatten()] = u  
        u_full = u.reshape(self.gdata.x1.shape)  

        Au_full_grid = self.ker(u_full, l)

        return Au_full_grid  
        

    #### RQI integration

    def interpolate_solution(self, x, y, sols):
        x = x.flatten()
        y = y.flatten()
        values = sols.flatten()

        pnts = np.column_stack((x, y))

        def interp_func(xi, yi):
            xi = np.asarray(xi).flatten()
            yi = np.asarray(yi).flatten()
            # Perform interpolation
            zi = griddata(pnts, values, (xi, yi), method="cubic")
            return zi

        return interp_func

    def integrate_function(self, f_values, weights):
        f_values = f_values.flatten()
        weights = weights.flatten()
        integral = np.sum(f_values * weights)
        return integral

    def rq_int(self, u_full, Au_full):
        eval_xi, eval_yi = self.gdata.eval_xi, self.gdata.eval_yi
        weights = self.gdata.weights

        u_interp_func = self.interpolate_solution(self.gdata.x1, self.gdata.x2, u_full)
        u_eval = u_interp_func(eval_xi, eval_yi)

        Au_interp_func = self.interpolate_solution(
            self.gdata.x1, self.gdata.x2, Au_full
        )
        Au_eval = Au_interp_func(eval_xi, eval_yi)

        # Compute numerator and denominator
        numerator = np.sum(weights * u_eval * Au_eval)
        denominator = np.sum(weights * u_eval * u_eval)

        return numerator / denominator

    def rq_int_iter(self, l, tol=1e-8, max_iter=100):
        rhs1 = np.ones(self.gdata.k)
        print(f'rhs1 shape : {rhs1.shape}')  
        shift = 0.0

        for iter in range(max_iter):
            u, cond = self.qrSolve_shift(rhs1, l, shift)
            u /= np.linalg.norm(u)
           
            print(f'u shape : {u.shape}')

            Au_full_grid = self.a_u(u, l)

            lambdaU_new = self.rq_int(u, Au_full_grid)

            if np.abs(lambdaU_new - shift) < tol:
                print(f"Converged after {iter+1} iterations.")
                return u, lambdaU_new, iter + 1

            shift = lambdaU_new
            print(f'shift : {shift}')
           
            rhs1 = u[:self.gdata.k].copy()
           
        print("Maximum iterations reached without convergence.")
        return u, lambdaU_new, max_iter
