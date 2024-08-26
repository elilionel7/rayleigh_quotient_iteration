# rayleigh_operator.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "SEEM_Chebyshev_master"))

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
    def iter_qr_solve(self, rhs1, l):
        rhs2 = np.zeros_like(self.gdata.b[:, 0])
        rhs = np.hstack((rhs1, rhs2))

        rct = self.make_rct_matrix(l)
        cond = np.linalg.cond(rct)

        Q, R = np.linalg.qr(rct)
        u = scipy.linalg.solve_triangular(R.T, rhs, lower=True)
        u = np.dot(Q, u)
        u = np.reshape(u, (self.gdata.m, self.gdata.m))
        u = self.ker(u, l).flatten()

        return u, cond

    def iter_qr_solve2(self, rhs1, l):
        rhs2 = np.zeros_like(self.gdata.b[:, 0])
        rhs = np.hstack((rhs1, rhs2))

        rct = self.make_rct_matrix(l)
        cond = np.linalg.cond(rct)
        Q, R = np.linalg.qr(rct)
        u = scipy.linalg.solve_triangular(R.T, rhs, lower=True)
        u = np.dot(Q, u)
        u = np.reshape(u, (self.gdata.m, self.gdata.m))
        u = self.ker(u, l).flatten()

        return u, cond

    def iter_solver(self, l, tol=1e-8, max_iter=100):
        rhs1 = np.ones_like(self.gdata.x1[self.gdata.flag])

        for iteration in range(max_iter):
            u, cond = self.iter_qr_solve(rhs1, l)

            u /= np.linalg.norm(u)

            u_reshaped = np.reshape(u, (self.gdata.m, self.gdata.m))
            new_rhs1 = u_reshaped[self.gdata.flag]
            

            if np.linalg.norm(new_rhs1 - rhs1) < tol:
                return u, cond, iteration + 1

            rhs1 = new_rhs1

        return u, cond
