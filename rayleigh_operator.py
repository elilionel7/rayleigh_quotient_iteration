# rayleigh_operator.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "SEEM_Chebyshev_master"))

from collections import defaultdict

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
    def __init__(self, gdata, p, precond=False):
        super().__init__(gdata, p, precond)

    def make_rct_matrix_square(self, l):
        total_points = self.gdata.k + self.gdata.p
        m = np.zeros((total_points, total_points))

        for i in range(total_points):
            z = np.zeros(total_points)
            z[i] = 1

            z_transformed = self.ker(self.Ct(z), l).flatten()

            if z_transformed.shape[0] > total_points:
                z_transformed = z_transformed[:total_points]
            elif z_transformed.shape[0] < total_points:
                z_transformed = np.pad(
                    z_transformed,
                    (0, total_points - z_transformed.shape[0]),
                    mode="constant",
                )

            m[:, i] = z_transformed

        return m

    def qrSolve(self, rhs1, l):
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

    def qrSolve2(self, rhs1, l):
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

        for iter in range(max_iter):
            u, cond = self.qrSolve(rhs1, l)

            u /= np.linalg.norm(u)

            u_reshaped = np.reshape(u, (self.gdata.m, self.gdata.m))
            new_rhs1 = u_reshaped[self.gdata.flag]

            if np.linalg.norm(new_rhs1 - rhs1) < tol:
                return u, cond, iter + 1

            rhs1 = new_rhs1

        return u, cond

    def qrSolve_shift(self, rhs1, l, shift=5):
        rhs2 = np.zeros_like(self.gdata.b[:, 0])
        rhs = np.hstack((rhs1, rhs2))

        rct = self.make_rct_matrix(l)

        rct_shifted = rct.copy()
        for row in range(len(rhs1)):
            rct_shifted[row][row] -= shift

        cond = np.linalg.cond(rct_shifted)

        Q, R = np.linalg.qr(rct_shifted)
        u = scipy.linalg.solve_triangular(R.T, rhs, lower=True)
        u = np.dot(Q, u)
        u = np.reshape(u, (self.gdata.m, self.gdata.m))
        u = self.ker(u, l).flatten()

        return u, cond

    def iter_solver_shift(self, l, tol=1e-8, max_iter=100, shift=0.0):
        rhs1 = np.ones_like(self.gdata.x1[self.gdata.flag])

        for iter in range(max_iter):
            u, cond = self.qrSolve_shift(rhs1, l, shift)

            u /= np.linalg.norm(u)

            u_reshaped = np.reshape(u, (self.gdata.m, self.gdata.m))
            new_rhs1 = u_reshaped[self.gdata.flag]

            if np.linalg.norm(new_rhs1 - rhs1) < tol:
                return u, cond, iter + 1

            rhs1 = new_rhs1

        return u, cond, iter + 1

    def iter_solver_variable_shift(self, l, shifts, tol=1e-8, max_iter=100):
        results = defaultdict(list)
        for shift in shifts:
            u, cond, _ = self.iter_solver_shift(
                l, tol=tol, max_iter=max_iter, shift=shift
            )
            results[shift].append((u, cond))

        return results

    def rayleigh_quotient(self, u, A):
        u = u.flatten()

        num = np.dot(u.T, np.dot(A, u))  # u^T A u
        denom = np.dot(u.T, u)  # u^T u
        return num / denom

    def solve_shifted_system(self, A, shift, rhs1, rhs2):
        rhs = np.hstack((rhs1, rhs2))

        A_shifted = A.copy()

        int_len = len(rhs1)
        for row in range(int_len):
            A_shifted[row, row] -= shift
        u = np.linalg.solve(A_shifted, rhs)

        return u

    def rayleigh_quotient_iteration(self, l, tol=1e-8, max_iter=100):
        A = self.make_rct_matrix_square(l)

        u = np.random.rand(A.shape[1])
        u /= np.linalg.norm(u)

        rhs1 = np.ones_like(self.gdata.x1[self.gdata.flag])
        rhs2 = np.zeros_like(self.gdata.b[:, 0])

        lambdaU = self.rayleigh_quotient(u, A)

        for iter in range(max_iter):
            u_new = self.solve_shifted_system(A, lambdaU, rhs1, rhs2)

            u_new /= np.linalg.norm(u_new)

            lambdaU_new = self.rayleigh_quotient(u_new, A)

            if np.abs(lambdaU_new - lambdaU) < tol and np.linalg.norm(u_new - u) < tol:
                return u_new, lambdaU_new, iter + 1

            u = u_new
            lambdaU = lambdaU_new

        return u, lambdaU_new, max_iter
