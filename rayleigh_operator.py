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
from scipy.interpolate import RegularGridInterpolator

import scipy

class RayleighOperator:
    def __init__(self, gdata, l, precond=True):
        self.gdata = gdata
        self.l = l
        self.MM = LinearOperator((gdata.k + gdata.p, gdata.k + gdata.p), matvec=self.M)
        if precond:
            self.PP = LinearOperator(
                (gdata.k + gdata.p, gdata.k + gdata.p), matvec=self.precond
            )

    def ker_main(self, w, l=None):
        if l is None:
            l = self.l
        w = np.reshape(w, (self.gdata.m, self.gdata.m))
        w = (
            np.real(
                idctn(dctn(w) * (1 + self.gdata.fx**2 + self.gdata.fy**2) ** -l)
            )
            / (2 * self.gdata.m) ** 2
        )
        return w.flatten()
    
   
    def ker(self, w, l=None):
        if l is None:
            l = self.l
        w = np.reshape(w, (self.gdata.m, self.gdata.m))
        w = np.real(idctn(dctn(w) * (1 + self.gdata.fx**2 + self.gdata.fy**2) ** -l))
        return w.flatten()

    def lap_main(self, w):
        w1 = np.copy(w)
        w1 = dct(w1, axis=0) * -self.gdata.fx
        w1 = np.roll(w1, -1, axis=0)
        w1 = idst(w1, axis=0) / (2 * self.gdata.m)
        w1 = w1 * self.gdata.x1 / (1 - self.gdata.x1**2) ** (3 / 2)
        w2 = np.copy(w)
        w2 = idct(dct(w2, axis=0) * -self.gdata.fx**2, axis=0) / (2 * self.gdata.m)
        w2 = w2 * -1 / (1 - self.gdata.x1**2)
        w = w2
        w = w1 + w2
        return w
    
    def lap(self, w):
        w1 = np.copy(w)
        w1 = dct(w1, axis=0) * -self.gdata.fx
        w1 = np.roll(w1, -1, axis=0)
        w1 = idst(w1, axis=0)  # Remove / (2 * m)
        w1 = w1 * self.gdata.x1 / (1 - self.gdata.x1**2) ** (3 / 2)
        w2 = np.copy(w)
        w2 = idct(dct(w2, axis=0) * -self.gdata.fx**2, axis=0)  # Remove / (2 * m)
        w2 = w2 * -1 / (1 - self.gdata.x1**2)
        w = w1 + w2
        scaling = (self.gdata.m / (2 * 0.95))**2  # Correct scaling for domain size
        return w * scaling
    
    def lapt(self, w):
        w1 = np.copy(w)
        w1 = w1 * self.gdata.x1 / (1 - self.gdata.x1**2) ** (3 / 2)
        w1 = dst(w1, axis=0) / (2 * (self.gdata.m))
        w1 = np.roll(w1, 1, axis=0)
        w1 = idct(w1 * -self.gdata.fx, axis=0)
        w2 = np.copy(w)
        w2 = w2 * -1 / (1 - self.gdata.x1**2)
        w2 = dct(w2, axis=0) / (2 * (self.gdata.m))
        w2 = w2 * -self.gdata.fx**2
        w2 = idct(w2, axis=0)
        z = w1 + w2
        return z

    def C(self, w):
        b = self.gdata.xx.dot(w)
        w = np.reshape(w, (self.gdata.m, self.gdata.m))
        z = self.lap(w) + np.transpose(self.lap(np.transpose(w)))
        z = z[self.gdata.flag]
        return np.hstack((z, b))

    def Ct_shift(self, w, shift):
        z = np.zeros((self.gdata.m, self.gdata.m))
        z[self.gdata.flag] = w[: self.gdata.k]
        z = self.lapt(z) + self.lapt(z.T).T - shift * z
        z = z.flatten()
        z += self.gdata.xxT.dot(w[-self.gdata.p :])
        return z

    def Ct(self, w):
        z = np.zeros((self.gdata.m, self.gdata.m))
        z[self.gdata.flag] = w[: self.gdata.k]
        z = self.lapt(z) + self.lapt(z.T).T
        z = z.flatten()
        z += self.gdata.xxT.dot(w[-self.gdata.p :])
        return z

    def precond(self, w):
        w1 = w[: self.gdata.k]
        if self.l == 3:
            w1 += self.gdata.FD(w1)
        elif self.l == 4:
            w1 += self.gdata.FD(2 * w1 + self.gdata.FD(w1))
        elif self.l == 5:
            w1 += self.gdata.FD(3 * w1 + self.gdata.FD(3 * w1 + self.gdata.FD(w1)))
        w2 = scipy.linalg.lu_solve(self.gdata.B, w[-self.gdata.p :])
        return np.hstack((w1, w2))

    def M(self, w):
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

    def a_u_main(self, u,l):
        u = self.ker(u,l)
        w = np.reshape(u, (self.gdata.m, self.gdata.m))
        Au_full_grid = self.lap(w) + np.transpose(self.lap(np.transpose(w)))
        
        return Au_full_grid.flatten()
    
    def a_u(self, u, l):
        u = self.ker(u, l)
        w = np.reshape(u, (self.gdata.m, self.gdata.m))
        w[~self.gdata.flag] = 0
        Au_full_grid = self.lap(w) + np.transpose(self.lap(np.transpose(w)))
        Au_full_grid[~self.gdata.flag] = 0
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
    

    def interpolate_solution1(self, x, y, sols):
    
        x = np.unique(x)
        y = np.unique(y)

        
        sols_grid = sols.reshape(len(x), len(y))

        interp_func = RegularGridInterpolator(
            (x, y), sols_grid, method='linear', bounds_error=False, fill_value=None
        )

        def interpolator(xi, yi):
            xi = np.asarray(xi).flatten()
            yi = np.asarray(yi).flatten()
            interp_points = np.column_stack((xi, yi))
            zi = interp_func(interp_points)
            zi = np.nan_to_num(zi, nan=0.0)
            return zi

        return interpolator


    def inner_product(self, u, v):
        eval_xi, eval_yi = self.gdata.eval_xi, self.gdata.eval_yi
        weights = self.gdata.weights
        u_interp_func = self.interpolate_solution1(self.gdata.x1, self.gdata.x2, u * v)
        uv_eval = u_interp_func(eval_xi, eval_yi)
        return np.sum(weights * uv_eval)

    def orthogonalize(self, u, eigenfunctions):
        u_orth = u.copy()
        for v in eigenfunctions:
            proj = self.inner_product(u_orth, v) / self.inner_product(v, v)
            u_orth -= proj * v
        return u_orth / np.linalg.norm(u_orth)
    

    
    
    def verify_eigenfunction(self, u, l, lambdaU):
        u_hat = self.a_u(u, l)
        relative_error = np.linalg.norm(u_hat - lambdaU * u) / np.linalg.norm(u_hat)
        print(f"Verification: ||Au - λu|| = {np.linalg.norm(u_hat - lambdaU * u):.2e}, ||Au|| = {np.linalg.norm(u_hat):.2e}")
        return relative_error, u_hat
    
    def initial_guess_sin(self, mode):
        x, y = self.gdata.x1, self.gdata.x2
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        if mode == 1:
            guess = np.sin(np.pi * r / 0.95)
        elif mode == 2:
            guess = np.sin(2 * np.pi * r / 0.95) * np.cos(theta)
        elif mode == 3:
            guess = np.sin(3 * np.pi * r / 0.95) * np.sin(theta)
        else:
            guess = np.random.rand(*r.shape)

        noise = np.random.normal(0, 0.05, size=guess.shape)
        return (guess + noise).flatten()

    def get_initial_guess(self, eigenfunctions):
        mode = len(eigenfunctions) + 1
        guess = self.initial_guess_sin(mode)

        return guess / np.linalg.norm(guess)

    def qrSolve_shift(self, rhs, shift, l):
        rct_shifted = self.make_rct_matrix_shift(l, shift).astype(np.complex128)
        rhs = rhs.astype(np.complex128)

        Q, R = np.linalg.qr(rct_shifted)  

        y = scipy.linalg.solve_triangular(R.conj().T, rhs, lower=True)

        u_small = Q @ y
        u_full_grid = np.zeros((self.gdata.m, self.gdata.m), dtype=np.complex128)
        u_full_grid[self.gdata.flag] = u_small[: self.gdata.k]

        u = self.ker(u_full_grid, l).flatten().astype(np.complex128)
        Au = self.a_u(u, l)
        residual = np.linalg.norm(Au - shift * u)
        print(f"Shift={shift:.4f}, Residual={residual:.2e}")
        return u

    def rq_int(self, u, Au):
        eval_xi, eval_yi = self.gdata.eval_xi, self.gdata.eval_yi
        weights = self.gdata.weights
        Au = Au.flatten()
        Au_interp_func = self.interpolate_solution1(self.gdata.x1, self.gdata.x2, np.conj(u) * Au)

        Au_eval = Au_interp_func(eval_xi, eval_yi)
        u_interp_func = self.interpolate_solution1(self.gdata.x1, self.gdata.x2, np.conj(u) * u)
        uu_eval = u_interp_func(eval_xi, eval_yi)
        numerator = np.sum(weights * Au_eval)
        denominator = np.sum(weights * uu_eval)
        return numerator / denominator

    # In rq_int_iter_eig, add boundary enforcement:
    def rq_int_iter_eig(self, l, u0=None, tol=1e-10, max_iter=500, eigenfunctions=None):
        if eigenfunctions is None:
            eigenfunctions = []
        if u0 is None:
            u0 = self.get_initial_guess(eigenfunctions)
        u = u0.astype(np.complex128)
        au = self.a_u(u, l)
        shift = self.rq_int(u, au)
        # shift = shift + 1j*.5
        rhs_b = np.zeros(self.gdata.p, dtype=np.complex128)
        rhs_i = np.ones(self.gdata.k, dtype=np.complex128)
        rhs = np.hstack((rhs_i, rhs_b))
        for iteration in range(1, max_iter + 1):
            u_new = self.qrSolve_shift(rhs, shift, l)
            u_new = u_new.astype(np.complex128)
            u_new_grid = u_new.reshape(self.gdata.m, self.gdata.m)
            u_new_grid[~self.gdata.flag] = 0  # Enforce BC
            u_new = u_new_grid.flatten()
            u_new /= np.linalg.norm(u_new)
            if eigenfunctions:
                u_new = self.orthogonalize(u_new, eigenfunctions)
            au_new = self.a_u(u_new, l)
            shift_new = self.rq_int(u_new, au_new)
            res = np.linalg.norm(au_new - shift_new * u_new)
            u_grid = u.reshape(self.gdata.m, self.gdata.m)
            un_grid = u_new.reshape(self.gdata.m, self.gdata.m)
            u_i = u_grid[self.gdata.flag]
            un_i = un_grid[self.gdata.flag]
            res1 = np.linalg.norm(un_i - u_i)
            boundary_values = un_grid[~self.gdata.flag]
            print(f"Iteration {iteration}: Shift={shift_new:.8f}, Residual={res:.2e}, Eigenfunction Change={res1:.2e}, Boundary Norm={np.linalg.norm(boundary_values):.2e}")
            if res1 < tol or res < tol:
                print(f"Converged after {iteration} iterations.")
                return u_new, shift_new, iteration
            u = u_new
            shift = shift_new
            if iteration % 50 == 0 or res > 1e-2:
                shift *= 0.9
                print(f"Adjusting shift to {shift:.8f}")
            u_grid = u.reshape(self.gdata.m, self.gdata.m)
            rhs_i = u_grid[self.gdata.flag]
            rhs = np.hstack((rhs_i, rhs_b))
        print("Maximum iterations reached.")
        return u, shift, iteration
