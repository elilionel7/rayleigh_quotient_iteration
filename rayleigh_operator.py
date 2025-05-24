# rayleigh_operator.py
import numpy as np
from scipy.fftpack import dct, idct, dctn, idctn, dst, idst
from scipy.sparse.linalg import LinearOperator
from scipy.interpolate import griddata, RegularGridInterpolator
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

    def ker(self, w, l=None):
        if l is None:
            l = self.l
        w = np.reshape(w, (self.gdata.m, self.gdata.m))
        w = (
            idctn(dctn(w) * (1 + self.gdata.fx**2 + self.gdata.fy**2) ** -l)
            / (2 * self.gdata.m) ** 2
        )
        return w.flatten()

    def lap(self, w):
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

    def C_shift(self, w, shift=0):
        b = self.gdata.xx.dot(w)
        w = np.reshape(w, (self.gdata.m, self.gdata.m))
        z = self.lap(w) + np.transpose(self.lap(np.transpose(w))) - shift * w
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

    def a_u(self, u, l):
        u = self.ker(u, l)
        w = np.reshape(u, (self.gdata.m, self.gdata.m))
        Au = self.lap(w) + np.transpose(self.lap(np.transpose(w)))
        Au[~self.gdata.flag] = 0
        return Au.flatten()

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
            (x, y), sols_grid, method="linear", bounds_error=False, fill_value=None
        )

        def interpolator(xi, yi):
            xi = np.asarray(xi).flatten()
            yi = np.asarray(yi).flatten()
            interp_points = np.column_stack((xi, yi))
            zi = interp_func(interp_points)
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
            proj = self.inner_product(u_orth, v)/self.inner_product(v, v)
            u_orth -= proj * v
        return u_orth / np.linalg.norm(u_orth)

    def verify_eigenfunction(self, u, l, eigenvalue):
        Au = self.a_u(u, l).reshape(self.gdata.m, self.gdata.m)
        Au = Au.flatten()
        u_hat = Au / eigenvalue
        relative_error = np.linalg.norm(u_hat - u) / np.linalg.norm(u)
        return relative_error, u_hat

    def qrSolve_shift(self, rhs, shift, l):
        rct_shifted = self.make_rct_matrix_shift(l, shift)
        Q, R = np.linalg.qr(rct_shifted)
        y = scipy.linalg.solve_triangular(R.T, rhs, lower=True)
        u_small = Q @ y
        m = self.gdata.m
        u_grid = u_small.reshape((m, m))

        u = self.ker(u_grid, l).flatten()
        Au = self.a_u(u, l)
        residual = np.linalg.norm(Au - shift * u)
        print(f"From qr_shift, Shift={shift:.4f}, Residual={residual:.2e}")
        return u

    

    def rq_int0(self, u, Au):
        eval_xi, eval_yi = self.gdata.eval_xi, self.gdata.eval_yi
        weights = self.gdata.weights
        Au = Au.flatten()
        Au_interp_func = self.interpolate_solution1(
            self.gdata.x1, self.gdata.x2, u * Au
        )
        Au_eval = Au_interp_func(eval_xi, eval_yi)
        u_interp_func = self.interpolate_solution1(self.gdata.x1, self.gdata.x2, u * u)
        uu_eval = u_interp_func(eval_xi, eval_yi)
        numerator = np.sum(weights * Au_eval)
        denominator = np.sum(weights * uu_eval)
        return numerator / denominator

    
    def rq_int(self, u, Au):
        
        m = self.gdata.m
        u_grid = u.reshape(m, m)
        Au_grid = Au.reshape(m, m)
        u_interior = u_grid[self.gdata.flag]
        Au_interior = Au_grid[self.gdata.flag]
        rq =  np.mean(Au_interior / u_interior)

        return rq



    def rq_int_iter_eig(self, l, u0=None, tol=1e-6, max_iter=100, eigenfunctions=None):
        if eigenfunctions is None:
            eigenfunctions = []

        if u0 is None:
            x, y = self.gdata.x1.flatten(), self.gdata.x2.flatten()
            r = np.sqrt(x**2 + y**2)
            u0 = 1 - r / 0.95
            u0 /= np.linalg.norm(u0)

        u = u0.astype(np.float64)
        au = self.a_u(u, l)
        shift = self.rq_int(u, au)
        

        u_grid = u.reshape(self.gdata.m, self.gdata.m)
        rhs_i  = u_grid[self.gdata.flag]      
        rhs_b  = np.zeros(self.gdata.p)       
        rhs    = np.hstack((rhs_i, rhs_b))

        for iteration in range(1, max_iter + 1):
            u_new = self.qrSolve_shift(rhs, shift, l)
            Cu = self.C_shift(u_new, shift)

            cu_res = np.linalg.norm(Cu - rhs) / np.linalg.norm(rhs)
            print(f"Relative constraint residual (||Cu - rhs||/||rhs||): {cu_res:.2e}")

            if eigenfunctions:
                u_new = self.orthogonalize(u_new, eigenfunctions)

            u_new /= np.linalg.norm(u_new)

            au_new = self.a_u(u_new, l)
            shift_new = self.rq_int(u_new, au_new)
            
            res1 = np.linalg.norm(u_new - u) / np.linalg.norm(u_new)
            res2 = np.linalg.norm(u_new + u) / np.linalg.norm(u_new)


            print(
                f"Iteration {iteration}: Shift={shift_new:.8f},||u_new - u||/||u_new||={res1:.2e}, ||u_new + u||/||u_new||={res2:.2e}"
            )

            if res1 < tol or res2 < tol:
                relative_error_verification, _ = self.verify_eigenfunction(u_new, l, shift_new)
                print(f"Final eigenfunction verification relative error: {relative_error_verification:.2e}")
                print(f"Converged after {iteration} iterations.")
                return u_new, shift_new, iteration
            
            u = u_new
            shift = shift_new

            u_grid = u_new.reshape(self.gdata.m, self.gdata.m)
            rhs_i  = u_grid[self.gdata.flag]       
            rhs    = np.hstack((rhs_i, rhs_b))

        print("Maximum iterations reached.")
        return u, shift, iteration