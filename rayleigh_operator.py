import numpy as np
from scipy.fftpack import dctn, idctn
from scipy.sparse.linalg import LinearOperator
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

    def lap(self, u):
        h = 2 * 0.95 / self.gdata.m  # Grid spacing
        u_full = np.reshape(u, (self.gdata.m, self.gdata.m))
        ux = np.gradient(u_full, h, axis=0)
        uxx = np.gradient(ux, h, axis=0)
        uy = np.gradient(u_full, h, axis=1)
        uyy = np.gradient(uy, h, axis=1)
        return (uxx + uyy).flatten()

    def C(self, w):
        w_grid = np.reshape(w, (self.gdata.m, self.gdata.m))
        lap_w = np.reshape(self.lap(w_grid.flatten()), (self.gdata.m, self.gdata.m))
        lap_w += np.reshape(self.lap(w_grid.T.flatten()), (self.gdata.m, self.gdata.m)).T
        lap_w[~self.gdata.flag] = 0  # Dirichlet boundary
        boundary_eval = self.gdata.xx @ w.flatten()
        return np.hstack((lap_w[self.gdata.flag], boundary_eval))
    
    def Ct(self, w):
        z = np.zeros((self.gdata.m, self.gdata.m))
        z[self.gdata.flag] = w[:self.gdata.k]
        z = self.lap(z) + self.lap(z.T).T
        z = z.flatten()
        z += self.gdata.xxT @ w[-self.gdata.p:]
        return z

    def Ct_shift(self, w, shift):
        z = np.zeros((self.gdata.m, self.gdata.m))
        z[self.gdata.flag] = w[:self.gdata.k]
        lap_z = self.lap(z.flatten()).reshape(self.gdata.m, self.gdata.m)
        lap_z_t = self.lap(z.T.flatten()).reshape(self.gdata.m, self.gdata.m).T
        z_shifted = lap_z + lap_z_t - shift * z
        z_flat = z_shifted.flatten()
        z_flat += self.gdata.xxT.dot(w[-self.gdata.p:])
        return z_flat

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

    def M(self, w):
        return self.C(self.ker(self.Ct(w)))

    def make_rct_matrix_shift(self, l, shift):
        m = np.zeros((self.gdata.k + self.gdata.p, self.gdata.k + self.gdata.p))
        for i in range(self.gdata.k + self.gdata.p):
            z = np.zeros(self.gdata.k + self.gdata.p)
            z[i] = 1
            m[:, i] = self.C(self.ker(self.Ct_shift(z, shift), l))
        return m - shift * np.eye(self.gdata.k + self.gdata.p)

    def a_u(self, u):
        Au_full_grid = self.lap(u.reshape(self.gdata.x1.shape))
        return Au_full_grid.flatten()

    def rayleigh_quotient(self, u, Au):
        return np.dot(u, Au) / np.dot(u, u)

    def inner_product(self, u, v):
        return np.dot(u, v)

    def orthogonalize(self, u, eigenfunctions):
        for v in eigenfunctions:
            u -= self.inner_product(u, v) / self.inner_product(v, v) * v
        return u / np.linalg.norm(u)

    def qrSolve_shift(self, rhs, shift, l):
        print(f"Shift before qrSolve_shift: {shift:.4f}")  # Debug shift
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

    def rq_int_iter_eig(self, l, u0=None, tol=1e-6, max_iter=100, eigenfunctions=None):
        if eigenfunctions is None:
            eigenfunctions = []

        if u0 is None:
            u0 = np.random.rand(self.gdata.m**2)
            u0 /= np.linalg.norm(u0)

        u = u0.copy()
        Au = self.a_u(u)
        shift = self.rayleigh_quotient(u, Au)

        for iteration in range(1, max_iter + 1):
            rhs_interior = u[:self.gdata.k]
            rhs_boundary = np.zeros(self.gdata.p)
            rhs = np.hstack((rhs_interior, rhs_boundary))

            u_new = self.qrSolve_shift(rhs, shift, l)
            norm_u_new = np.linalg.norm(u_new)

            if norm_u_new < 1e-14:
                raise ValueError("Solution vector became numerically zero.")

            u_new /= norm_u_new

            if eigenfunctions:
                u_new = self.orthogonalize(u_new, eigenfunctions)

            Au_new = self.a_u(u_new)
            shift_new = self.rayleigh_quotient(u_new, Au_new)
            residual1 = np.linalg.norm(Au_new - shift_new * u_new)  # Eigenvalue equation error
            residual2 = np.linalg.norm(u_new - u)  # Change in u
            print(f"Iteration {iteration}: Shift={shift_new:.8f}, Residual1={residual1:.2e}, Residual2={residual2:.2e}")

            # Adjusted convergence criteria
            if residual1 < 1e-2 and residual2 < 1e-6:
                print(f"Converged after {iteration} iterations.")
                return u_new, shift_new, iteration

            u = u_new
            shift = shift_new

        print("Max iterations reached without convergence.")
        return u, shift, iteration