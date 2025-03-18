from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.fftpack import dctn, idctn
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags
from scipy.sparse.linalg import gmres
from scipy.special import jn
import scipy

import numpy as np

class RayleighOperatorv1:
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
    
    def inner_product(self, u, v):
        return np.dot(u, v)
    
    def orthogonalize(self, u, eigenfunctions):
        for ef in eigenfunctions:
            u -= np.dot(ef, u) * ef
        return u / np.linalg.norm(u)
    
    def initial_guess(self, mode):
        
        m = self.gdata.m  # Grid points
        x = np.linspace(-1, 1, m)
        y = np.linspace(-1, 1, m)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        # Use Bessel function J_0 for radial symmetry; adjust k for mode
        k = mode  # Approximate mode index; refine with known zeros if needed
        u0 = jn(0, k * np.pi * R) * (R < 1)  # Mask outside unit disk
        u0 = u0.flatten()
        return u0 / np.linalg.norm(u0)

    def build_laplacian_matrix(self):
        m = self.gdata.m
        h = 2 * 0.95 / m  # Grid spacing
        n = m * m
        # Finite difference Laplacian stencil
        diagonals = [
            -4 / h**2 * np.ones(n),           # Main diagonal
            1 / h**2 * np.ones(n - 1),        # Right
            1 / h**2 * np.ones(n - 1),        # Left
            1 / h**2 * np.ones(n - m),        # Up
            1 / h**2 * np.ones(n - m)         # Down
        ]
        offsets = [0, 1, -1, m, -m]
        A = diags(diagonals, offsets, shape=(n, n)).tocsr()
        # Enforce Dirichlet BCs: set boundary rows to identity
        boundary_indices = np.where(~self.gdata.flag.flatten())[0]
        for idx in boundary_indices:
            A[idx, :] = 0
            A[idx, idx] = 1
        return A

    def rayleigh_quotient(self, u, Au):
        return np.dot(u, Au) / np.dot(u, u)

    def orthogonalize(self, u, eigenfunctions):
        for ef in eigenfunctions:
            u -= np.dot(u, ef) * ef
        return u / np.linalg.norm(u)

    def rq_int_iter_eig(self, l, u0=None, tol=1e-6, max_iter=100, eigenfunctions=None, mode=1):
        
        if eigenfunctions is None:
            eigenfunctions = []
        if u0 is None:
            u0 = self.initial_guess(mode)
        else:
            u0 = u0 / np.linalg.norm(u0)

        u = u0.copy()
        A = self.build_laplacian_matrix()
        shift = self.rayleigh_quotient(u, A @ u)

        for iteration in range(1, max_iter + 1):
            I = eye(A.shape[0], format='csr')
            try:
                u_new, info = gmres(A - shift * I, u, tol=1e-8)
                if info != 0:
                    print(f"GMRES failed at iteration {iteration}, info={info}")
                    Au = A @ u
                    residual1 = np.linalg.norm(Au - shift * u)
                    if residual1 < 1e-10:
                        print(f"Converged due to small Residual1: {residual1}")
                        return u, shift, iteration
                    else:
                        print(f"Residual1 too large: {residual1}, continuing...")
                        u_new = u
            except Exception as e:
                print(f"Solve failed at iteration {iteration}: {e}")
                break

            norm_u_new = np.linalg.norm(u_new)
            if norm_u_new < 1e-14:
                raise ValueError("Solution vector became numerically zero.")
            u_new /= norm_u_new

            if eigenfunctions:
                u_new = self.orthogonalize(u_new, eigenfunctions)

            Au_new = A @ u_new
            shift_new = self.rayleigh_quotient(u_new, Au_new)
            residual1 = np.linalg.norm(Au_new - shift_new * u_new)
            residual2 = np.linalg.norm(u_new - u)
            residual3 = np.linalg.norm(u_new + u)

            print(f"Iteration {iteration}: Shift={shift_new:.8f}, Residual1={residual1:.2e}, Residual2={residual2:.2e}, Residual3={residual3:.2e}")
            if residual1 < 1e-10 and residual2 < tol:
                print(f"Converged after {iteration} iterations.")
                return u_new, shift_new, iteration

            u = u_new
            shift = shift_new

        print("Max iterations reached without convergence.")
        return u, shift, iteration