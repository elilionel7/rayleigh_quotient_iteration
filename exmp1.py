# exmp1.py
import numpy as np
from scipy.special import jv, jn_zeros
from rayleigh_operator import RayleighOperator
from grid_data2 import grid_data
from plot_eigfunc import plot_eigenfunctions
import matplotlib.pyplot as plt

nr_quad = 40
ntheta_quad = 60

def exact_eigenfunc(x, y, mode=1):
    R = 0.95
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if mode == 1:
        return jv(0, jn_zeros(0, 1)[0] * r / R)
    elif mode == 2:
        return jv(1, jn_zeros(1, 1)[0] * r / R) * np.cos(theta)
    elif mode == 3:
        return jv(1, jn_zeros(1, 1)[0] * r / R) * np.sin(theta)

def main():
    print("Starting Rayleigh Quotient Iteration with Valid Eigenfunction...")

    p = [4]
    pts = [60]
    num_eigfuncs = 3

    results = {
        "L2": np.zeros((len(p), len(pts), num_eigfuncs)),
        "eigenvalues": np.zeros((len(p), len(pts), num_eigfuncs)),
        "iterations": np.zeros((len(p), len(pts), num_eigfuncs), dtype=int),
        "orthogonality": np.zeros((len(p), len(pts), num_eigfuncs, num_eigfuncs))
    }

    for l_idx, l in enumerate(p):
        for k_idx, pts in enumerate(pts):
            print(f"\nRunning for p={l} and pts={pts}...")

            gdata = grid_data(
                pts,
                [lambda x: 0.95 * np.cos(x), lambda y: 0.95 * np.sin(y)],
                l,
                precond=True,
                order="spectral",
                nr_quad=nr_quad, ntheta_quad=ntheta_quad
            )
            odata = RayleighOperator(gdata, l, precond=True)

            eigfuncs = []

            for eigen_idx in range(num_eigfuncs):
                mode = eigen_idx + 1
                u0 = exact_eigenfunc(gdata.x1, gdata.x2, mode=mode).flatten()
                u0 /= np.linalg.norm(u0)

                u, lam, iters = odata.rq_int_iter_eig(l, u0=u0, eigenfunctions=eigfuncs)

                R = 0.95
                if mode == 1:
                    m = 0
                    k = 1
                elif mode == 2 or mode == 3:
                    m = 1
                    k = 1
                else:
                    raise ValueError("Mode not implemented for analytic eigenvalue.")
                j = jn_zeros(m, k)[-1]
                lambda_exact = (j / R) ** 2
                
                print(f"   λ_computed = {lam: .6f}")
                print(f"   λ_exact    = {lambda_exact: .6f}")
                print(f"   rel. error = {abs(lam-lambda_exact)/lambda_exact: .2e}")

                eigfuncs.append(u)
                results["eigenvalues"][l_idx, k_idx, eigen_idx] = lam
                results["iterations"][l_idx, k_idx, eigen_idx] = iters

                sol = exact_eigenfunc(gdata.x1, gdata.x2, mode=mode)
                sol_norm = np.linalg.norm(sol[gdata.flag])
                u_interior = u[:gdata.k]
                sol_interior = sol[gdata.flag]

                sign = np.sign(np.dot(u_interior, sol_interior))
                rel_error = np.linalg.norm(sign * u_interior - sol_interior) / sol_norm
                results["L2"][l_idx, k_idx, eigen_idx] = rel_error

                print(f"Relative L2 Error (Eigenfunction {eigen_idx+1}): {rel_error:.2e}")
                print(f"Computed Eigenvalue: {lam:.4f}, Iterations: {iters}")

                
                u_normed = u / np.linalg.norm(u)
                sol_normed = sol.flatten() / np.linalg.norm(sol.flatten())
                iprod = np.dot(u_normed, sol_normed)
                print(f"Inner product (computed vs analytic): {iprod:.4f}")

                plot_eigenfunctions(u_normed, sol_normed, gdata,
                                    filename=f"eigenfunc_p{l}_pts{pts}_mode{mode}.png")
                plt.close()

            for i in range(num_eigfuncs):
                for j in range(num_eigfuncs):
                    iprod = np.real(odata.inner_product(eigfuncs[i], eigfuncs[j]))
                    results["orthogonality"][l_idx, k_idx, i, j] = iprod
                    if i != j:
                        print(f"Orthogonality <ψ{i},ψ{j}>: {iprod:.2e}")

    print("\n########-Done-########\n")

if __name__ == "__main__":
    main()
