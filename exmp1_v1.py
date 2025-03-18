import numpy as np
from scipy.special import jv, jn_zeros
from rq_operator import RayleighOperatorv1
from grid_data2 import grid_data
from plot_eigfunc import plot_eigenfunction

def exact_eigenfunc(x, y, mode):
    r = np.sqrt(x**2 + y**2)
    R = 0.95
    theta = np.arctan2(y, x)
    if mode == 1:
        j0_k = jn_zeros(0, 1)[0]
        return jv(0, j0_k * r / R)
    elif mode == 2:
        j1_k = jn_zeros(1, 1)[0]
        return jv(1, j1_k * r / R) * np.cos(theta)
    elif mode == 3:
        j1_k = jn_zeros(1, 1)[0]
        return jv(1, j1_k * r / R) * np.sin(theta)
    else:
        raise ValueError("Mode not implemented")

def main():
    print("Starting Rayleigh Quotient Iteration with Valid Eigenfunction...")

    results = dict()
    results["p"] = [1]
    results["pts"] = [20, 24, 30]
    num_eigfuncs = 3
    
    results["L2"] = np.zeros((len(results["p"]), len(results["pts"]), num_eigfuncs))
    results["eigenvalues"] = np.zeros_like(results["L2"])
    results["iterations"] = np.zeros_like(results["L2"], dtype=int)
    results["orthogonality"] = np.zeros(
        (len(results["p"]), len(results["pts"]), num_eigfuncs, num_eigfuncs)
    )

    for l_idx, l in enumerate(results["p"]):
        for k_idx, pts in enumerate(results["pts"]):
            print(f"\nRunning for p={l} and pts={pts}...")
            gdata = grid_data(
                pts,
                [lambda x: 0.95 * np.cos(x), lambda y: 0.95 * np.sin(y)],
                l,
                precond=True,
                order="spectral",
            )
            odata = RayleighOperatorv1(gdata, l, precond=True)

            eigfuncs = []
            for eigen_idx in range(num_eigfuncs):
                print(f"\nComputing eigenfunction {eigen_idx + 1}...")
                u0 = exact_eigenfunc(gdata.x1, gdata.x2, eigen_idx + 1)#1-f(x,y)
                u0 = u0.flatten()
                u0 /= np.linalg.norm(u0)
                
                u, lambdaU, iterations = odata.rq_int_iter_eig(
                    l,
                    u0=u0,
                    eigenfunctions=eigfuncs,
                    mode=eigen_idx + 1
                )

                sol = exact_eigenfunc(gdata.x1, gdata.x2, eigen_idx + 1)
                sol_norm = np.linalg.norm(sol[gdata.flag])
                u_interior = u[:gdata.k]
                sol_interior = sol[gdata.flag]
                sign = np.sign(np.dot(u_interior, sol_interior))
                rel_error = np.linalg.norm(sign * u_interior - sol_interior) / sol_norm
                
                eigfuncs.append(u)
                results["eigenvalues"][l_idx, k_idx, eigen_idx] = lambdaU
                results["iterations"][l_idx, k_idx, eigen_idx] = iterations
                results["L2"][l_idx, k_idx, eigen_idx] = rel_error

                print(f"Relative L2 Error (Eigenfunction {eigen_idx+1}): {rel_error:.2e}")
                print(f"Computed Eigenvalue: {lambdaU:.4f}, Iterations: {iterations}")

            for i in range(num_eigfuncs):
                for j in range(num_eigfuncs):
                    iprod = odata.inner_product(eigfuncs[i], eigfuncs[j])
                    results["orthogonality"][l_idx, k_idx, i, j] = iprod
                    if i != j:
                        print(f"Orthogonality <ψ{i+1},ψ{j+1}>: {iprod:.2e}")

    print("\n########-Done-########\n")

if __name__ == "__main__":
    main()