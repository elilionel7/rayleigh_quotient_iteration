# exmp1.py
import sys
import os
import numpy as np
from scipy.special import jv, jn_zeros
from rayleigh_operator import RayleighOperator
from grid_data2 import grid_data
from plot_eigfunc import plot_eigenfunctions
import pandas as pd


nr_quad = 20
ntheta_quad = 60


def exact_eigenfunc(x, y, mode=1):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if mode == 1:
        j0_k = jn_zeros(0, 1)[0]
        return jv(0, j0_k * r / 0.95)
    elif mode == 2:
        j1_k = jn_zeros(1, 1)[0]
        return jv(1, j1_k * r / 0.95) * np.cos(theta)
    elif mode == 3:
        j1_k = jn_zeros(1, 1)[0]
        return jv(1, j1_k * r / 0.95) * np.sin(theta)


def main():
    print("Starting Rayleigh Quotient Iteration with Valid Eigenfunction...")

    results = dict()
    results["p"] = [2]
    results["pts"] = [30]
    num_eigfuncs = 2

    

    results["L2"] = np.zeros((1, 1, num_eigfuncs))
    results["eigenvalues"] = np.zeros_like(results["L2"], dtype=np.complex128)
    results["iterations"] = np.zeros_like(results["L2"], dtype=int)
    results["orthogonality"] = np.zeros((1, 1, num_eigfuncs, num_eigfuncs), dtype=np.complex128)

    for l_idx, l in enumerate(results["p"]):
        for k_idx, pts in enumerate(results["pts"]):
            print(f"\nRunning for p={l} and pts={pts}...")

            gdata = grid_data(
                pts,
                [lambda x: 0.95 * np.cos(x), lambda y: 0.95 * np.sin(y)],
                l,
                precond=True,
                order="spectral",
                nr_quad=nr_quad,
                ntheta_quad=ntheta_quad,
            )
            
            odata = RayleighOperator(gdata, l, precond=True)

            # Diagnostic Laplacian check
            # Add diagnostic check in exmp1.py or similar:
            u_test = np.cos(np.pi * gdata.x1) * np.cos(np.pi * gdata.x2)
            lap_u_computed = odata.lap(u_test)
            lap_u_exact = -2 * np.pi**2 * u_test

            error = np.linalg.norm((lap_u_computed - lap_u_exact)[gdata.flag]) / np.linalg.norm(lap_u_exact[gdata.flag])
            print(f"Diagnostic Laplacian Error: {error:.2e}")

            
            # Compute eigenfunctions
            eigfuncs = []
            eigvals = []
            for eigen_idx in range(num_eigfuncs):
                mode =  eigen_idx + 1
                u0 = exact_eigenfunc(gdata.x1, gdata.x2, mode=mode)
                u0 = u0.flatten()
                u0 /= np.linalg.norm(u0)
                u, lambdaU, iterations = odata.rq_int_iter_eig(l,u0=u0, eigenfunctions=eigfuncs)
                relative_error_verification, u_hat = odata.verify_eigenfunction(u, l, lambdaU)
                print(f"Eigenfunction verification relative error: {relative_error_verification:.2e}")


                sol = exact_eigenfunc(gdata.x1, gdata.x2, mode=mode)
                sol_norm = np.linalg.norm(sol[gdata.flag])

                u_interior = u[:gdata.k]  
                sol_interior = sol[gdata.flag]  
                
                sign = np.sign(np.dot(u_interior, sol_interior))
                rel_error = np.linalg.norm(sign * u_interior - sol_interior) / sol_norm
                
                eigfuncs.append(u)
                eigvals.append(lambdaU)

                plot_eigenfunctions(
                    u_computed=u.reshape(gdata.m, gdata.m),
                    u_exact=sol,
                    gdata=gdata,
                    mode= mode,
                    eigenvalue=eigvals[eigen_idx],
                    l2_error=rel_error,
                )
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
                        print(f"Orthogonality <ψ{i},ψ{j}>: {iprod:.2e}")

            print("Verifying boundary conditions...")
            for i in range(num_eigfuncs):
                u = eigfuncs[i]
                bvals = gdata.xx @ u
                max_bc_error = np.max(np.abs(bvals))
                print(f"Max BC error (Computed Mode {i+1}): {max_bc_error:.2e}")

    print("\n########-Done-########\n")

    # Save results
    rows = []
    for i in range(num_eigfuncs):
        rows.append({
            "p": l,
            "pts": pts,
            "mode": i + 1,
            "eigenvalue": results["eigenvalues"][0, 0, i].real,
            "iterations": results["iterations"][0, 0, i],
            "L2_error": results["L2"][0, 0, i].real,
        })

    df = pd.DataFrame(rows)
    df.to_csv("eigen_results.csv", index=False)
    print("Saved results to eigen_results.csv")


if __name__ == "__main__":
    main()
  