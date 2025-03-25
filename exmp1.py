# exmp1.py
import sys
import os
import numpy as np
from scipy.special import jv, jn_zeros
from rayleigh_operator import RayleighOperator
from grid_data2 import grid_data
from plot_eigfunc import plot_eigenfunction

j0 = jn_zeros(0, 1)[0]  

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
    results["p"] = [1]  # Polynomial degrees
    results["pts"] = [30, 40, 50]  # Grid resolutions
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
            odata = RayleighOperator(gdata, l, precond=True)  

            eigfuncs = []
            eigvals = []

            for eigen_idx in range(num_eigfuncs):
                mode =  1
                u0 = exact_eigenfunc(gdata.x1, gdata.x2, mode=mode)
                u0 = u0.flatten()
                u0 /= np.linalg.norm(u0)
                u, lambdaU, iterations = odata.rq_int_iter_eig(l, eigenfunctions=eigfuncs)
                sol = exact_eigenfunc(gdata.x1, gdata.x2, mode=mode)
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
                        print(f"Orthogonality <ψ{i},ψ{j}>: {iprod:.2e}")

    print("\n########-Done-########\n")
        
if __name__ == "__main__":
    main()
   # Generate convergence plots
       # After main(), add:
    # from print_results import make_graph, make_chart
    # from make_graph_rq import make_graph_qr
    # from make_chart_rq import make_chart_qr

    # # Generate convergence plots
    # make_graph(results, "eigenvalues", title="Eigenvalue Convergence")
    # make_chart(results, "L2", title="L2 Error vs Grid Resolution")
    #plot_eigenfunction(u, gdata)
        


 