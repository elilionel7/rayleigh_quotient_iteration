# exmp1.py
import numpy as np
from scipy.special import jv, jn_zeros
from rayleigh_operator import RayleighOperator
from grid_data2 import grid_data
from plot_eigfunc import plot_eigenfunctions, plot_Au_surfaces
import matplotlib.pyplot as plt

nr_quad = 20
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

    p = [2]
    pts_list = [50, 100, 200] 
    num_eigfuncs = 3
    Au_list = []
    gdata_list = []

    

    for l_idx, l in enumerate(p):
        for k_idx, pts in enumerate(pts_list):
            print(f"\nRunning for p={l} and pts={pts}...")

            gdata = grid_data(
                pts,
                [lambda x: 0.95 * np.cos(x), lambda y: 0.95 * np.sin(y)],
                l,
                precond=True,
                order="spectral",
                nr_quad=nr_quad, ntheta_quad=ntheta_quad
            )
            gdata_list.append(gdata)
            odata = RayleighOperator(gdata, l, precond=True)


            for eigen_idx in range(num_eigfuncs):
                mode = eigen_idx + 1
                u0 = exact_eigenfunc(gdata.x1, gdata.x2, mode=mode)
                u0_flat = u0.flatten()
                u0 /= np.linalg.norm(u0)

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

               
                u_exact_interior = u0_flat[gdata.flag.flatten()]
                Au_exact_interior = odata.a_u(u0_flat, l)[gdata.flag.flatten()]
                mean_Au_over_u = np.mean(Au_exact_interior / u_exact_interior)
                print(f"pts: {pts}, Mode: {mode}")
                print(f"Mean(Au/u): {mean_Au_over_u}")
                print(f"Exact eigenvalue: {lambda_exact}")


   

                Au0 = odata.a_u(u0, l)
                Au_list.append(Au0)
                gdata_list.append(gdata)

            plot_Au_surfaces(Au_list, gdata_list, pts_list, filename="Au_surfaces.png")
            plt.close()

              
    print("\n########-Done-########\n")

if __name__ == "__main__":
    main()
