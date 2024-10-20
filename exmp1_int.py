# # exmp1_v4.py: is the main script that runs the Rayleigh Quotient Iteration method on the modified RayleighOperator class. The main function initializes the grid and operator, runs the Rayleigh Quotient Iteration method, and calculates the errors and results. The results are saved and graphs are generated using the make_graph and make_chart functions.
# import sys
# import os

# sys.path.append(os.path.join(os.path.dirname(__file__), "SEEM_Chebyshev_master"))

# import time
# import numpy as np
# from CSSEM import get_h
# from operator_data import operator_data
# from rayleigh_operator import (
#     RayleighOperator,
# )  # Importing your modified RayleighOperator
# from grid_data2 import grid_data
# # from griddata import GridDataWithWeights
# from tabulate import tabulate
# from matplotlib import pyplot as plt
# from print_results import make_graph, make_chart


# def main():
#     print("Starting the Rayleigh Quotient Iteration...")

#     # Define exact solution for error calculation
#     exact_sol = lambda x, y: x**3 + y**3

#     # Results dictionary initialization
# results = dict()
# results["p"] = [1, 2, 3]  # Polynomial degrees
# results["pts"] = [8, 12, 16]  # Number of points
# results["L2"] = np.zeros((np.size(results["p"]), np.size(results["pts"])))
# results["inf"] = np.zeros_like(results["L2"])
# results["iteration"] = np.zeros_like(results["L2"], dtype=int)
# results["times"] = np.zeros_like(results["L2"])
# results["condition"] = np.zeros_like(results["L2"])
# results["intpts"] = np.zeros_like(results["pts"])
# results["bdrypts"] = np.zeros_like(results["pts"])

#     for l in np.arange(len(results["p"])):
#         for k in range(len(results["pts"])):
#             print(f"Running for p={results['p'][l]} and pts={results['pts'][k]}...")

#             # Grid and operator setup
#             gdata = grid_data(
#                 results["pts"][k],
#                 [lambda x: 0.95 * np.cos(x), lambda y: 0.95 * np.sin(y)],
#                 results["p"][l],
#                 precond=False,
#                 order="spectral",
#             )
#             odata = RayleighOperator(gdata, results["p"][l], precond=False)
#             sol = exact_sol(gdata.x1, gdata.x2)

#             # Run Rayleigh Quotient Iteration with Integration and Interpolation
#             print("Running Rayleigh Quotient Iteration...")
#             u, cond, iterations = odata.rq_int_iter(l)

#             # Process u_interior and compute errors as needed
#             u_interior = u[: gdata.k]  # Extract the interior points
#             l2_norm = np.linalg.norm(u_interior - sol[gdata.flag]) / np.linalg.norm(
#                 sol[gdata.flag]
#             )
#             results["L2"][l, k] = l2_norm

#             print(
#                 f"L2 Norm for p={results['p'][l]}, pts={results['pts'][k]}: {l2_norm}"
#             )

#     print("Rayleigh Quotient Iteration completed.")
#     np.save("exp1_results.npy", results)
#     make_graph(results, "exp1_graph.pdf")
#     make_chart(results)


# if __name__ == "__main__":
#     main()


# exmp1_v4.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "SEEM_Chebyshev_master"))

import numpy as np
from rayleigh_operator import RayleighOperator
from grid_data2 import grid_data
from print_results import make_graph, make_chart


def main():
    print("Starting the Rayleigh Quotient Iteration...")

    # Define exact solution for error calculation
    exact_sol = lambda x, y: x**3 + y**3

    # Results dictionary initialization

    results = dict()
    results["p"] = [1, 2, 3]  # Polynomial degrees
    results["pts"] = [8, 12, 16]  # Number of points
    results["L2"] = np.zeros((np.size(results["p"]), np.size(results["pts"])))
    results["inf"] = np.zeros_like(results["L2"])
    results["iteration"] = np.zeros_like(results["L2"], dtype=int)
    results["times"] = np.zeros_like(results["L2"])
    results["condition"] = np.zeros_like(results["L2"])
    results["intpts"] = np.zeros_like(results["pts"])
    results["bdrypts"] = np.zeros_like(results["pts"])

    for l_idx, l in enumerate(results["p"]):
        for k_idx, pts in enumerate(results["pts"]):
            print(f"Running for p={l} and pts={pts}...")

            # Grid and operator setup
            gdata = grid_data(
                pts,
                [lambda x: 0.95 * np.cos(x), lambda y: 0.95 * np.sin(y)],
                l,
                precond=False,
                order="spectral",
            )
            odata = RayleighOperator(gdata, l, precond=False)
            sol = exact_sol(gdata.x1, gdata.x2)

            # Run Rayleigh Quotient Iteration
            print("Running Rayleigh Quotient Iteration...")
            u, lambdaU, iterations = odata.rq_int_iter(l)

            # Compute errors
            u_interior = u[: gdata.k]
            sol_interior = sol[gdata.flag]

            l2_norm = np.linalg.norm(u_interior - sol_interior) / np.linalg.norm(
                sol_interior
            )
            results["L2"][l_idx, k_idx] = l2_norm
            results["iteration"][l_idx, k_idx] = iterations

            print(f"L2 Norm for p={l}, pts={pts}: {l2_norm}")
            print(f"Iterations: {iterations}")

    print("Rayleigh Quotient Iteration completed.")
    np.save("exp1_results.npy", results)
    make_graph(results, "exp1_graph.pdf")
    make_chart(results)


if __name__ == "__main__":
    main()
