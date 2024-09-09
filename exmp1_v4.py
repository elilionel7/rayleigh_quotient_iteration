# exmp1_v4.py: is the main script that runs the Rayleigh Quotient Iteration method on the modified RayleighOperator class. The main function initializes the grid and operator, runs the Rayleigh Quotient Iteration method, and calculates the errors and results. The results are saved and graphs are generated using the make_graph and make_chart functions.
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "SEEM_Chebyshev_master"))

import time
import numpy as np
from CSSEM import get_h
from operator_data import operator_data
from rayleigh_operator import (
    RayleighOperator,
)  # Importing your modified RayleighOperator
from grid_data2 import grid_data
from tabulate import tabulate
from matplotlib import pyplot as plt
from print_results import make_graph, make_chart


def main():
    # Define exact solution for error calculation
    exact_sol = lambda x, y: x**3 + y**3

    # Results dictionary initialization
    results = dict()
    results["p"] = [1, 2, 3]
    results["pts"] = [8, 12, 16]

    results["L2"] = np.zeros((np.size(results["p"]), np.size(results["pts"])))
    results["inf"] = np.zeros_like(results["L2"])
    results["iteration"] = np.zeros_like(results["L2"], dtype=int)
    results["times"] = np.zeros_like(results["L2"])
    results["condition"] = np.zeros_like(results["L2"])
    results["intpts"] = np.zeros_like(results["pts"])
    results["bdrypts"] = np.zeros_like(results["pts"])

    for l in np.arange(len(results["p"])):
        for k in range(len(results["pts"])):
            # Grid and operator setup
            gdata = grid_data(
                results["pts"][k],
                [lambda x: 0.95 * np.cos(x), lambda y: 0.95 * np.sin(y)],
                results["p"][l],
                precond=False,
                order="spectral",
            )
            odata = RayleighOperator(gdata, results["p"][l], precond=False)
            sol = exact_sol(gdata.x1, gdata.x2)

            tic = time.time()

            u, cond, iterations = odata.rayleigh_quotient_iteration(l)

            toc = time.time()

            u_interior = u[: gdata.k]

            results["intpts"][k] = gdata.k
            results["bdrypts"][k] = gdata.p
            results["times"][l, k] = toc - tic
            results["condition"][l, k] = cond
            results["iteration"][l, k] = iterations

            # Correctly index u_interior using 1D indexing
            sol_interior = sol[
                gdata.flag
            ].flatten()  # Extract the solution for interior points
            results["L2"][l, k] = np.linalg.norm(
                u_interior - sol_interior
            ) / np.linalg.norm(sol_interior)
            results["inf"][l, k] = np.max(np.abs(u_interior - sol_interior)) / np.max(
                np.abs(sol_interior)
            )

    np.save("exp1_v3results.npy", results)
    make_graph(results, "exp1_v3pseu.pdf")
    make_chart(results)


if __name__ == "__main__":
    main()
