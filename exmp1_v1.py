#exmp1_v1.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'SEEM_Chebyshev_master'))

import time
import numpy as np
from CSSEM import get_h
from operator_data import operator_data
from grid_data2 import grid_data
from tabulate import tabulate
from matplotlib import pyplot as plt
from print_results import make_graph, make_chart

def main():
    '''  Custom Problem - Laplace Equation with Constant Dirichlet Boundary Condition '''
    # Define the PDE and boundary condition
    f = lambda x, y: np.zeros_like(x)   
    g = lambda x, y: np.ones_like(x) 

    # Set the regularization parameters and grid points
    results = dict()
    results['p'] = [1,2]#[1, 2, 3, 4, 5, 6]
    results['pts'] = [8,12]#[8, 12, 16, 20, 24, 28, 32, 36]

    # Error vectors
    results['L2'] = np.zeros((np.size(results['p']), np.size(results['pts'])))
    results['inf'] = np.zeros_like(results['L2'])
    results['iteration'] = np.zeros_like(results['L2'], dtype=int)
    results['times'] = np.zeros_like(results['L2'])
    results['condition'] = np.zeros_like(results['L2'])
    results['intpts'] = np.zeros_like(results['pts'])
    results['bdrypts'] = np.zeros_like(results['pts'])

    for l in np.arange(len(results['p'])):
        for k in range(len(results['pts'])):
            gdata = grid_data(results['pts'][k],
                              [lambda x: .95 * np.cos(x), lambda y: .95 * np.sin(y)],
                              results['p'][l], precond=False, order='spectral')
            odata = operator_data(gdata, results['p'][l], precond=False)
            sol = g(gdata.x1, gdata.x2)
            
            # Solve the problem using QR decomposition
            tic = time.time()
            u, cond = odata.qr_solve(f, g, results['p'][l])
            toc = time.time()
            
            # Calculate errors
            u = np.reshape(u, (gdata.m, gdata.m))
            results['intpts'][k] = gdata.k
            results['bdrypts'][k] = gdata.p
            results['times'][l, k] = toc - tic
            results['condition'][l, k] = cond
            results['L2'][l, k] = np.linalg.norm(u[gdata.flag] - sol[gdata.flag]) / np.linalg.norm(sol[gdata.flag])
            results['inf'][l, k] = np.max(np.abs(u[gdata.flag] - sol[gdata.flag])) / np.max(np.abs(u[gdata.flag]))

    # Save the results and generate graphs and tables
    np.save('exmp1_v1_results.npy', results)
    make_graph(results, 'exmp1_v1_pseu.pdf')
    make_chart(results)

if __name__ == '__main__':
    main()
