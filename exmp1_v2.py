import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'SEEM_Chebyshev_master'))

import time
import numpy as np
from CSSEM import get_h
from operator_data import operator_data
from rayleigh_operator import RayleighOperator
from grid_data2 import grid_data
from tabulate import tabulate
from matplotlib import pyplot as plt
from print_results import make_graph, make_chart

def main():
    
    # f = lambda x, y: np.ones_like(x)  
    g = lambda x, y: np.zeros_like(x)  

    results = dict()
    results['p'] = [1, 2, 3, 4, 5, 6]
    results['pts'] = [8, 12, 16, 20, 24, 28, 32, 36]

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
            odata = RayleighOperator(gdata, results['p'][l], precond=False)
            sol = g(gdata.x1, gdata.x2)
            
            # Solve the problem using Rayleigh Quotient Iteration
            tic = time.time()
           
            u, eigenvalue, _ = odata.rayleigh_quotient_iteration(g,results['p'][l])
           
            toc = time.time()

            # Calculate errors
            u = np.reshape(u, (gdata.m, gdata.m))
            results['intpts'][k] = gdata.k
            results['bdrypts'][k] = gdata.p
            results['times'][l, k] = toc - tic
            results['condition'][l, k] = eigenvalue  
            results['L2'][l, k] = np.linalg.norm(u[gdata.flag] - sol[gdata.flag]) / np.linalg.norm(sol[gdata.flag])
            results['inf'][l, k] = np.max(np.abs(u[gdata.flag] - sol[gdata.flag])) / np.max(np.abs(u[gdata.flag]))

    np.save('exp1_results.npy', results)
    make_graph(results, 'exp1_pseu.pdf')
    make_chart(results)

if __name__ == '__main__':
    main()
