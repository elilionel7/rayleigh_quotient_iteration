import time
import numpy as np
from CSSEM import get_h
from operator_data import operator_data
from grid_data2 import grid_data
from tabulate import tabulate
from matplotlib import pyplot as plt
from print_results import make_graph, make_chart

def main():
    '''  Experiment 2 - Robin BVP - Analytic Solution '''
    #   Part A: Pseudoinverse Method
    f = lambda x,y: 0*x
    s = lambda x,y: x**2 - y**2
    g = lambda x: (2*(.75*(1+.2*np.cos(5*x))*np.cos(x))*(.75*(-np.sin(5*x)*np.sin(x)+(1+.2*np.cos(5*x))*np.cos(x)))-2*(.75*(1+.2*np.cos(5*x))*np.sin(x))*(.75*(np.sin(5*x)*np.cos(x)+(1+.2*np.cos(5*x))*np.sin(x))))/np.sqrt((.75*(-np.sin(5*x)*np.sin(x)+(1+.2*np.cos(5*x))*np.cos(x)))**2+(.75*(np.sin(5*x)*np.cos(x)+(1+.2*np.cos(5*x))*np.sin(x)))**2)+(.75*(1+.2*np.cos(5*x))*np.cos(x))**2-(.75*(1+.2*np.cos(5*x))*np.sin(x))**2

    bx = lambda x: .75*(1+.2*np.cos(5*x))*np.cos(x)
    by = lambda x: .75*(1+.2*np.cos(5*x))*np.sin(x)
    
    #The different order regularizers are those in p.
    p = [1,2,3,4,5,6]
    n = [6,10,14,18,22,28,34,40]

    #   Error vectors.
    results = dict()
    results['L2'] = np.zeros((np.size(p),np.size(n)))
    results['inf'] = np.zeros_like(results['L2'])
    results['iteration'] = np.zeros_like(results['L2'],dtype=int)
    results['times'] = np.zeros_like(results['L2'])
    results['condition'] = np.zeros_like(results['L2'])
    results['pts'] = n
    results['intpts'] = np.zeros_like(n)
    results['bdrypts'] = np.zeros_like(n)
    results['p'] = p
    for l in p:
        for k in range(len(n)):
#        for k in n:       
            gdata = grid_data(n[k],[bx,by],l,precond=False,order='spectral',bc='robin')
            odata = operator_data(gdata,l,precond=False)
            sol = s(gdata.x1,gdata.x2)
            ''' The following method computes the matrix implicitly and uses the PCG solver.'''
            tic = time.time()
            #import pdb;pdb.set_trace()
            u, cond = odata.qr_solve2(f,g,l)
            toc = time.time()
            ''' Calculate errors.'''
            u = np.reshape(u,(gdata.m,gdata.m))
            results['intpts'][k] = gdata.k
            results['bdrypts'][k] = gdata.p
            results['times'][l-p[0],k] = toc-tic
            results['condition'][l-p[0],k] = cond
            results['L2'][l-p[0],k] = np.linalg.norm(u[gdata.flag]-sol[gdata.flag])/np.linalg.norm(sol[gdata.flag])
            results['inf'][l-p[0],k] = np.max(np.abs(u[gdata.flag]-sol[gdata.flag]))/np.max(np.abs(u[gdata.flag]))

    # Save results.
    np.save('exp2_results.npy',results)
    make_graph(results,'exp2_pseu.pdf')
    make_chart(results)

if __name__=='__main__':
    main()
