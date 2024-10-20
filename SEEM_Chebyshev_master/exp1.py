import time
import numpy as np
from CSSEM import get_h
from operator_data import operator_data
from grid_data2 import grid_data
from tabulate import tabulate
from matplotlib import pyplot as plt
from print_results import make_graph, make_chart

def main():
    '''  Experiment 1 - Dirichlet BVP - Analytic Solution '''
    #   Part A: Pseudoinverse Method
    f = lambda x,y: -6*x + 6*y
    g = lambda x,y: x**3 - y**3
    #The different order regularizers are those in p.
    results = dict()
    results['p'] = [1,2,3,4,5,6]
    results['pts'] = [8,12,16,20,24,28,32,36]

    #   Error vectors.
    results['L2'] = np.zeros((np.size(results['p']),np.size(results['pts'])))
    results['inf'] = np.zeros_like(results['L2'])
    results['iteration'] = np.zeros_like(results['L2'],dtype=int)
    results['times'] = np.zeros_like(results['L2'])
    results['condition'] = np.zeros_like(results['L2'])
    results['intpts'] = np.zeros_like(results['pts'])
    results['bdrypts'] = np.zeros_like(results['pts'])
    for l in np.arange(len(results['p'])):
        for k in range(len(results['pts'])):       
            gdata = grid_data(results['pts'][k],[lambda x:.95*np.cos(x),lambda y:.95*np.sin(y)],results['p'][l],precond=False,order='spectral')
            odata = operator_data(gdata,results['p'][l],precond=False)
            sol = g(gdata.x1,gdata.x2)
            ''' The following method computes the matrix implicitly and uses the PCG solver.'''
            tic = time.time() 
            u, cond = odata.qr_solve2(f,g,results['p'][l])
            toc = time.time()
            ''' Calculate errors.'''
            u = np.reshape(u,(gdata.m,gdata.m))
            results['intpts'][k] = gdata.k
            results['bdrypts'][k] = gdata.p
            results['times'][l,k] = toc-tic
            results['condition'][l,k] = cond
            results['L2'][l,k] = np.linalg.norm(u[gdata.flag]-sol[gdata.flag])/np.linalg.norm(sol[gdata.flag])
            results['inf'][l,k] = np.max(np.abs(u[gdata.flag]-sol[gdata.flag]))/np.max(np.abs(u[gdata.flag]))
    np.save('exp1_results.npy',results)
    make_graph(results,'exp1_pseu.pdf')
    make_chart(results)
    # for k in n:
    #     print('$'+str(pts[k-n[0]])+'^2 = '+str(pts[k-n[0]]**2)+'$',' & ',intpts[k-n[0]],' & ', bdrypts[k-n[0]],' & ',end='')
    #     print(' & '.join(str("{:.2E}".format(x)) for x in error[:,k-n[0]]),end=' ')
    #     print('\\\\',end=' ') 
    #     print('\\hline')
    # print(
    #       '\\multicolumn{3}{|c|}{Rate of Convergence:}',
    #       ' & '.join([str("{:.2E}".format(x)) for x in np.log(error[:,0]/error[:,-1])/np.log(pts[0]/pts[-1])]),
    #       '\\\\ \\hline'
    #       )
    ##   Part B: Schur Complement Method
    #f = lambda x,y: -6*x + 6*y
    #g = lambda x,y: x**3 - y**3
    #p = [2,3,4]
    #n = np.arange(4,11)
    #h = get_h()
    #
    ##   Error vectors.
    #error = np.zeros((np.size(p),np.size(n)))
    #errorinf = np.zeros_like(error)
    #iteration = np.zeros_like(error,dtype=int)
    #times = np.zeros_like(error)
    #cond = np.zeros_like(error)
    #pts = np.zeros_like(n)
    #bdrypts = np.zeros_like(n)
    #
    #for l in p:
    #    for k in n:  
    #        if l != 4 or k <= 9:
    #            gdata = grid_data(2**k,[lambda x:.95*np.cos(x),lambda y:.95*np.sin(y)],l,precond=True,density=2)
    #            odata = operator_data(gdata,l,precond=True)
    #            sol = g(gdata.x1,gdata.x2)
    #            ''' The following method computes the matrix implicitly and uses the PCG solver.'''
    #            tic = time.time()
    #            u, it = odata.solve(f,g)
    #            toc = time.time()
    #            ''' Calculate errors.'''
    #            u = np.reshape(u,(gdata.m,gdata.m))
    #            times[l-p[0],k-n[0]] = toc-tic
    #            iteration[l-p[0],k-n[0]] = it
    #            error[l-p[0],k-n[0]] = np.linalg.norm(u[gdata.flag]-sol[gdata.flag])/np.linalg.norm(sol[gdata.flag])
    #            errorinf[l-p[0],k-n[0]] = np.max(np.abs(u[gdata.flag]-sol[gdata.flag]))/np.max(np.abs(u[gdata.flag])) 
    #for k in n:
    #    print('$'+str(2**k)+'^2='+str(2**k**2),
    #          '&',
    #          ' & '.join(str(x) for x in iteration[:,k-n[0]].tolist()),
    #          '&',
    #          ' & '.join(str(round(x,1)) for x in times[:,k-n[0]].tolist()),
    #          '\\\\',
    #          '\\hline') 
    #

if __name__ == '__main__':
    main()

