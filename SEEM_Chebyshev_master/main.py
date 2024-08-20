import time
import numpy as np
from CSSEM import get_h
from operator_data import operator_data
from grid_data2 import grid_data
from tabulate import tabulate
from matplotlib import pyplot as plt

'''  Experiment 1 - Analytic Solution '''
#   Part A: Pseudoinverse Method
#f = lambda x,y: 0*x
#g = lambda x,y: x**2 - y**2
#
#
##Basic PCG method for solving problem.
##The finest grid is of size 2**(n-1).
##The different order regularizers are those in p.
#p = [2,3,4]
#n = np.arange(4,9)
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
#        gdata = grid_data(2**k,[lambda x:.95*np.cos(x),lambda y:.95*np.sin(y)],l)
#        odata = operator_data(gdata,l)
#        sol = g(gdata.x1,gdata.x2)
#        ''' The following method computes the matrix implicitly and uses the PCG solver.'''
#        tic = time.time()
#        u, it = odata.solve(f,g)
#        toc = time.time()
#        ''' Calculate errors.'''
#        u = np.reshape(u,(gdata.m,gdata.m))
#        times[l-p[0],k-n[0]] = toc-tic
#        iteration[l-p[0],k-n[0]] = it
#        error[l-p[0],k-n[0]] = np.linalg.norm(u[gdata.flag]-sol[gdata.flag])/np.linalg.norm(sol[gdata.flag])
#        errorinf[l-p[0],k-n[0]] = np.max(np.abs(u[gdata.flag]-sol[gdata.flag]))/np.max(np.abs(u[gdata.flag]))
#    print('Smoother: p=',l)
#    print(tabulate({'Grid':n,
#                   'Times':times[l-p[0],:],
#                   'Iterations':iteration[l-p[0],:],
#                   'L2 Error':error[l-p[0],:],
#                   'L Infinity Error':errorinf[l-p[0],:]},headers="keys"))
#    
#for k in n:
#    print(2**k,
#          '&',
#          ' & '.join(str(x) for x in iteration[:,k-n[0]].tolist()),
#          '&',
#          ' & '.join(str(round(x,1)) for x in times[:,k-n[0]].tolist()),
#          '\\\\') 
#    print('\\hline')

f = lambda x,y: -6*x + 6*y
g = lambda x,y: x**3 - y**3
#The different order regularizers are those in p.
p = [1,2,3,4,5,6]
n = np.arange(9)

#   Error vectors.
error = np.zeros((np.size(p),np.size(n)))
errorinf = np.zeros_like(error)
iteration = np.zeros_like(error,dtype=int)
times = np.zeros_like(error)
condition = np.zeros_like(error)
pts = np.zeros_like(n)
bdrypts = np.zeros_like(n)
for l in p:
    for k in n:       
        gdata = grid_data(8 + 4*k,[lambda x:.95*np.cos(x),lambda y:.95*np.sin(y)],l,precond=False,order='spectral')
        odata = operator_data(gdata,l,precond=False)
        sol = g(gdata.x1,gdata.x2)
        ''' The following method computes the matrix implicitly and uses the PCG solver.'''
        tic = time.time()
        u, cond = odata.qr_solve(f,g,l)
        toc = time.time()
        ''' Calculate errors.'''
        u = np.reshape(u,(gdata.m,gdata.m))
        times[l-p[0],k-n[0]] = toc-tic
        condition[l-p[0],k-n[0]] = cond
        error[l-p[0],k-n[0]] = np.linalg.norm(u[gdata.flag]-sol[gdata.flag])/np.linalg.norm(sol[gdata.flag])
        errorinf[l-p[0],k-n[0]] = np.max(np.abs(u[gdata.flag]-sol[gdata.flag]))/np.max(np.abs(u[gdata.flag]))
    print('Smoother: p=',l)
    print(tabulate({'Grid':n,
                   'Times':times[l-p[0],:],
                   'Iterations':iteration[l-p[0],:],
                   'L2 Error':error[l-p[0],:],
                   'L Infinity Error':errorinf[l-p[0],:]},headers="keys"))
#markers = [ '+', 'o', '*','v','s','p']
#fig = plt.figure(figsize=(25,10))
#ax1 = fig.add_subplot(121)
#ax1.set_title('$L_2$ Error')
#ax2 = fig.add_subplot(122)
#ax2.set_title('$L_\infty$ Error')
#for l in p:
#    ax1.loglog(8+4*n,error[l-p[0],:],label='$S_'+str(l)+'$',marker=markers[l-1])
#    ax2.loglog(8+4*n,errorinf[l-p[0],:],label='$S_'+str(l)+'$',marker=markers[l-1])
#ax1.set_xticks(8+4*n)
#ax1.set_xticklabels([str(x) for x in 8+4*n])
#ax1.minorticks_off()
#ax1.legend(loc='lower left')
#ax2.set_xticks(8+4*n)
#ax2.set_xticklabels([str(x) for x in 8+4*n])
#ax2.minorticks_off()
#ax2.legend(loc='lower left')
#fig.savefig('exp1_pseu.pdf',dpi=600)
#for k in n:
#    print(12+4*n,
#          '&',
#          ' & '.join(str("{:.2E}".format(x)) for x in error[:,k-n[0]].tolist()),
#          '&',
#          ' & '.join(str("{:.2E}".format(x))  for x in errorinf[:,k-n[0]].tolist()),
#          '\\\\') 
#    print('\\hline')
