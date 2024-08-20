import numpy as np
from matplotlib import pyplot as plt

def make_graph(results,file_name):
    if type(results) is str:
        results = np.load(results,allow_pickle=True).item()
    markers = [ '+', 'o', '*','v','s','p']
    p = results['p']
    fig = plt.figure(figsize=(25,10))
    ax1 = fig.add_subplot(121)
    ax1.set_title('$L_2$ Error',fontsize=25)
    ax1.set_xlabel('Grid Size',fontsize=22)
    ax1.set_ylabel('$L_2$ Error',fontsize=22)
    ax2 = fig.add_subplot(122)
    ax2.set_title('$L_\infty$ Error',fontsize=25)
    ax2.set_xlabel('Grid Size',fontsize=22)
    ax2.set_ylabel('$L_\infty$ Error',fontsize=22)
    for l in p:
        ax1.loglog(results['pts'],
                   results['L2'][l-p[0],:],
                   label='$S_'+str(l)+'$',
                   marker=markers[l-1],
                   linewidth=4,
                   markersize=16)
        ax2.loglog(results['pts'],
                   results['inf'][l-p[0],:],
                   label='$S_'+str(l)+'$',
                   marker=markers[l-1],
                   linewidth=4,
                   markersize=16)
    ax1.set_xticks(results['pts'])
    ax1.set_xticklabels([str(x) for x in results['pts']])
    ax1.minorticks_off()
    ax1.legend(loc='lower left',fontsize=16)
    ax2.set_xticks(results['pts'])
    ax2.set_xticklabels([str(x) for x in results['pts']])
    ax2.minorticks_off()
    ax2.legend(loc='lower left',fontsize=16)
    ax1.tick_params(labelsize=18)
    ax2.tick_params(labelsize=18)
    fig.savefig(file_name,dpi=600)

def make_chart(results):
    if type(results) is str:
        results = np.load(results,allow_pickle=True).item()
    pts = results['pts']
    for k in range(len(pts)):
        print('$'+str(pts[k])+'^2 = '+str(pts[k]**2)+'$',' & ',results['intpts'][k],' & ',results['bdrypts'][k],' & ',end='')
        print(' & '.join(str("{:.2E}".format(x)) for x in results['L2'][:,k]),end=' ')
        print('\\\\',end=' ') 
        print('\\hline')
    print(
          '\\multicolumn{3}{|c|}{Rate of Convergence:}',
          ' & '.join([str(round(x,2)) for x in -np.log(results['L2'][:,0]/results['L2'][:,-1])/np.log(pts[0]/pts[-1])]),
          '\\\\ \\hline'
          )
