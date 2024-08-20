import numpy as np
from grid_data2 import grid_data
from operator_data import operator_data
from scipy.fftpack import idctn as idctn;
from delta import dirac

def create_random_function(decay=2,grid=2048):
    gdata = grid_data(grid,[lambda x: .95*np.cos(x),lambda x: .95*np.sin(x)],2)
    u = np.random.rand(gdata.x1.shape[0],gdata.x1.shape[1])*2-1
    u = u/(1+gdata.fx**2+gdata.fy**2)**(decay/2)
    u = np.real(idctn(u))
    np.save('h'+str(decay)+'_func',u)
    return u

def get_rhs(fctn,gdata):
    u = np.load(fctn)
    m = u.shape[0]
    gdata2 = grid_data(m,[lambda x: .95*np.cos(x),lambda x: .95*np.sin(x)],2)
    odata = operator_data(gdata2,2)
    lap_u = odata.lap(u) + odata.lap(u.T).T
    lap_u_int = np.zeros(np.sum(gdata.flag))
    counter = 0
    for i in range(gdata.m):
        for j in range(gdata.m):
            if gdata.flag[i,j] == True:
                x = gdata.x1[i,j]
                y = gdata.x2[i,j]
                delta = dirac([x,y],gdata2.x,order='spectral')
                ev = np.dot(delta,lap_u.flatten())
                lap_u_int[counter] = ev
                counter += 1
    u_bdry = np.zeros(gdata.b.shape[0])
    for i in np.arange(gdata.b.shape[0]):
        pt = gdata.b[i,:]
        delta = dirac(pt,gdata2.x,order='spectral')
        ev = np.dot(delta,u.flatten())
        u_bdry[i] = ev
    rhs = np.hstack((lap_u_int,u_bdry))
    np.save(fctn[:-4]+str(gdata.m)+'_rhs',rhs)
    return rhs

if __name__ == '__main__':
    gdata = grid_data(2048,[lambda x: .95*np.cos(x),lambda x: .95*np.sin(x)],2)
    u = create_random_function(decay=3)
    gdata.contour(u)
