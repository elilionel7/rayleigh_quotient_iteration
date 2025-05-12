# delta.py
import numpy as np
from scipy.sparse import csr_matrix, kron

def dirac(x,grid,order='cubic'):
    if order == 'spectral':
        if type(x) in [np.float64, np.int32, int, float]:
            m = grid.shape[0]
            fs = np.arange(m)*1.0
            w = (-1)**fs*np.sin((2*fs+1)*np.pi/(2*m)) 
            xx1 = x - grid
            xx1 = w/xx1
            xx1 = xx1/np.sum(xx1)
            return xx1
        elif len(x) == 2:
            x_delta = dirac(x[0],grid,order=order)
            y_delta = dirac(x[1],grid,order=order)
            return np.outer(x_delta,y_delta).flatten()
    elif order == 'cubic':
        if type(x) in [np.float64, np.int32, int, float]:
            x = np.arccos(x)
            x_floor = np.floor(((x/np.pi*2*len(grid))-1)/2)
            x_dec = ((x/np.pi*2*len(grid)-1)/2 % 1.0)
            row = np.array([0,0,0,0])
            col = np.array([x_floor-1,x_floor,x_floor+1,x_floor+2])
            data = np.array([x_dec*(x_dec-1)*(x_dec-2)/-6,
                             (x_dec+1)*(x_dec-1)*(x_dec-2)/2,
                             (x_dec+1)*x_dec*(x_dec-2)/-2,
                             (x_dec+1)*x_dec*(x_dec-1)/6])
            return csr_matrix((data,(row,col)),shape=(1,len(grid)))
        elif len(x) == 2:
            x_delta = dirac(x[0],grid,order=order)
            y_delta = dirac(x[1],grid,order=order)
            return csr_matrix.reshape(kron(x_delta,y_delta.T),(1,len(grid)**2))
        
def ddirac(x,grid,order='spectral'):
    if order == 'spectral':
        m = grid.shape[0]
        fs = np.arange(m)*1.0
        w = (-1)**fs*np.sin((2*fs+1)*np.pi/(2*m)) 
        xx = w/(x-grid)
        xx2 = w/(x-grid)**2
        a = -xx2/np.sum(xx) + xx * np.sum(xx2) / np.sum(xx)**2
        return a
    
def dvdirac(x,v,grid,order='spectral'):
    a=v[0]*np.outer(ddirac(x[0],grid,order),dirac(x[1],grid,order)).flatten()
    b=v[1]*np.outer(dirac(x[0],grid,order),ddirac(x[1],grid,order)).flatten()
    return a+b