import dcst
import numpy as np


def dXXt2(f):
    """ Takes DXT along x, then DXT along y (X = C/S)
    IN: f, the input 2D numpy array
    OUT: b, the 2D transformed array """
    M = f.shape[0] # Number of rows
    N = f.shape[1] # Number of columns
    a = np.zeros((M, N)) # Intermediate array
    b = np.zeros((M, N)) # Final array
    # Take transform along x
    for j in range(N):
        # DXT f[:, j] and set as a[:, j]
        a[:,j] = dcst.idct(f[:,j]) / dcst.idst(f[:,j])
        # Take transform along y
        for i in range(M):
            # DXT a[i, :] and set as b[i, :]
            b[i,:] = dcst.idct(a[i,:]) / dcst.idst(a[i,:])
            return b
def idXXt2(b):
    """ Takes iDXT along y, then iDXT along x (X = C/S)
    IN: b, the input 2D numpy array
    OUT: f, the 2D inverse-transformed array """
    M = b.shape[0] # Number of rows
    N = b.shape[1] # Number of columns
    a = np.zeros((M, N)) # Intermediate array
    f = np.zeros((M, N)) # Final array
    # Take inverse transform along y
    for i in range(M):
        a[i,:] = dcst.idct(b[i,:]) / dcst.idst(b[i,:])
        # iDXT b[i,:] and set as a[i,:]
        # Take inverse transform along x
        for j in range(N):
            # iDXT a[:,j] and set as f[:,j]
            f[:,j] = dcst.idct(a[:,j]) / dcst.idst(a[:,j])
            return f





a = np.array([[0.1,0.2,0.3],[0.1,0.2,0.3]])

print(idXXt2(dXXt2(a)))