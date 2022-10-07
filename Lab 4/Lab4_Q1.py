



# SolveLinear.py
# Python module for PHY407
# Paul Kushner, 2015-09-26
# Modifications by Nicolas Grisouard, 2018-09-26
# This module contains useful routines for solving linear systems of equations.
# Based on gausselim.py from Newman
from numpy import array, empty, argmax, copy
from numpy.linalg import solve
import numpy as np
from time import time
import matplotlib.pyplot as plt
# The following will be useful for partial pivoting
# from numpy import empty, copy


def GaussElim(A_in, v_in):
    """Implement Gaussian Elimination. This should be non-destructive for input
    arrays, so we will copy A and v to
    temporary variables
    IN:
    A_in, the matrix to pivot and triangularize
    v_in, the RHS vector
    OUT:
    x, the vector solution of A_in x = v_in """
    # copy A and v to temporary variables using copy command
    A = copy(A_in)
    v = copy(v_in)
    N = len(v)

    for m in range(N):
        # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div

        # Now subtract from the lower rows
        for i in range(m+1, N):
            mult = A[i, m]
            A[i, :] -= mult*A[m, :]
            v[i] -= mult*v[m]

    # Backsubstitution
    # create an array of the same type as the input array
    x = empty(N, dtype=v.dtype)
    for m in range(N-1, -1, -1):
        x[m] = v[m]
        for i in range(m+1, N):
            x[m] -= A[m, i]*x[i]
    return x


# Part A

def PartialPivot(A_in, v_in):
    """ In this function, code the partial pivot (see Newman p. 222) """
    # copy A and v to temporary variables using copy command
    A = copy(A_in)
    v = copy(v_in)
    N = len(v)
    
    for m in range(N):
        # Find the row with greatest value at the mth column from
        # row m and below
        flip = argmax(abs(A[m:,m]))+m
        # Now flip the two rows for both the matrix A and the matrix v
        A[[m, flip]] = A[[flip,m]]
        v[[m, flip]] = v[[flip,m]]

        # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div

        # Now subtract from the lower rows
        for i in range(m+1, N):
            mult = A[i, m]
            A[i, :] -= mult*A[m, :]
            v[i] -= mult*v[m]

    # Backsubstitution
    # create an array of the same type as the input array
    x = empty(N, dtype=v.dtype)
    for m in range(N-1, -1, -1):
        x[m] = v[m]
        for i in range(m+1, N):
            x[m] -= A[m, i]*x[i]
    return x

# Method that uses scipy solve to calculate x using LU decomposition
def LU(A, v):
    return solve(A, v)



# Same array from example 6.2
A = array([[2,  1,  4,  1],
              [3,  4, -1, -1],
              [1, -4,  1,  5],
              [2, -2,  1,  3]], float)

v = array([-4, 3, 9, 7], float)

# Print the value that is calculated by both the Gauss Elimination method and
# the Partial Pivot method
print(PartialPivot(A, v))
print(GaussElim(A, v))




# Part B

# Method that calulates the run-time and the error of the three different methods
# for a range of matrix sizes 1 to N
def Timing(N):
    gauss_times = np.zeros(N)
    pivot_times = np.zeros(N)
    LU_times = np.zeros(N)
    err1 = np.zeros(N)
    err2 = np.zeros(N)
    err3 = np.zeros(N)
    
    # for loop to calculate the run-times and errors of a matrix of size i
    for i in range(1, N):
        A = np.random.randint(9, size=(N, N))
        v = np.random.randint(9, size=(N))
        
        start = time()
        x1 = PartialPivot(A.astype(float), v.astype(float))
        end = time()
        pivot_times[i] = end - start
        err1[i] = np.mean(abs(v - np.dot(A, x1)))
        
        start = time()
        x2 = GaussElim(A.astype(float), v.astype(float))
        end = time()
        gauss_times[i] = end - start
        err2[i] = np.mean(abs(v - np.dot(A, x2)))
        
        start = time()
        x3 = LU(A.astype(float), v.astype(float))
        end = time()
        LU_times[i] = end - start
        err3[i] = np.mean(abs(v - np.dot(A, x3)))
     
    # Plot the errors on a log-log plot
    plt.loglog(err1)
    plt.loglog(err2)
    plt.loglog(err3)
    plt.legend(['Pivot', 'Gauss', 'LU'])
    plt.xlabel('NxN size of matrix A')
    plt.ylabel('Error')
    plt.title('Error vs size of matrix')
    plt.show()
    return pivot_times, gauss_times, LU_times

# Plot the run-times on a log-log plot
times1, times2, times3 = Timing(50)
plt.plot(times1)
plt.plot(times2)
plt.plot(times3)
plt.legend(['Pivot', 'Gauss', 'LU'])
plt.xlabel('NxN size of matrix A')
plt.ylabel('Runtime')
plt.title('Runtime vs size of matrix')



