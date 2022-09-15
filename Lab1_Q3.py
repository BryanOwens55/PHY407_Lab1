# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import time



sizes = np.arange(2, 200, 1)
times = np.zeros(len(sizes))

for i in range(len(sizes)):
    A = np.ones([i,i])*3
    B = np.ones([i,i])*3
    C = np.zeros([i,i])
    start = time()
    for j in range(len(C)): 
        for k in range(len(C)): 
            for l in range(len(C)): 
                C[j,k] += A[j,l]*B[l,k]
    end = time()
    times[i] = end - start

print(times)
plt.plot(sizes, times)
plt.xlabel('Size of array (NxN)')
plt.ylabel('Time to square matrix')
plt.title('Size of array vs multiplication time')
plt.show()



times = times * 0

for i in range(len(sizes)):
    A = np.ones([i,i])*3
    B = np.ones([i,i])*3
    C = np.zeros([i,i])
    start = time()
    C = np.dot(A,B)
    end = time()
    times[i] = end - start



print(times)
plt.plot(sizes, times)
plt.xlabel('Size of array (NxN)')
plt.ylabel('Time to square matrix')
plt.title('Size of array vs multiplication time')
plt.show()

