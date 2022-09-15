'''
Authors: Bryan Owens, Dharmik Patel
Purpose: To time how long it takes to compute a dot product of two matrices 
         of size NxN using two different methods: three for-loops and numpys function
Collaboration: Code was evenly created and edited by both lab partners
'''
# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Using textbook method for multiplying two matrices

# Create arrays for the different sizes/dimensions of array, and a time array
# to keep track of calculation length
sizes = np.arange(2, 300, 1)
times = np.zeros(len(sizes))

# First for-loop to loop through all dimensions of matrix from 2 to 300
for i in range(len(sizes)):
    # Matrix A and B will be multiplied together, the resulting matrix is C
    A = np.ones([i,i])*3
    B = np.ones([i,i])*3
    C = np.zeros([i,i])
    # Start the clock for the calculation
    start = time()
    # Three for-loops to loop through matrices A, B, and C
    for j in range(len(C)): 
        for k in range(len(C)): 
            for l in range(len(C)): 
                C[j,k] += A[j,l]*B[l,k]
    # Stop the clock and append calculation time to times array
    end = time()
    times[i] = end - start

# Print and plot the times taken for each size of matrix from 2 to 300
print(times)
plt.plot(sizes, times)
plt.xlabel('Size of array (NxN)')
plt.ylabel('Time to square matrix')
plt.title('Size of array vs multiplication time')
plt.show()


# Reset time array
times = times * 0

# First for-loop to loop through all dimensions of matrix from 2 to 300
for i in range(len(sizes)):
    # Set up same matrices A, B, C
    A = np.ones([i,i])*3
    B = np.ones([i,i])*3
    C = np.zeros([i,i])
    # Start clock for timing calculation
    start = time()
    # Using numpy function to calculate dot product
    C = np.dot(A,B)
    # Stop clock and append calculation time to times array 
    end = time()
    times[i] = end - start


# Print and plot the times taken for each size of matrix from 2 to 300
print(times)
plt.plot(sizes, times)
plt.xlabel('Size of array (NxN)')
plt.ylabel('Time to square matrix')
plt.title('Size of array vs multiplication time')
plt.show()

