'''
Authors: Bryan Owens, Dharmik Patel
Purpose: To study the trapezoidal and Simpson methods of integration
Collaboration: Code was evenly created and edited by both lab partners
'''

# Import numpy and the time function
import numpy as np
from time import time


# PART B

# Take in a value of x and return the value of the integrand f(x)
def f(x):
	return 4/(1+x**2)

# Set up variables for slices (N), upper and lower bounds (b and a respectively)
# and a value for delta x
n = 2
N = 2**n
a = 0
b = 1
delta_x = (b-a)/N

# Method that takes in number of steps N and returns the value of the integral 
# using f(x) from previous method and using the trapezoidal method
def Trapezoidal(N):
	s = 0.5*f(a) + 0.5*f(b)
	for i in range(1, N):
		s += f(a + i*delta_x)
	return s * delta_x


# Method that takes in number of steps N and returns the value of the integral 
# using f(x) from previous method and using the Simpson method
def Simpson(N):
	s = f(a) + f(b)
	for i in range(1, N):
		if i % 2 == 0:
			s += 2*(f(a + i*delta_x))
		else:
			s += 4*(f(a + i*delta_x))
	return s * delta_x/ 3

# Printout of the deviation from true value for trapezoidal and Simpson method
print('Deviation from true value (Trapezoidal): ', abs(Trapezoidal(N) - np.pi))
print('Deviation from true value (Simpson): ', abs(Simpson(N) - np.pi))


# PART C

# Tme how long it takes for trapezoidal and Simpson method to compute the integral
# 200 times with an error on the order of O(1e-9)
start  = time()
for i in range(200):
	Trapezoidal(2**12)
end = time()
print(end - start)

start  = time()
for i in range(200):
	Simpson(2**4)
end = time()
print(end - start)


# PART D

# Calculate and print the practical estimation of errors for trapezoidal method
N_2 = 32
N_1 = 16
epsilon_1 = (Trapezoidal(N_2) - Trapezoidal(N_1)) / 3
print(epsilon_1)


