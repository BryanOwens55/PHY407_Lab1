
import numpy as np
import matplotlib.pyplot as plt
from time import time




# PART B



def f(x):
	return 4/(1+x**2)

n = 2 # 4 for Simpson, 12 for Trapezoidal
N = 2**n
a = 0
b = 1
delta_x = (b-a)/N

def Trapezoidal(N):
	s = 0.5*f(a) + 0.5*f(b)
	for i in range(1, N):
		s += f(a + i*delta_x)
	return s * delta_x



def Simpson(N):
	s = f(a) + f(b)
	for i in range(1, N):
		if i % 2 == 0:
			#print(2*f(a + i*delta_x))
			s += 2*(f(a + i*delta_x))
		else:
			#print(4*f(a + i*delta_x))
			s += 4*(f(a + i*delta_x))
	return s * delta_x/ 3


print(abs(Trapezoidal(N) - np.pi), abs(Simpson(N) - np.pi))


# PART C

## 4 for Simpson, 12 for Trapezoidal

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

N_2 = 32
N_1 = 16

epsilon_1 = (Trapezoidal(N_2) - Trapezoidal(N_1)) / 3

#print(epsilon_1)




