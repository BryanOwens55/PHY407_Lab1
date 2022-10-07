
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constant

a = 10 * 1.60218e-19 # eV
L = 5 * 1e-10 #e-10 # Meters           Angstroms
h = constant.hbar #6.582e-16  # eV*s
M = 9.1094e-31 #0.5109989461 # eV/c^2       # kg

def H(m, n):
    H = np.zeros((m,n))
    for i in range(1, m+1):
        for j in range(1, n+1):
            if i == j:
                H[i-1,j-1] = 0.5*a + (np.pi**2 * h**2 * i**2) / (2 * M * L**2)
            elif (i - j) % 2 != 0:
                H[i-1,j-1] = -1 * (8*a*i*j) / ((np.pi**2)*(i**2 - j**2)**2)
            else:
                H[i-1,j-1] = 0
    return H



#print(H(3,3))



H_matrix = H(10,10)


print(H_matrix)

print(np.linalg.eigh(H_matrix)[0] / 1.60218e-19)

w, v = np.linalg.eigh(H_matrix)
x = np.linspace(0, L, len(v[:,0]))





#print(0.5*a + (np.pi**2 * h**2 * 1**2) / (2 * M * L**2))


