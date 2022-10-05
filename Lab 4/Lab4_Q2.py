
import numpy as np
import matplotlib.pyplot as plt


a = 10 # eV
L = 5e-10 # Meters           Angstroms
h = 6.582e-16  # eV*s
M = 0.5109989461 # eV/c^2       #9.1094e-31 kg

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



H_matrix = H(100,100)

print()

#print(np.linalg.eigh(H(10,10))[0])

w, v = np.linalg.eigh(H_matrix)
x = np.linspace(0, L, len(v[:,0]))

plt.plot(x, v[:,0])
plt.plot(x, v[:,1])
plt.plot(x, v[:,2])
plt.show()


#print(np.linalg.eigvalsh(H_matrix))
    

def Normalization(Psi):
    N = len(Psi)
    a = 0
    b = L
    delta_x = (b-a)/N
    s = 0.5*(Psi[0]**2) + 0.5*(Psi[-1]**2)
    for i in range(1, N-1):
        s += (Psi[i]**2)
    A = s * delta_x
    return A


A = Normalization(v[:,0])


plt.plot(x, v[:,0] / np.sqrt(A))


print(Normalization(v[:,0]/np.sqrt(A)))

