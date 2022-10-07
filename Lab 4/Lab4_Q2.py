
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


#print(H_matrix)

print(np.linalg.eigh(H_matrix)[0] / 1.60218e-19)

H_matrix = H(100,100)

print(np.linalg.eigh(H_matrix)[0] / 1.60218e-19)

w, v = np.linalg.eigh(H_matrix)
x = np.linspace(0, L, len(v[:,0]))

def Psi(x, v):
    Psi = x*0
    for i in range(1, len(x)+1):
        Psi += v[i-1]*np.sin(i*np.pi*x/L)
    return Psi


v = v*-1
ground_state = Psi(x, v[:,0])
first_excited = Psi(x, v[:,1])
second_excited = Psi(x, v[:,2])

plt.plot(x, Psi(x, v[:,0]))
plt.plot(x, Psi(x, v[:,1]))
plt.plot(x, Psi(x, v[:,2]))
plt.show()

def Normalization(x, Psi):
    N = 100
    a = 0
    b = L
    delta_x = (b-a)/N
    
    def Simpson(N):
    	s = Psi[0]**2 + Psi[-1]**2
    	for i in range(1, len(x)):
    		if i % 2 == 0:
    			s += 2*(Psi[i]**2)
    		else:
    			s += 4*(Psi[i]**2)
    	return s * delta_x/ 3
    
    return Simpson(N)

A1 = Normalization(x, ground_state)
A2 = Normalization(x, first_excited)
A3 = Normalization(x, second_excited)


plt.plot(x, ground_state**2 / A1)
plt.plot(x, first_excited**2 / A2)
plt.plot(x, second_excited**2 / A3)
plt.legend(['Ground state','First excited','Second excited'])
plt.xlabel('Position x (meters)')
plt.ylabel('Probability density')
plt.title('Probability density vs position')
plt.ylim(0, 7e9)


