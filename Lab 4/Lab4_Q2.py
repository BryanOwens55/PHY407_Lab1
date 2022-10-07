# Import numpy, matplotlib, scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constant

# Save necessary constants
a = 10 * 1.60218e-19 # Joules
L = 5 * 1e-10 # Meters
h = constant.hbar # Joule * s
M = 9.1094e-31 # kg

# Method that calculates the matrix H for given dimensions
def H(m, n):
    # Create H with zeros of size mxn
    H = np.zeros((m,n))
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Checks that see what the value should
            if i == j:
                H[i-1,j-1] = 0.5*a + (np.pi**2 * h**2 * i**2) / (2 * M * L**2)
            elif (i - j) % 2 != 0:
                H[i-1,j-1] = -1 * (8*a*i*j) / ((np.pi**2)*(i**2 - j**2)**2)
            else:
                H[i-1,j-1] = 0
    return H


# Calculate and print the eigenvalues of H for size 10x10
H_matrix = H(10,10)
print(np.linalg.eigh(H_matrix)[0] / 1.60218e-19)

# Calculate and print the eigenvalues of H for size 100x100
H_matrix = H(100,100)
print(np.linalg.eigh(H_matrix)[0] / 1.60218e-19)

# Calculate the eigenvectors for the matrix H for size 100x100
w, v = np.linalg.eigh(H_matrix)
x = np.linspace(0, L, len(v[:,0]))

# Method that calculates the wave-function psi for a given state
def Psi(x, v):
    Psi = x*0
    for i in range(1, len(x)+1):
        Psi += v[i-1]*np.sin(i*np.pi*x/L)
    return Psi


# Calculate and plot the wavefunction psi
v = v*-1
ground_state = Psi(x, v[:,0])
first_excited = Psi(x, v[:,1])
second_excited = Psi(x, v[:,2])

plt.plot(x, Psi(x, v[:,0]))
plt.plot(x, Psi(x, v[:,1]))
plt.plot(x, Psi(x, v[:,2]))
plt.show()


# Method that normalizes the wave function by calculating the integral 
# of the probability density
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

# Calculate the normalization constants
A1 = Normalization(x, ground_state)
A2 = Normalization(x, first_excited)
A3 = Normalization(x, second_excited)

# Plot the probability wave functions of the asymmetric quantum well
plt.plot(x, ground_state**2 / A1)
plt.plot(x, first_excited**2 / A2)
plt.plot(x, second_excited**2 / A3)
plt.legend(['Ground state','First excited','Second excited'])
plt.xlabel('Position x (meters)')
plt.ylabel('Probability density')
plt.title('Probability density vs position')
plt.ylim(0, 7e9)


