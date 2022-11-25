"""
Authors: Bryan Owens, Dharmik Patel
Purpose: To solve the time-dependant Schrodinger equation for a particle in an
         infinite square well
Collaberation: Code was evenly created and edited by both lab partners
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

# Constants
L = 1e-8
m = 9.109e-31
sigma = L / 25
k = 500 / L
tau = 1e-18  # time step
x0 = L / 5
N = 3000  # Iteration of time steps
T = N * tau
P = 1024  # Number of segments
V_0 = 6e-17
interval = L / P



psi_0 = 1/((2*np.pi*(sigma**2))**(1/4)) # Analytical solution

# Method that constucts initial state of Psi
def Psi_initial(x):
    return psi_0 * np.exp(-((x-x0)**2)/(4*sigma**2) + 1j*k*x)

# Plot initial condition
x = np.linspace(-L/2, L/2, P-1)
plt.plot(x,np.real(np.conj(Psi_initial(x))*Psi_initial(x)))
plt.legend(['T=0'])
plt.show()

# Potential
def V(x):
    return 0




# Method that constructs a Hmiltonian given a potential and spatial step size
def hamiltonian_matrix(V, N):

    A = -(sc.hbar**2)/(2*m*(interval**2))
    B_p = np.zeros(P-1)
    # create array of B values
    for i in range(P-1):
        B_p[i] = V(i*interval - L/2) - (2*A)
    # Loop through and fill out hamiltonian matrix
    H_D = np.zeros((P-1,P-1))
    for a in range(P-1):
        for b in range(P-1):
            if a == b:
                H_D[a,b] = B_p[a]
            elif b == a - 1 or b == a + 1:
                H_D[a,b] = A
    return H_D

# create hamiltonian matrix with potential for this lap
H_matrix = hamiltonian_matrix(V, P)


# Constrcut L and R matrix from lab handout 
I = np.diag(np.full(P-1, 1.0), 0)
i = complex(0,1) 
L_matrix = I + (i*tau/(2*sc.hbar))*H_matrix
R_matrix = I - (i*tau/(2*sc.hbar))*H_matrix


# time array and psi(x,t) array
psi = np.zeros((P-1,N), dtype=complex)
psi[:,0] = Psi_initial(x)
t_array = np.linspace(0,T,N)


# Position array
x = np.zeros(P-1)
for i in range(P-1):
    x[i] = (i+1)*interval-L/2
    
# Method that calculates the normalization constant using dot product
def Norm(psi):
    return np.real(np.dot(np.conj(psi), psi))*interval

# Calculate the energy of the function Psi
def Energy(psi):
    H = hamiltonian_matrix(V, P)
    H_psi = np.dot(H, psi)
    return np.real(np.sum(np.dot(np.conj(psi), H_psi))) * interval

# Calculate the expectation value of the wavefunction
def Expectation(psi):
    x_psi = np.dot(x, psi)
    return np.real(np.sum(np.dot(np.conj(psi), x_psi))) * interval

# Create energy, normalization, and expectation arrays
energy_array = [Energy(psi[:,0])]
norm_array = [Norm(psi[:,0])]
expectation_array = [Expectation(psi[:,0])]


# Main time loop
for i in range(N-1):
    # Solve for v
    v = np.dot(R_matrix, psi[:,i])
    # Solve for the next step of psi
    psi[:,i+1] = np.linalg.solve(L_matrix, v)
    # calculate the normalization
    norm = Norm(psi[:,i+1])
    # Normalize
    psi[:,i+1] = psi[:,i+1]/norm
    # Append normalization, Energy, and expectation value to respected arrays
    norm_array.append(norm)
    energy_array.append(Energy(psi[:,i+1]))
    expectation_array.append(Expectation(psi[:,i+1]))
    
    print((i/(N-1))*100)

# Method that calculates the complex square of the wavelength
def complex_square(Psi):
    return np.real(np.conj(Psi)*Psi)


t = np.arange(0, N*tau, tau)

# Plot Wavelength, energy, expectation value, and normalization

plt.plot(x, complex_square(psi[:,0]), label = "$t = 0$")
plt.plot(x, complex_square(psi[:,750]), label = "$t = T/4$")
plt.plot(x, complex_square(psi[:,1500]), label = "$t = T/2$")
plt.plot(x, complex_square(psi[:,2250]), label = "$t = 3T/4$")
plt.plot(x, complex_square(psi[:,N-1]), label = "$t = T$")
plt.xlabel("x (m)")
plt.ylabel("$|\psi|^2$")
plt.title("Probability Density")
plt.legend()
plt.show()


plt.plot(t, expectation_array)
plt.title("Wavefunction Expectation value vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Expectation Value (m)")
plt.show()


plt.plot(t, energy_array)
plt.title("Energy vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.ylim(1e-17,2e-17)
plt.show()

plt.plot(t, norm_array)
plt.title("Wavefunction Normalization vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Normalization")
plt.ylim(0,2)
plt.show()

