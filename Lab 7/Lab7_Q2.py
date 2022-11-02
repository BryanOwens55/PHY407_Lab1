"""
Authors: Bryan Owens, Dharmik Patel
Purpose: To calculate the energies and the eigenvector of the radial
         component of the hydrogen atom for a given state n and l
Collaberation: Code was evenly created and edited by both lab partners
"""

# Import methods
from numpy import array,arange
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const 

# Constants
m = 9.1094e-31     # Mass of electron
hbar = 1.0546e-34  # Planck's constant over 2*pi
e = 1.6022e-19     # Electron charge
a = 5.2918e-11     # Bohr radius
N = 1000
h = 0.002*a
l = 0
n = 1

# Potential function
def V(x):
    v = -1*e**2/(4*const.pi*const.epsilon_0*x)
    return v

# Method that returns the components of the ODE
def f(r,x,E,l):
    R = r[0]
    S = r[1]
    dR = S/x**2
    dS = (2*m*x**2/hbar**2)*(V(x)-E)*R + l*(l+1)*R
    return array([dR,dS],float)

# Calculate the wavefunction for a particular energy
def solve(E,l):
    psi = 0.0
    phi = 1.0
    r = array([psi,phi],float)

    for x in arange(h,20*a,h):
        k1 = h*f(r,x,E,l)
        k2 = h*f(r+0.5*k1,x+0.5*h,E,l)
        k3 = h*f(r+0.5*k2,x+0.5*h,E,l)
        k4 = h*f(r+k3,x+h,E,l)
        r += (k1+2*k2+2*k3+k4)/6

    return r[0]

# Main program to find the energy using the secant method
E1 = -15*e/n**2
E2 = -13*e/n**2
psi2 = solve(E1,0)

target = e/10000
while abs(E1-E2)>target:
    psi1,psi2 = psi2,solve(E2,0)
    E1,E2 = E2,E2-psi2*(E2-E1)/(psi2-psi1)

# Print final energy
print("E =",E2/e,"eV")

# Method that returns the eigenvector with given energy and l values
def ODE(E,l):
    psi = 0.0
    phi = 1.0
    r = array([psi,phi],float)
    R = [r[0]]

    for x in arange(h,20*a,h):
        k1 = h*f(r,x,E,l)
        k2 = h*f(r+0.5*k1,x+0.5*h,E,l)
        k3 = h*f(r+0.5*k2,x+0.5*h,E,l)
        k4 = h*f(r+k3,x+h,E,l)
        r += (k1+2*k2+2*k3+k4)/6
        R.append(r[0])
    return np.array(R)

R = ODE(E2,0)
r = np.linspace(h,20*h,len(R))
plt.plot(r,R)

# method that returns the two energy values in the secant method
def shooting(E1,E2,l):  
    target = e/10000
    psi1 = 1.0
    psi2 = solve(E1,l)
    while abs(E1-E2)>target:
        psi1,psi2 = psi2,solve(E2,l)
        E1,E2 = E2,E2-psi2*(E2-E1)/(psi2-psi1)
    return E1,E2

# Method that returns the eigenvector for a given state n and l
def Nth_state(n,l):
    E1 = -15*e/n**2
    E2 = -13*e/n**2
    E1,E2 = shooting(E1,E2,l)
    R = ODE(E2,l)
    return R

# Plot non-normalized states
r = np.linspace(h,20*h,len(R))
plt.plot(r,Nth_state(1,0))
plt.plot(r,Nth_state(2,0))
plt.plot(r,Nth_state(2,1))
plt.show()

# Method that integrates wavefunction to find normalization constant
def integration(a,b,R):
    R = (np.array(R))**2
    h = (b-a)/len(R)
    s = 0.5*R[0] + 0.5*R[-1]
    for k in range(1,len(R)):
        s += R[k]
    return s*h

# Plot the first three normalized eigen states
R1 = Nth_state(1,0) 
norm1 = integration(0,20*a,R1)
plt.plot(r,R1/(norm1)**0.5)

R2 = Nth_state(2,0) 
norm2 = integration(0,20*a,R2)
plt.plot(r,R2/(norm2)**0.5)


R3 = Nth_state(2,1) 
norm3 = integration(0,20*a,R3)
plt.plot(r,R3/(norm3)**0.5)

plt.xlabel('Radius r (m)')
plt.ylabel('R(r)')
plt.title('Radial Function of Hydrogen Atom')
plt.legend(['n=1,l=0','n=2,l=0','n=2,l=1'])
plt.savefig('R(r).png')


