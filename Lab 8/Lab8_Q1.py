"""
Authors: Bryan Owens, Dharmik Patel
Purpose: To solve Laplace's equation for an electronic capacitor
Collaberation: Code was evenly created and edited by both lab partners
"""
# Imports
from numpy import empty,zeros,max 
from pylab import imshow,gray,show
import matplotlib.pyplot as plt 
from time import time

# Constants 
M = 100 # grid (M+1)x(M+1)
V = 1.0 # Voltage
target = 1e-6 


def potential(omega):
    # Create initial condition
    phi = zeros([M+1,M+1] ,float) 
    phi[20:-20,20] = V 
    phi[20:-20,80] = -V 
    
    # Main loop 
    delta = 1.0 
    while delta>target: 
        # Save old values
        old = phi.copy()
        for i in range(M+1): 
            for j in range(M+1):
                # If at boundaries
                if i==0 or i==M or j==0 or j==M: 
                    phi[i,j] = phi[i,j] 
                # If at capacitor
                elif 80>=i>=20 and j==20:
                    phi[i,j] = phi[i,j]
                elif 80>=i>=20 and j==80:
                    phi[i,j] = phi[i,j]
                else: 
                    phi[i,j] = (1+omega)*(phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])/4 - phi[i,j]*omega
        # Calculate maximum difference from old values 
        delta = max(abs(old-phi))

    # Plot solution
    fig = plt.figure(figsize=(6, 3))
    potential = plt.contourf(phi)
    plt.title('Potential of an Electronic Capacitor')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    cbar = fig.colorbar(potential)
    cbar.set_label('Potential $V$')
    gray() 
    show() 
    return


# Part a
# Running the code with no over-relaxation (omega = 0)
start = time()
potential(0)
print(time() - start)

# Part b
# Running the code with over-relaxation, and with omega = 0.1, 0.5

start = time()
potential(0.1)
print(time() - start)

start = time()
potential(0.5)
print(time() - start)


