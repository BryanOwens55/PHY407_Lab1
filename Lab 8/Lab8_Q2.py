"""
Authors: Bryan Owens, Dharmik Patel
Purpose: To simulate the shallow water system using the FTCS method
Collaberation: Code was evenly created and edited by both lab partners
"""
# Imports
from numpy import empty 
import numpy as np
from pylab import plot,xlabel,ylabel,show 

# Constants 
L = 1 # Length of system (meters)
N = 50 # Number of points on line
a = L/N # Length between points (meters)
h = 0.01 # Time-step
mu = 0.05 # meters
sigma = 0.05 # meters
H = 0.01 # meters
nu_b = 0 # Fixed bottom topography
A = 0.002 # meters
g = 9.81 # m/s^2

epsilon = h/1000 
 

t1 = 0.0 
t2 = 1.0
t3 = 4.0 
tend = t3 + epsilon


# position array
x = np.linspace(0,L,N+1) 

# Set up inital conditions for eta and u
eta = empty(N+1,float)
eta = H + A*np.exp(-1*(x-mu)**2/sigma**2) 
eta_next = empty(N+1,float) 
eta_next = H + A*np.exp(-1*(x-mu)**2/sigma**2)-np.mean(A*np.exp(-1*(x-mu)**2/sigma**2))
eta_b = 0

u = empty(N+1,float)
u[0] = 0 #Thi 
u[N] = 0 #Tlo 
u[1:N] = 0
u_next = empty(N+1,float) 
u_next[0] = 0#Thi 
u_next[N] = 0#Tlo 

# Main loop 
t = 0.0 
plot(x,eta)
show()
while t<tend: 
    # Calculate the new values of u and eta
    u_next[0] = u[0] + (u[0] - u[1])/a
    u_next[-1] = u[-1] + (u[-1] - u[-2])/a
    u_next[1:N] = u[1:N] - h*((u[2:N+1]**2 - u[0:N-1]**2)/2 - g*(eta[2:N+1] - eta[0:N-1]))/(2*a)
    
    eta_next[0] = (eta[0] - eta[1])/a
    eta_next[-1] = (eta[-1] - eta[-2])/a
    eta_next[1:N] = eta[1:N] - h*(u[2:N+1]*(eta[2:N+1] - eta_b) - u[0:N-1]*(eta[0:N-1] - eta_b))/(2*a)
    
    
    eta,eta_next = eta_next,eta 
    u,u_next = u_next,u 
    
    t += h 
    
    # Make plots at the given times 
    if abs(t-t1)<epsilon: 
        print('hi')
        plot(x,eta) 
    if abs(t-t2)<epsilon: 
        plot(x,eta) 
    if abs(t-t3)<epsilon: 
        plot(x,eta) 
xlabel ("x") 
ylabel("T") 
show() 


# In[ ]:




