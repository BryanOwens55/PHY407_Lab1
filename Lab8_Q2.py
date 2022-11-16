#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import empty 
import numpy as np
from pylab import plot,xlabel,ylabel,show, ylim, xlim, clf, pause, draw
import matplotlib.pyplot as plt
# Constants 
L = 1 
N = 50
a = L/N 
h = 0.01
mu = 0.5
sigma = 0.05
H = 0.01
A = 0.002
g = 9.81

epsilon = h/1000 
 


t1 = 0.1 
t2 = 1#1.0 
t3 = 4.0 
tend = t3 + epsilon


# In[8]:
x = np.linspace(0,L,N+1) 

eta = empty(N+1,float)
eta = H + A*np.exp(-1*(x-mu)**2/sigma**2)-np.mean(A*np.exp(-1*(x-mu)**2/sigma**2))
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
ylim(bottom=0, top=0.02)
xlim(0,1)
plt.fill_between(x,eta,color='cyan')
plt.title("Shallow Water Simulation t=" + str(round(t,3)) + 's')
plt.xlabel("Spatial Dimension (m)")
plt.ylabel("Free Surface Altitude ($\eta$)")        
plt.show()


C = h/2/a
while t<tend: 
    # Calculate the new values of T 
    u_next[0] = u[0] - C*(0.5*u[1]**2 + g*eta[1] - 0.5*u[0]**2 - g*eta[0])
    u_next[-1] = u[N] - C*(0.5*u[N]**2 + g*eta[N] - 0.5*u[N-1]**2 - g*eta[N-1])
    u_next[1:N] = u[1:N] - C*(0.5*u[2:N+1]**2 + g*eta[2:N+1] - 0.5*u[0:N-1]**2 - g*eta[0:N-1])
    
    eta_next[0] = eta[0] - C*(u[1]*(eta[1] - eta_b) - u[0]*(eta[0] - eta_b))
    eta_next[-1] = eta[N] - C*(u[N]*(eta[N] - eta_b) - u[N-1]*(eta[N-1] - eta_b))
    eta_next[1:N] = eta[1:N] - C*(u[2:N+1]*(eta[2:N+1] - eta_b) - u[0:N-1]*(eta[0:N-1] - eta_b))
    

    eta = np.copy(eta_next)
    u = np.copy(u_next)
    
    t += h 
    # Make plots at the given times 
    if abs(t-t1)<epsilon: 
        plot(x,eta) 
        ylim(bottom=0, top=0.02)
        xlim(0,1)
        plt.fill_between(x,eta,color='cyan')
        plt.title("Shallow Water Simulation t=" + str(round(t,3)) + 's')
        plt.xlabel("Spatial Dimension (m)")
        plt.ylabel("Free Surface Altitude ($\eta$)")        
        plt.show()
    if abs(t-t2)<epsilon: 
        plot(x,eta) 
        ylim(bottom=0, top=0.02)
        xlim(0,1)
        plt.fill_between(x,eta,color='cyan')
        plt.title("Shallow Water Simulation t=" + str(round(t,3)) + 's')
        plt.xlabel("Spatial Dimension (m)")
        plt.ylabel("Free Surface Altitude ($\eta$)")          
        plt.show()
    if abs(t-t3)<epsilon: 
        plot(x,eta)
        ylim(bottom=0, top=0.02)
        xlim(0,1)
        plt.fill_between(x,eta,color='cyan')
        plt.title("Shallow Water Simulation t=" + str(round(t,3)) + 's')
        plt.xlabel("Spatial Dimension (m)")
        plt.ylabel("Free Surface Altitude ($\eta$)")           
        plt.show()
    '''
    clf()
    plt.title("Shallow Water Simulation t=" + str(round(t,3)),fontsize=16)
    plt.xlabel("Spatial Dimension (m)",fontsize=14)
    plt.ylabel("Free Surface Altitude",fontsize=14)
    plt.plot(x,eta,marker='o',c='g')
    plt.fill_between(x,y1=0,y2=eta,color='b',alpha=0.4)
    plt.ylim([0,0.02])
    plt.tight_layout()

    draw()
    pause(0.001)
    '''

show() 



'''
while t<tend: 
    # Calculate the new values of T 
    u_next[0] = u[0] + h*(u[0] - u[1])/a
    u_next[-1] = u[-1] + h*(u[-1] - u[-2])/a
    u_next[1:N] = u[1:N] - h*((u[2:N+1]**2 - u[0:N-1]**2)/2 - g*(eta[2:N+1] - eta[0:N-1]))/(2*a)
    
    eta_next[0] = eta[0] + h*(eta[0] - eta[1])/a
    eta_next[-1] = eta[-1] + h*(eta[-1] - eta[-2])/a
    eta_next[1:N] = eta[1:N] - h*(u[2:N+1]*(eta[2:N+1] - eta_b) - u[0:N-1]*(eta[0:N-1] - eta_b))/(2*a)
    

    eta = np.copy(eta_next)
    u = np.copy(u_next)
    
    t += h 
    # Make plots at the given times 
    if abs(t-t1)<epsilon: 
        plot(x,eta) 
    if abs(t-t2)<epsilon: 
        plot(x,eta) 
    if abs(t-t3)<epsilon: 
        plot(x,eta) 
    

xlabel ("x") 
ylabel("T") 
show() 


# In[ ]:
'''



