#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Question 1

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time


# In[2]:


# Q1, part (a)

def rhs(r):
    """ The right-hand-side of the equations
    INPUT:
    r = [x, vx, y, vy] are floats (not arrays)
    note: no explicit dependence on time
    OUTPUT:
    1x2 numpy array, rhs[0] is for x, rhs[1] is for vx, etc"""

    M = 10.
    L = 2.

    x = r[0]
    vx = r[1]
    y = r[2]
    vy = r[3]

    r2 = x**2 + y**2
    Fx, Fy = - M * np.array([x, y], float) / (r2 * np.sqrt(r2 + .25 * L**2))

    return np.array([vx, Fx, vy, Fy], float)


# In[3]:


a = 0.0
b = 10.
N = 1000  # let's leave it at that for now
h = (b - a) / N

error_tolerated = 1e-6  #tolerated error per second, in meters per second

# Arrays for adaptive time stepping
tpoints = [0]
xpoints = []
vxpoints = []  # the future dx/dt
ypoints = []
vypoints = []  # the future dy/dt

# below: ordering is x, dx/dt, y, dy/dt
r = np.array([1., 0., 0., 1.], float)


# In[4]:


# Implementing adaptive time step RK4

adaptive_start = time.perf_counter() #starting the timer for the adaptive stepsize method

t = a #setting initial time to 'a' as defined above: a = 0.0 seconds

while t < b:
    
    r_size2h = r.copy()  #copying an array of r for RK4 with stepsize 2h
    r2_sizeh = r.copy()  #copying an array of r for RK4 with stepsize h implemented twice

    #Implementing the first step of RK4 with stepsize h
    
    xpoints.append(r[0])
    vxpoints.append(r[1])
    ypoints.append(r[2])
    vypoints.append(r[3])
    
    k1_h = h * rhs(r)  # all the k's are vectors
    k2_h = h * rhs(r + 0.5 * k1_h)  # note: no explicit dependence on time of the RHSs
    k3_h = h * rhs(r + 0.5 * k2_h)
    k4_h = h * rhs(r + k3_h)
    
    r2_sizeh += (k1_h + 2 * k2_h + 2 * k3_h + k4_h) / 6

    #Implementing the second step of RK4 with stepsize h
    
    r2_sizeh_2 = r2_sizeh.copy()

    k1 = h * rhs(r2_sizeh_2)  # all the k's are vectors
    k2 = h * rhs(r2_sizeh_2 + 0.5 * k1)  # note: no explicit dependence on time of the RHSs
    k3 = h * rhs(r2_sizeh_2 + 0.5 * k2)
    k4 = h * rhs(r2_sizeh_2 + k3)
    
    r2_sizeh_2 += (k1 + 2 * k2 + 2 * k3 + k4) / 6  

    #Implementing RK4 with stepsize 2h
    
    kp1_2h = 2 * h * rhs(r_size2h)  # all the k's are vectors
    kp2_2h = 2 * h * rhs(r_size2h + 0.5 * kp1_2h)  # note: no explicit dependence on time of the RHSs
    kp3_2h = 2 * h * rhs(r_size2h + 0.5 * kp2_2h)
    kp4_2h = 2 * h * rhs(r_size2h + kp3_2h)
    
    r_size2h += (kp1_2h + 2 * kp2_2h + 2 * kp3_2h + kp4_2h) / 6  

    #Calculating epsilon_x and epsilon_y (errors in x and y coordinates)
    
    eps_x = (1/30) * abs(r2_sizeh_2[0] - r_size2h[0])
    eps_y = (1/30) * abs(r2_sizeh_2[2] - r_size2h[2])

    # Calculate rho (ratio of tolerated error to actual error for stepsize h)
    
    rho = h * error_tolerated / np.sqrt(eps_x**2 + eps_y**2)

    #calculating the new step size and setting the stepsize to the new one
    
    h_new = h * abs(rho) ** (1 / 4)

    # If rho < 1, repeating the step
    
    if rho < 1:
        k1 = h_new * rhs(r)  # all the k's are vectors
        k2 = h_new * rhs(r + 0.5 * k1)  # note: no explicit dependence on time of the RHSs
        k3 = h_new * rhs(r + 0.5 * k2)
        k4 = h_new * rhs(r + k3)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h_new
        tpoints.append(t)

    #otherwise, keeping the outcome of the step
    
    else:
        r = r2_sizeh
        t += h
        tpoints.append(t)
    h = h_new

adaptive_end = time.perf_counter() #stopping the timer for the adaptive stepsize method

adaptive_time = adaptive_end - adaptive_start  #calculating the total time taken to run the adaptive stepsize method


# In[20]:


#Creating the arrays for non-adaptive stepsizes
N_nonadapt = 10000
h_nonadapt = (b - a) / N_nonadapt

t_nonadapt_points = np.arange(a, b, h_nonadapt)

x_nonadapt_points = []
v_nonadapt_xpoints = []  # the future dx/dt
y_nonadapt_points = []
v_nonadapt_ypoints = []  # the future dy/dt

# below: ordering is x, dx/dt, y, dy/dt
r_nonadapt = np.array([1., 0., 0., 1.], float)

nonadapt_start = time.perf_counter()

# Non-adaptive time step
for t in t_nonadapt_points:
    x_nonadapt_points.append(r_nonadapt[0])
    v_nonadapt_xpoints.append(r_nonadapt[1])
    y_nonadapt_points.append(r_nonadapt[2])
    v_nonadapt_ypoints.append(r_nonadapt[3])
    k1_nonadapt = h_nonadapt * rhs(r_nonadapt)  # all the k's are vectors
    k2_nonadapt = h_nonadapt * rhs(r_nonadapt + 0.5 * k1_nonadapt)  # note: no explicit dependence on time of the RHSs
    k3_nonadapt = h_nonadapt * rhs(r_nonadapt + 0.5 * k2_nonadapt)
    k4_nonadapt = h_nonadapt * rhs(r_nonadapt + k3_nonadapt)
    r_nonadapt += (k1_nonadapt + 2 * k2_nonadapt + 2 * k3_nonadapt + k4_nonadapt) / 6
    
nonadapt_end = time.perf_counter()

nonadapt_time = nonadapt_end - nonadapt_start


#Q1, part (b)

print("Time for each method: Adaptive=",np.round(adaptive_time,4), " seconds. Non-adaptive =",np.round(nonadapt_time,4), "seconds.")
print("The nonadaptive method time took", np.round((nonadapt_time/adaptive_time)*100,1), "% more time than the adaptive method")


# In[22]:


#Q1, part (c)
plt.figure(dpi=1200)
plt.plot(xpoints, ypoints, 'r.', label='Adaptive')  # Adaptive
plt.plot(x_nonadapt_points, y_nonadapt_points, 'k:', label='Non-adaptive')  # Non-adaptive
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.suptitle("Trajectory of a ball bearing around a space rod")
plt.title("via 4th-Order Runge-Kutta Method with adaptive/non-adaptive stepsize")
plt.axis('equal')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('lab07q1.png', dpi=150)
plt.show()


# In[ ]:




