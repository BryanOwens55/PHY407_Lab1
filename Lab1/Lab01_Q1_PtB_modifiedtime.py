#!/usr/bin/env python
# coding: utf-8

'''

This code implements an Euler-Cromer method to simulate the orbit of the Earth about the Sun in a system including Jupiter 
whose mass has been amended to 1 Solar Mass. The duration of the simulation is 5 Earth years (as opposed to 3 Earth years for the main code). The output of the code is plots of
x-position vs time, y-position vs time, and x-position vs y-position (phase) for Earth and Jupiter respectively. This code was written by
Dharmik Patel and Bryan Owens together. This code is identical to that for Part B except for the sole modification of orbit simulation duration. 

'''

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy as sc


# In[2]:


# Set variables
delta_t = 0.0001 #yrs
G = 39.5 #AU^3 M^−1 yr^−2
M_sun = 1 # Solar Masses
M_jupiter = M_sun


# In[3]:


# Create parameters

# Creating the time array from 0 yrs to 3 yrs with difference delta_t = 0.0001 yr
t = np.arange(0,5,delta_t)

#Creating x,y-position variables for Earth
x_earth = 1.0*t 
y_earth = 0.0*t

#Creating x,y-position variables for Jupiter
x_jupiter = 5.2*t
y_jupiter = 0.0*t

#Creating x,y-velocity variables for Earth
v_x_earth = 0.0*t
v_y_earth = 6.18*t

#Creating x,y-velocity variables for Jupiter
v_x_jupiter = 0.0*t
v_y_jupiter = 2.63*t


#Creating numpy arrays initially full of 0's for acceleration: 
a_x_earth = np.zeros(len(t)-1)   #x-acceleration of Earth relative to the Sun
a_x_earth_jup = np.zeros(len(t)-1)   #x-acceleration of Earth relative to Jupiter
a_y_earth = np.zeros(len(t)-1)   #y-acceleration of Earth relative to the Sun
a_y_earth_jup = np.zeros(len(t)-1)   #y-acceleration of Earth relative to Jupiter
a_x_jupiter = np.zeros(len(t)-1)   #x-acceleration of Jupiter relative to the Sun
a_y_jupiter = np.zeros(len(t)-1)   #y-acceleration of Jupiter relative to the Sun

# Set up initial conditions for each position and velocity variable for Earth and Jupiter
x_earth[0] = 1.0
y_earth[0] = 0.0
x_jupiter[0] = 5.2
y_jupiter[0] = 0.0
v_x_earth[0] = 0.0
v_y_earth[0] = 6.18
v_x_jupiter[0] = 0.0
v_y_jupiter[0] = 2.63


# In[4]:


# Calculate acceleration values
def acceleration(x, y, Mass):
    # Calculate radial value
    r = (x**2 + y**2)**0.5
    # Calculate acceleration values
    a_x = -1*G*Mass*x/(r**3)
    a_y = -1*G*Mass*y/(r**3)
    return a_x, a_y


# In[5]:


# Calculate velocity values
def velocity(v_x_old, v_y_old, a_x, a_y):
    v_x_new = v_x_old + a_x * delta_t
    v_y_new = v_y_old + a_y * delta_t
    return v_x_new, v_y_new


# In[6]:


# Calculate position values
def position(x_old, y_old, v_x, v_y, a_x, a_y):
    x_new = x_old + v_x*delta_t + a_x*(delta_t**2)
    y_new = y_old + v_y*delta_t + a_y*(delta_t**2)
    return x_new, y_new


# In[7]:


# Start for loop
for i in range(len(t)-1):
    
    #JUPITER PARAMETER CALCULATIONS
    
    # Calculate Jupiter acceleration values
    a_x_jupiter[i], a_y_jupiter[i] = acceleration(x_jupiter[i],y_jupiter[i], M_sun)
    
    # Calculate Jupiter velocity values                                          
    v_x_jupiter[i+1], v_y_jupiter[i+1] = velocity(v_x_jupiter[i], v_y_jupiter[i], a_x_jupiter[i], a_y_jupiter[i])
                                              
    # Calculate Jupiter position values        
    x_jupiter[i+1], y_jupiter[i+1] = position(x_jupiter[i], y_jupiter[i], v_x_jupiter[i+1], v_y_jupiter[i+1], a_x_jupiter[i], a_y_jupiter[i])                                          
    
    
    #EARTH PARAMETER CALCULATIONS
    
    # Calculate Earth acceleration values
    a_x_earth[i], a_y_earth[i] = acceleration(x_earth[i],y_earth[i], M_sun)
    a_x_earth_jup[i], a_y_earth_jup[i] = acceleration(x_earth[i]-x_jupiter[i],y_earth[i]-y_jupiter[i], M_jupiter)
    a_x_earth[i] = a_x_earth[i] + a_x_earth_jup[i]
    a_y_earth[i] = a_y_earth[i] + a_y_earth_jup[i]
    
    # Calculate Earth velocity values
    v_x_earth[i+1], v_y_earth[i+1] = velocity(v_x_earth[i], v_y_earth[i], a_x_earth[i], a_y_earth[i])
     
    # Calculate Earth position values    
    x_earth[i+1], y_earth[i+1] = position(x_earth[i], y_earth[i], v_x_earth[i+1], v_y_earth[i+1], a_x_earth[i], a_y_earth[i])


# In[8]:


# Plots for phase, x-position vs t, y-position vs t

plt.plot(t, x_earth)
plt.xlabel("Time (yr)")
plt.ylabel("x-position of Earth (Au)")
plt.title("X-position of Earth vs. Time")
plt.savefig("xvst_earth.png")
plt.show()


plt.plot(t, x_jupiter)
plt.xlabel("Time (yr)")
plt.ylabel("x-position of Jupiter (Au)")
plt.title("X-position of Jupiter vs. Time")
plt.savefig("xvst_jupiter.png")
plt.show()


plt.plot(t, y_earth)
plt.xlabel("Time (yr)")
plt.ylabel("y-position of Earth (AU)")
plt.title("Y-position of Earth vs. Time")
plt.savefig("yvst_earth.png")
plt.show()


plt.plot(t, y_jupiter)
plt.xlabel("Time (yr)")
plt.ylabel("y-position of Jupiter (AU)")
plt.title("Y-position of Jupiter vs. Time")
plt.savefig("yvst_jupiter.png")
plt.show()
plt.show()


plt.plot(x_earth, y_earth)
plt.xlabel("x-position of Earth (AU)")
plt.ylabel("y-position of Earth (AU)")
plt.title("Phase plot of Earth's orbit about the Sun for 5 yr period")
plt.savefig("phase_earth.png")
plt.show()


plt.plot(x_jupiter, y_jupiter)
plt.xlabel("x-position of Jupiter (AU)")
plt.ylabel("y-position of Jupiter (AU)")
plt.title("Phase plot of Jupiter's orbit about the Sun for 5 yr period")
plt.savefig("phase_jupiter.png")
plt.show()


# In[ ]:




