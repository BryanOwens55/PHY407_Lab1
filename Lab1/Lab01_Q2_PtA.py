#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy as sc


# In[2]:


# Set variables
delta_t = 0.0001 #0.0001
G = 39.5
M_sun = 1
M_jupiter = M_sun/1000


# In[3]:


# Create parameters

t = np.arange(0,10,delta_t)

x_earth = 1.0*t 
y_earth = 0.0*t

x_jupiter = 5.2*t
y_jupiter = 0.0*t

v_x_earth = 0.0*t
v_y_earth = 6.18*t

v_x_jupiter = 0.0*t
v_y_jupiter = 2.63*t

a_x_earth = np.zeros(len(t)-1)
a_x_earth_jup = np.zeros(len(t)-1)
a_y_earth = np.zeros(len(t)-1)
a_y_earth_jup = np.zeros(len(t)-1)

a_x_jupiter = np.zeros(len(t)-1)
a_y_jupiter = np.zeros(len(t)-1) 

# Set up initial conditions

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
    a_x_earth_jup[i], a_y_earth_jup[i] = acceleration(x_earth[i] - x_jupiter[i],y_earth[i] - y_jupiter[i], M_jupiter)
    a_x_earth[i] = a_x_earth[i] + a_x_earth_jup[i]
    a_y_earth[i] = a_y_earth[i] + a_y_earth_jup[i]
    
    # Calculate Earth velocity values
    v_x_earth[i+1], v_y_earth[i+1] = velocity(v_x_earth[i], v_y_earth[i], a_x_earth[i], a_y_earth[i])
     
    # Calculate Earth position values    
    x_earth[i+1], y_earth[i+1] = position(x_earth[i], y_earth[i], v_x_earth[i+1], v_y_earth[i+1], a_x_earth[i], a_y_earth[i])
    


# In[8]:


# Plot x vs y, x vs t, y vs t
#MAKE FEATURES AXES ETC
plt.plot(t, x_earth)
plt.show()
plt.plot(t, x_jupiter)
plt.show()
plt.plot(t, y_earth)
plt.show()
plt.plot(t, y_jupiter)
plt.show()
plt.plot(x_earth, y_earth)
plt.show()
plt.plot(x_jupiter, y_jupiter)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




