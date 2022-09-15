#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#Create lists
t = np.arange(0,1,delta_t)
x = np.zeros(len(t)) 
y = np.zeros(len(t))
v_x = np.zeros(len(t))
v_y = np.zeros(len(t))

#set up initial conditions
x[0] = 0.47
v_y[0] = [8.17]

#Set variables
delta_t = 0.0001
G = 39.5
M = 1.651e-7

#Start for loop
for i in range(len(t)):
    r = (x[i]**2 + y[i]**2)**0.5
    a_x = -1*G*M*x[i]/(r**3)
    v_x[i+1] = a_x * delta_t
    x[i+1] = x[i] + v_x*delta_t + 0.5*a_x*(delta_t**2)
print('hello')


# In[13]:


plt.plot(x,t)
print(x)


# In[ ]:




