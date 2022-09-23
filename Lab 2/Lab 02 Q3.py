#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Authors: Dharmik Patel and Bryan Owens

This code defines a function that was derived by using the formula for blackbody radiation within Stefan's law. 
It is numerically integrated using Simpson's rule and we derive a value for the Stefan-Boltzmann constant from it. 
This value is then compared to the value within the scipy.constants module. The output is print statements for the
integral of the original function itself, the value of the constant derived from numerical integration, and the 
value of the constant as it is within scipy.constants.

'''


# In[42]:


import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import constants


# In[49]:


#Defining the function to be numerically integrated

def f(x):
    return (x**3 / (np.exp(x) - 1) )


#Setting number of intervals N, initial value a, final value b, and step-size h

N = 1000
a = 0
b = 100
h = (b-a)/N

#Defining Simpson's rule to integrate the function defined earlier

def Simpson():
    s = 0 + f(b)
    for k in range(1,N):
        if k % 2 ==0:
            s += 2*(f(a + k*h))
        else:
            s += 4*(f(a + k*h))
    return h * (s/3)

#Printing the value of the integral of function by Simpson's rule

print("The numerical integration of the function by Simpson's rule is", Simpson())

#Setting the Stefan-Boltzmann constant value from our numerical integration value

W_simpson = ((2*constants.pi*constants.k**4)/(constants.c**2 * constants.h**3))*Simpson()

#Printing the values from the numerical integration and from scipy.constants

print("The value of the Stefan-Boltzmann constant from Simpson's rule is",W_simpson)
    
print("The value of the Stefan-Boltzmann constant from scipy is", constants.Stefan_Boltzmann)

