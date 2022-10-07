#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Authors: Dharmik Patel and Bryan Owens
Purpose: To explore different methods of solving nonlinear equations; namely, using relaxation, overrelaxation, and 
binary search/bisection. 
The outputs are print output statements of the solutions, number of iterations taken to reach the solution, and plots of iteration vs solution
and value of parameter c vs solution where c is defined in the nonlinear equation.

'''


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import constants


# In[2]:


# Lab 04 Q3 Part (a)
# Exercise 6.10 part (a)

# Defining the nonlinear equation
def f(x,c):
    return 1 - np.exp(-c*x)

# Defining initial conditions and array for the solutions (x's)
x = 0.1  # initial x guess
dx = 1  # initial distance (just needs to be big)
threshold = 1e-6  # convergence threshold
x_list = [x]  # will fill up with successive x's

# Executing the relaxation method 

while dx > threshold:
    x_list.append(f(x_list[-1],2))
    dx = np.abs(x_list[-1]-x_list[-2])
    
print("The solution to the equation for c=2 is",np.round(x_list[-1],6))


# In[3]:


#Plotting the iteration number vs the solution to the equation reached at the iteration

def print_sol():
    plt.figure(dpi=150)
    plt.plot(x_list)
    plt.xlabel("Iteration number")
    plt.ylabel("Solution to $1-\exp(-cx)$")
    plt.grid()
    
print_sol()


# In[4]:


#Exercise 6.10 part (b)


#Defining the range of c values

c = np.arange(0,3.01,0.01)

#Defining the function that gives the solutions

def nonlin(c):
    x = 0.1  # initial x guess
    dx = 1  # initial distance (just needs to be big)
    threshold = 1e-10  # convergence threshold
    x_list = [x]  # will fill up with successive x's
 
    while dx > threshold:
        x_list.append(f(x_list[-1],c))
        dx = np.abs(x_list[-1]-x_list[-2])
    return (np.round(x_list[-1],10))
    
soln = c*0

#Indexing solutions as given by the function defined above
for i in range(len(c)):
    soln[i] = nonlin(c[i])
    

#Plotting
plt.plot(c, soln)
plt.xlabel("values of c")
plt.ylabel("solution x for c")
plt.title("Parameter c vs Solution x for $1-\exp(-cx)$")
plt.grid()


# In[5]:


#Lab 04 Q3 part (b)

#Exercise 6.11 Part (b)

#Defining the function

def f(x,c):
    return 1 - np.exp(-c*x)

x = 0.1  # initial x guess
dx = 1  # initial distance (just needs to be big)
threshold = 1e-6  # convergence threshold
x_list = [x]  # will fill up with successive x's

#Defining initial value of iteration count
itercount = 0

#Defining the while loop and iteration counter

while dx > threshold:
    itercount += 1
    x_list.append(f(x_list[-1],2))
    dx = np.abs(x_list[-1]-x_list[-2])
print("The solution to the equation for c=2 is", np.round(x_list[-1],6))
print("The number of iterations required to reach a solution accurate to 10^-6 is", itercount)


# In[6]:


#Exercise 6.11 Part (c)


#Defining function to be solved by overrelaxation, with weight w

def f(x,c,w):
    return (1+w)*(1-np.exp(-2*x)) - w*x

#Finding the solution using overrelaxation

w = 0.7
x = 0.1  # initial x guess
dx = 1  # initial distance (just needs to be big)
threshold = 1e-6  # convergence threshold
x_list = [x]  # will fill up with successive x's
 
while dx > threshold:
    x_list.append(f(x_list[-1],2,w))
    dx = np.abs(x_list[-1]-x_list[-2])
    print("The solution at the iteration is",np.round(x_list[-1], 6))
print("The solution to the equation for c=2 is",np.round(x_list[-1],6))


#finding number of iterations

x = 0.1  # initial x guess
dx = 1  # initial distance (just needs to be big)
threshold = 1e-6  # convergence threshold
x_list = [x]  # will fill up with successive x's

#Defining the iteration count initial value
count = 0

while dx > threshold:
    count += 1
    x_list.append(f(x_list[-1],2,w))
    dx = np.abs(x_list[-1]-x_list[-2])
print("The number of iterations required to reach a solution accurate to 10^-6 is", count)


# In[7]:


#Exercise 6.11 Part (d)

'''

For functions like x = 1-exp((1-x^2)), the choice of a negative weight will make the function converge.
If implemented for a function like the one in part (c), the divergence makes the method of relaxation (over-
or otherwise) futile.

'''


# In[9]:


#Lab 04 Q3 Part (c)

#Exercise 6.13 part (b)

#Defining the binary search method for the nonlinear function

def binary(f, x1, x2, accuracy,return_x_list=False):
    f_x1 = f(x1)
    if f_x1*f(x2) > 0:
        print ("The function does not have opposite signs at the endpoints of the interval.")
    xM = float(x1 + x2)/2.0
    f_M = f(xM)
    iteration_counter = 1
    if return_x_list:
        x_list = []
    while abs(f_M) > accuracy:
        if f_x1*f_M > 0:   # i.e. same sign
            x1 = xM
            f_x1 = f_M
        else:
            x2 = xM
        xM = float(x1 + x2)/2
        f_M = f(xM)
        iteration_counter += 1
        if return_x_list:
            x_list.append(xM)
    if return_x_list:
        return x_list, iteration_counter
    else:
        return xM, iteration_counter
    
#Defining the nonlinear function

def wienfunction(x):
    return 5*np.exp(-x) + x - 5

#Defining the left endpoint and the right endpoint
a = 0.1
b = 10

solution, number_iterations = binary(wienfunction, a, b, 1e-6)

print ("Number of iterations (function calls)=",(1 + 2*number_iterations))
print ("Our solution is x =",np.round((solution),6))

#b = Wien displacement constant

b = (constants.h*constants.c)/(constants.k*solution)

print("The value of Wien's displacement constant is found to be", np.round(b,6), "meters-Kelvin")


# In[10]:


#Exercise 6.13 part (c)

wavelengthpeak_Sun = 502e-9

surfacetemp_Sun = b/wavelengthpeak_Sun

print("The surface temperature of the Sun by the Wien displacement law is", np.round(surfacetemp_Sun,6), "Kelvin")


# In[ ]:




