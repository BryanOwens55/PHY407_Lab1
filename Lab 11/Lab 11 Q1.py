#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''

Authors: Dharmik Patel and Bryan Owens.
Purpose: To find different solutions to the travelling salesman problem via simulated annealing and explore the 
consequences due to variations in time constant and initial seed value, and to find the global minima of two
different functions via the same method. 

'''



#Import necessary packages and modules

import numpy as np
import matplotlib.pyplot as plt
from random import random,randrange
from random import seed
from numpy import empty

seed(10) #Initialise random number generator with seed value 10.

N = 25  #number of city locations
Tmax = 10.0 #Maximal temperature acceptance
Tmin = 1e-3 #Minimal temperature acceptance
tau = 1e4 #Annealing time constant

#Function to calculate the magnitude of a vector 
def mag(x): 
    return np.sqrt(x[0]**2+x[1]**2)

#Function to calculate the total distance that the trip traverses
def distance() : 
    s = 0.0 
    for i in range(N) : 
        s += mag(r[i+1] - r[i])
    return s 

#Choosing N city locations and calculating the initial distance taken on a satisfactory trip
r = empty([N+1,2] ,float) 
for i in range(N): 
    r[i,0] = random() 
    r[i,1] = random()
r[N] = r[0]
D = distance() 

# Loop for simulated annealing
t = 0 
T = Tmax
n = 0.01
seed(n) # varying seed number for Monte Carlo
while T>Tmin:
    t += 1 
    T = Tmax*np.exp(-t/tau) # Second cooling iteration
    # Choosing two cities to swap and ensuring uniqueness
    i,j = randrange(1,N),randrange(1,N) 
    while i==j: 
        i,j = randrange(1,N),randrange(1,N) 
    # Swapping and calculating change in distance traversed
    oldD = D 
    r[i,0],r[j,0] = r[j,0],r[i,0]
    r[i,1],r[j,1] = r[j,1] ,r[i,1]
    D = distance()
    deltaD = D - oldD

    # If the move is rejected in the Monte Carlo process, swapping locations back to original states
    if random()>=np.exp(-deltaD/T): 
        r[i,0] ,r[j,0] = r[j,0] ,r[i,0] 
        r[i,1],r[j,1] = r[j,1],r[i,1] 
        D = oldD 

#Plotting 

txt = "The seed number is", n, "with D = ", round(D,4), "and cooling time = ", tau

plt.plot(r[:,0],r[:,1])
plt.ylim(0,1.2)
plt.xlim(0,1.4)
plt.title('Travelling Salesman Solution')
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.show()


# In[2]:


#Question 1, Part (b)
n = 5
seed(n) #Changing the seed number

#Defining the function
def func1(x,y):
    return x**2 - np.cos(4*np.pi*x) + (y-1)**2

#Setting initial conditions and creating initial arrays to be filled
x = 2
xvals = []
xvals.append(x)
y = 2
yvals = []
yvals.append(y)

#Defining cooling parameters
Tmax = 10.0 
Tmin = 1e-3
tau = 1e4

#Loop for the simulated annealing method
t = 0 
T = Tmax
while T>Tmin:
    rand_gaussx = np.random.normal(loc=0.0, scale=1.0)
    rand_gaussy = np.random.normal(loc=0.0, scale=1.0)

    t += 0.1 
    T = Tmax*np.exp(-t/tau) #Second cooling

    #Defining the value of the function at the initial guess
    current = func1(x,y)

    #Given the gaussian random variable, determining the next values of x and y
    x_best = x + rand_gaussx*x
    y_best = y + rand_gaussy*y

    #Evaluting the function at these new determined values of x and y
    candidate = func1(x_best,y_best)

    #If our new value of the function is smaller than the old one, we keep it
    if candidate < current:
        xvals.append(x_best)
        yvals.append(y_best)
        x = x_best
        y = y_best

xvals = np.array(xvals)
yvals = np.array(yvals)
print('final positions (x,y):',xvals[-1],yvals[-1])

#Plotting 

fig,(ax0,ax1) = plt.subplots(figsize=(12,5),ncols=2,nrows=1)
ax0.plot(xvals,label='x value',c='r')
ax0.plot(yvals,label='y value',c='b',alpha=0.6)
ax0.set_title(r'Convergence of $x$ and $y$ during cooling')
ax0.grid()
ax0.legend()
ax1.scatter(xvals,yvals,s=5)
ax1.set_title(r'Variation of $x$ and $y$ points for each step')
ax1.grid()
plt.savefig('q1b.png')

#Question 1, Part (c)

#Defining the function
def func2(x,y):
    return np.cos(x) + np.cos(np.sqrt(2)*x) + np.cos(np.sqrt(3)*x) + (y-1)**2

#Setting initial conditions and creating initial arrays to be filled
x2 = 10
xvals2 = []
xvals2.append(x2)
y2 = 5
yvals2 = []
yvals2.append(y2)

Tmax = 10.0 
Tmin = 1e-3
tau = 1e3

#Loop for the simulated annealing method 
t = 0 
T = Tmax
while T>Tmin:
    rand_gaussx2 = np.random.normal(loc=0.0, scale=1.0)
    rand_gaussy2 = np.random.normal(loc=0.0, scale=1.0)

    t += 0.1 
    T = Tmax*np.exp(-t/tau) #Second cooling

    #Defining the value of the function at the initial guess
    current2 = func2(x2,y2)

    #Given the gaussian random variable, determining the next values of x and y
    x_best2 = x2 + rand_gaussx2*x2
    y_best2 = y2 + rand_gaussy2*y2

    #Evaluting the function at these new determined values of x and y
    candidate2 = func2(x_best2,y_best2)

    #If our new value of the function is smaller than the old one, we keep it
    if candidate2 < current2 and 0<x_best2<50 and -20<y_best2<20:
        xvals2.append(x_best2)
        yvals2.append(y_best2)
        x2 = x_best2
        y2 = y_best2

xvals2 = np.array(xvals2)
yvals2 = np.array(yvals2)
print('x,y arrays: ',xvals2,yvals2)
print('final positions (x,y):',xvals2[-1],yvals2[-1])

#Plotting

fig,(ax0,ax1) = plt.subplots(figsize=(12,5),ncols=2,nrows=1)
ax0.plot(xvals2,label='x value',c='b')
ax0.plot(yvals2,label='y value',c='r',alpha=0.6)
ax0.set_title('Convergence of $x$ and $y$ during cooling')
ax0.legend()
ax0.grid()
ax1.scatter(xvals2,yvals2,s=5)
ax1.set_title('Variation of $x$ and $y$ points for each step')
ax1.grid()
plt.savefig('q1c.png')
plt.show()


# In[ ]:




