#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Authors: Dharmik Patel and Bryan Owens

Purpose: This code calculates the position, velocity, and acceleration of a relativistic spring-mass system
using the Euler-Cromer method, which is then used to compute the Fourier Coefficients of position in order
to create a Fourier transform plot of position as a function of angular frequency vs angular frequency. 
The time period of the system in the large and small amplitude limits from lab 03 (computed using Gaussian
quadrature) is used to calculate the angular frequencies in the large and small amplitude limits. These
values are plotted on the Fourier transform plot to see if the angular frequency corresponding to the 
peaks of the spectrum of the transform plot (which represent the characteristic frequencies) match the 
frequencies derived from Gaussian quadrature. We find that the values agree.

'''


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.constants as const
from numpy.fft import fft
from numpy.fft import rfft


# In[2]:


#part (a)

# Set variables
delta_t = 0.01   #timestep (s)
m = 1   #mass on the spring (kg)
c = const.speed_of_light   #speed of light (m/s)
k = 12   #spring constant (Newtons/metre)
xc = c * np.sqrt(m/k)   #initial position of spring such that speed of mass at equilibrium position x = 0 is c 
x01 = 1.0   #first initial position (metres)
x02 = xc   #second initial position (metres)
x03 = 10*xc   #third iniitial position (metres)


# In[3]:


# Create arrays for t, positions (x), velocities (v), acceleration (a)

t = np.arange(0,150,delta_t).  #time array from 0 to 150 seconds with interval spacing delta_t = 0.01 s
x_spring1 = x01*t 
x_spring2 = x02*t
x_spring3 = x03*t
v_spring1 = 0.0*t
v_spring2 = 0.0*t
v_spring3 = 0.0*t
a_spring1 = np.zeros(len(t)-1) 
a_spring2 = np.zeros(len(t)-1) 
a_spring3 = np.zeros(len(t)-1) 

# Set up initial values for the system in each case

x_spring1[0] = x01
x_spring2[0] = x02
x_spring3[0] = x03
v_spring1[0] = 0.0
v_spring2[0] = 0.0
v_spring3[0] = 0.0


# In[4]:


# Defining formulae to calculate the acceleration, velocity, position of the mass-spring system.

# Calculate acceleration values
def acceleration(v, x):
    # Calculate acceleration values
    a = -(k/m)*x*(1-(v**2 / c**2))**(3/2)
    return a


# Calculate velocity values
def velocity(v_old, a):
    v_new = v_old + a * delta_t
    return v_new


# Calculate position values
def position(x_old, v, a):
    x_new = x_old + v*delta_t + a*(delta_t**2)
    return x_new


# In[5]:


# Start for loop
for i in range(len(t)-1):
    
    #CALCULATIONS
    # Calculate spring acceleration values
    a_spring1[i] = acceleration(v_spring1[i], x_spring1[i])
    
    # Calculate spring velocity values                                          
    v_spring1[i+1]= velocity(v_spring1[i],a_spring1[i])
                                              
    # Calculate spring position values        
    x_spring1[i+1]= position(x_spring1[i], v_spring1[i+1], a_spring1[i])
    

for i in range(len(t)-1):
    
    #CALCULATIONS
    # Calculate spring acceleration values
    a_spring2[i] = acceleration(v_spring2[i], x_spring2[i])
    
    # Calculate spring velocity values                                          
    v_spring2[i+1]= velocity(v_spring2[i],a_spring2[i])
                                              
    # Calculate spring position values        
    x_spring2[i+1]= position(x_spring2[i], v_spring2[i+1], a_spring2[i])  
    
for i in range(len(t)-1):
    
    #CALCULATIONS
    # Calculate spring acceleration values
    a_spring3[i] = acceleration(v_spring3[i], x_spring3[i])
    
    # Calculate spring velocity values                                          
    v_spring3[i+1]= velocity(v_spring3[i],a_spring3[i])
                                              
    # Calculate spring position values        
    x_spring3[i+1]= position(x_spring3[i], v_spring3[i+1], a_spring3[i]) 


# In[9]:


# Plot position of mass on the spring (x /m) vs t (time /s)

plt.plot(t, x_spring1, label="initial x0=1.0")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.suptitle("Position vs Time")
plt.title("for a Relativistic Spring-Mass System")
plt.legend()
plt.show()

plt.plot(t, x_spring2, label="initial x0=xc")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.suptitle("Position vs Time")
plt.title("for a Relativistic Spring-Mass System")
plt.legend()
plt.show()

plt.plot(t, x_spring3, label="initial x0=10xc")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.suptitle("Position vs Time")
plt.title("for a Relativistic Spring-Mass System")
plt.legend()
plt.show()



# In[10]:


#Parts (b, c)

#Calculating Fourier Coefficients (representing amplitudes) for each position array using Fast Fourier Transform

posfft1 = fft(x_spring1)
posfft2 = fft(x_spring2)
posfft3 = fft(x_spring3)

#Creating arrays for the real part of the fourier coefficients via np.abs()

Re_posfft1 = np.array(abs(posfft1))
Re_posfft2 = np.array(abs(posfft2))
Re_posfft3 = np.array(abs(posfft3))

#Extracting the largest value amplitude from each of the arrays above.

maxvalue_amp1 = abs(np.max(Re_posfft1))
maxvalue_amp2 = abs(np.max(Re_posfft2))
maxvalue_amp3 = abs(np.max(Re_posfft3))

#Defining the scaled values of the amplitudes to be plotted

plotamp1 = np.array(Re_posfft1/maxvalue_amp1)
plotamp2 = np.array(Re_posfft2/maxvalue_amp2)
plotamp3 = np.array(Re_posfft3/maxvalue_amp3)

freq = np.fft.fftfreq(len(x_spring1), delta_t)
omega = np.array(2*np.pi*freq)

#Time periods obtained in Lab 03, Q2 from Eq(7) in the large and small amplitude limits:

largeT_lab03 = 11.6609
smallT_lab03 = 1.79193

#Large Limit/Small Limit Angular Frequency as w = 2*pi*f = 2*pi*1/T

largelimfreq_lab03 = np.full((len(x_spring1)), 2*np.pi*1/largeT_lab03)
smalllimfreq_lab03 = np.full((len(x_spring1)), 2*np.pi*1/smallT_lab03)

#Plotting Fourier Transform Plot
 
plt.figure(dpi=1200)
plt.plot(omega, plotamp1, label="x0 = 1.0 m")
plt.plot(omega, plotamp2, label="x0 = xc")
plt.plot(omega, plotamp3, label="x0 = 10xc")
plt.plot(largelimfreq_lab03, omega, label="large amplitude limit frequency")
plt.plot(smalllimfreq_lab03, omega, label="small amplitude limit frequency")
plt.xlabel("Angular Frequency (rad/s)")
plt.ylabel("Scaled Fourier Component of Position")
plt.ylim(ymin = 0, ymax = 1.25)
plt.xlim(xmin = -10, xmax = 10)
plt.suptitle("Fourier Component of Position as a function of Angular Frequency")
plt.title("for a Relativistic Spring with m = 1 kg, k = 12")
plt.grid()
plt.legend(prop={'size': 6})
plt.savefig("FourierComponents.png")
plt.show()


# In[ ]:




