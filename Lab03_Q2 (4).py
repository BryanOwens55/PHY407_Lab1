#!/usr/bin/env python
# coding: utf-8

# In[27]:


'''
This code is for Q2 of Lab 03, PHY407.
Authors: Dharmik Patel and Bryan Owens


'''


# In[28]:


import numpy as np
import scipy as sc
from gaussxw import gaussxw
import gaussint
import scipy.constants as const
import matplotlib.pyplot as plt


# In[29]:


#PART A



#We first define a function for the inverse of the velocity of a relativistic spring-particle system

def g(x):
    return 4/(c*np.sqrt((k*(x0**2 - x**2)*(2*m*c**2 + (k*(x0**2 - x**2))/2))/(2*(m*c**2 + k*(x0**2 - x**2)/2)**2)))


#Defining constants and parameters

c = const.speed_of_light
x0 = 0.01 #metres
k = 12 #N/kg
m = 1 #kg

N = 8
a = 0.0
b = x0

#Defining the quadrature

x8,w8 = gaussxw(N)
xp8 = 0.5*(b-a)*x8 + 0.5*(b+a)
wp8 = 0.5*(b-a)*w8
# Perform the integration
s8 = 0.0
for i in range(N):
    s8 += wp8[i]*g(xp8[i])
    

QuadratureValue_8 = s8
ClassicalValue = 2*np.pi*np.sqrt(m/k)

    
print("The gaussian quadrature method for N=8 points fitted gives",np.round(s8, 4), "seconds.")
print("The classical limit formula 2pisqrt(m/k) gives", np.round(ClassicalValue,4), "seconds.")





# In[35]:


c = const.speed_of_light
x0 = 0.01 #metres
k = 12 #N/kg
m = 1 #kg

N = 16
a = 0.0
b = x0


x16,w16 = gaussxw(N)
xp16 = 0.5*(b-a)*x16 + 0.5*(b+a)
wp16 = 0.5*(b-a)*w16
# Perform the integration
s16 = 0.0
for i in range(N):
    s16 += wp16[i]*g(xp16[i])
    

QuadratureValue_16 = s16
QuadratureError_8N = QuadratureValue_16 - QuadratureValue_8

print("The gaussian quadrature method for N=16 points fitted gives",np.round(s16,4), "seconds.")
print("The quadrature error for N=8 is",np.round(QuadratureError_8N,4), "seconds.")


# In[31]:


c = const.speed_of_light
x0 = 0.01 #metres
k = 12 #N/kg
m = 1 #kg

N = 32
a = 0.0
b = x0


x32,w32 = gaussxw(N)
xp32 = 0.5*(b-a)*x32 + 0.5*(b+a)
wp32 = 0.5*(b-a)*w32
# Perform the integration
s32 = 0.0
for i in range(N):
    s32 += wp32[i]*g(xp32[i])
    
QuadratureValue_32 = s32
QuadratureError_16N = QuadratureValue_32 - QuadratureValue_16

print("The gaussian quadrature method for N=32 points fitted gives",s32)
print("The quadrature error for N = 16 is",np.round(QuadratureError_16N,4))


# In[32]:


print("The fractional error for N=16 is", np.round(QuadratureError_16N/QuadratureValue_16, 4))
print("The fractional error for N=8 is", np.round(QuadratureError_8N/QuadratureValue_8, 4))


# In[54]:


#PART B

#Plotting the weighted values and integrands

x_points8 = np.array(x8*0.001)
y_points8 = np.array(g(x_points8))
weights8 = np.array(w8)

x_points16 = np.array(x16*0.001)
y_points16 = np.array(g(x_points16))
weights16 = np.array(w16)


plt.plot(x_points8*1000, y_points8, label="integrands N = 8")
plt.plot(x_points16*1000, y_points16, label="integrands N = 16")
plt.xlabel("x points calculated by gaussxw (10^-3)")
plt.ylabel("values of the integrand")
plt.legend()
plt.savefig("integrandplot.png")
plt.show()


plt.plot(x_points8*1000, y_points8*w8, label="weights N = 8")
plt.plot(x_points16*1000, y_points16*w16, label="weights N = 16")
plt.xlabel("x points calculated by gaussxw (10^-3)")
plt.ylabel("values of the integrand multiplied by the weights")
plt.legend()
plt.savefig("integrandweightplot.png")
plt.show()


# In[55]:


#part D

c = const.speed_of_light
x0 = 0.01 #metres
k = 12 #N/kg
m = 1 #kg

N = 200
a = 0.0
b = x0

#Defining the quadrature

x200,w200 = gaussxw(N)
xp200 = 0.5*(b-a)*x200 + 0.5*(b+a)
wp200 = 0.5*(b-a)*w200
# Perform the integration
s200 = 0.0
for i in range(N):
    s200 += wp200[i]*g(xp200[i])
    

QuadratureValue_200 = s200
ClassicalValue = 2*np.pi*np.sqrt(m/k)
PercentageError = abs((QuadratureValue_200 - ClassicalValue)/ClassicalValue)*100

print("The gaussian quadrature method for N=200 points fitted gives",np.round(QuadratureValue_200,4))
print("The classical (small amplitude) limit formula 2pisqrt(m/k) gives", np.round(ClassicalValue,4))
print("The percentage error in the quadrature value relative to the classical value is", np.round(PercentageError,4),"%")





# In[78]:


#part E

c = const.speed_of_light
k = 12 #N/kg
m = 1 #kg
x_c = c * np.sqrt(m/k)
x0 = 10*x_c #metres

N = 200
a = 1
b = x0

#Defining the quadrature

xc,wc = gaussxw(N)
xpc = 0.5*(b-a)*xc + 0.5*(b+a)
wpc = 0.5*(b-a)*wc
# Perform the integration
sc = 0.0
for i in range(N):
    sc += wpc[i]*g(xpc[i])

QuadratureValue_x_c = sc
LargeAmplitudeLimitValue = 4*x0 / c

print("The quadrature value for the time period of a relativistic particle on a spring as a function of x0, 1 m<x0<10x_c where x_c = the initial displacement of the particle such that v = c at the equilibrium length, is", np.round(QuadratureValue_x_c ,4), "seconds")

print("The classical limit formula 2pisqrt(m/k) gives", np.round(ClassicalValue,4), "seconds")

print("The large limit formula 4x0/c gives", np.round(LargeAmplitudeLimitValue,4), "seconds")

xpoints_c = np.array(xc)
ypoints_c = np.array(g(xpoints_c))

plt.plot(xpoints_c,ypoints_c)
plt.xlabel("x points calculated by gaussxw in the range 1 m <x<10x_c (metres)")
plt.ylabel("Time Period (s)")
plt.savefig("timeperiodxc.png")
plt.show()

print(x_c)


# In[ ]:




