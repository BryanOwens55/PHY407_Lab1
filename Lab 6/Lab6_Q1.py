# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:58:17 2022

@author: bryan
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, asin
from numpy import arange 

eps = 1
alpha = 1
m = 1


def f(r,t): 
    x = r[0]
    y = r[1] 
    fx = x*24*eps*((-1*alpha**6/(x**2+y**2)**7) + 2*(alpha**12/(x**2+y**2)**13)) / (m)
    fy = y*24*eps*((-1*alpha**6/(x**2+y**2)**7) + 2*(alpha**12/(x**2+y**2)**13)) / (m)
    print(fx)
    return np.array([fx,fy] ,float) 



dt = 0.01
time = np.arange(0,1000*dt,dt)

def Molecules(initial_1, initial_2):

    particle1 = ([],[])
    particle2 = ([],[])
    
    r1 = np.array([initial_1[0],initial_1[1]] ,float) 
    r2 = np.array([initial_2[0],initial_2[1]] ,float) 
    v = np.array([0,0] ,float) + (dt/2) * f(r1-r2,0) 
    
    for t in time:
        particle1[0].append(r1[0])
        particle1[1].append(r1[1])
        
        particle2[0].append(r2[0])
        particle2[1].append(r2[1])
        
        r1 += dt*v
        r2 -= dt*v
        k = dt*f(r1-r2,t)
        v += k
    return particle1, particle2
    

particle1, particle2 = Molecules([4,4], [5.2,4])
plt.plot(particle1[0], particle1[1], '.') 
plt.plot(particle2[0], particle2[1], '.')
plt.show() 




particle1, particle2 = Molecules([4.5,4],[5.2,4])
plt.plot(particle1[0], particle1[1], '.') 
plt.plot(particle2[0], particle2[1], '.')
plt.show() 




particle1, particle2 = Molecules([2,3],[3.5,4.4])
plt.plot(particle1[0], particle1[1], '.') 
plt.plot(particle2[0], particle2[1], '.')
plt.show() 



