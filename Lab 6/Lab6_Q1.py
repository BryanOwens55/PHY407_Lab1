# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:58:17 2022

@author: bryan
"""
import numpy as np
import matplotlib as plt
from math import sin, cos, asin
from numpy import arange 
from pylab import plot,xlabel,ylabel,show

eps = 1
alpha = 1
m = 1


def f(r,t): 
    x = r[0]
    y = r[1] 
    f = 4*eps*(6*(alpha**6/r**7) - 12*(alpha**12/r**13)) / m
    print(f)
    return np.array([f[0],f[1]] ,float) 

a = 1
b = 100
N = 10
h = (b-a)/N 
tpoints = arange(a, b,h) 
xpoints = []
ypoints  = []
r = np.array([4.0,4.0] ,float) 

for t in tpoints: 
    xpoints.append(r[0]) 
    ypoints.append(r[1]) 
    k1 = h*f(r,t) 
    k2 = h*f(r+0.5*k1,t+0.5*h) 
    k3 = h*f(r+0.5*k2,t+0.5*h) 
    k4 = h*f(r+k3,t+h) 
    r += (k1+2*k2+2*k3+k4)/6 



plot(xpoints, ypoints, '.') 
show() 

plot(tpoints, ypoints, '.') 
show() 



