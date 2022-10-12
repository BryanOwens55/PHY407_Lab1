# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:54:03 2022

@author: bryan
"""

from numpy import loadtxt
import numpy as np
from matplotlib.pyplot import contourf, xlabel, ylabel, title, colorbar
import matplotlib.pyplot as plt



SLP = loadtxt('SLP.txt')
Longitude = loadtxt('lon.txt')
Times = loadtxt('times.txt')
contourf(Longitude, Times, SLP)
xlabel('longitude(degrees)')
ylabel('days since Jan. 1 2015')
title('SLP anomaly (hPa)')
colorbar()
plt.show()


frequency = np.fft.rfft2(SLP)

for i in range(len(frequency)):
    if i != 3 and i != 5:
        frequency[i] = 0
plt.plot(abs(frequency))
plt.xlim(0,10)
plt.show()


z = np.fft.irfft(frequency)
z3 = z[3]
z5 = z[5]
print(z3)
contourf(Longitude, Times, z)

newSLP = np.copy(SLP)


