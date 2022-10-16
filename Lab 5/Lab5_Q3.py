# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:54:03 2022

@author: bryan
"""

# Import numpy and matplotlib
from numpy import loadtxt
import numpy as np
from matplotlib.pyplot import contourf, xlabel, ylabel, title, colorbar
import matplotlib.pyplot as plt


# import the files used in this question
SLP = loadtxt('SLP.txt')
Longitude = loadtxt('lon.txt')
Times = loadtxt('times.txt')

# Plot the imported files
contourf(Longitude, Times, SLP)
xlabel('longitude(degrees)')
ylabel('days since Jan. 1 2015')
title('SLP anomaly (hPa)')
colorbar()
plt.show()


# Take the fourier transform
frequency = np.fft.rfft2(SLP)

'''
for i in range(len(frequency)):
    if i != 3 and i != 5:
        frequency[i] = 0
#plt.plot(abs(frequency))
#plt.xlim(0,10)
#plt.show()
'''



# Create a 2D list with only the frequencies associated with m=3
frequency3 = frequency * 0
frequency3[:,3] = frequency[:,3]


# Create a 2D list with only the frequencies associated with m=3
frequency5 = frequency * 0
frequency5[:,5] = frequency[:,5]

# Reverse the fourier transform
z = np.fft.irfft2(frequency)
z3 = np.fft.irfft2(frequency3)
z5 = np.fft.irfft2(frequency5)

# Plot the contour of the two sets of frequencies associated with
# m=3 and m=5 respcetively
contourf(Longitude, Times, z3)
plt.show()
contourf(Longitude, Times, z5)


