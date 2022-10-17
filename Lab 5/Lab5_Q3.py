"""
Authors: Bryan Owens, Dharmik Patel
Purpose: To Fourier decompose a 2D list into two Fourier wavenumber 
m = 3 and to m = 5
Collaberation: Code was evenly created and edited by both lab partners
"""

# Import numpy and matplotlib
from numpy import loadtxt
import numpy as np
from matplotlib.pyplot import contourf, xlabel, ylabel, title, colorbar
import matplotlib.pyplot as plt


# Import SLP, longitude, and times file
SLP = loadtxt('SLP.txt')
Longitude = loadtxt('lon.txt')
Times = loadtxt('times.txt')

# Plot the contour plot of SLP
contourf(Longitude, Times, SLP)
xlabel('longitude(degrees)')
ylabel('days since Jan. 1 2015')
title('SLP anomaly (hPa)')
colorbar()
plt.show()


# Take the Fourier transform of SLP
frequency = np.fft.rfft2(SLP)

# Isolate frequencies associated with m=3
frequency3 = frequency * 0
frequency3[:,3] = frequency[:,3]


# Isolate frequencies associated with m=5
frequency5 = frequency * 0
frequency5[:,5] = frequency[:,5]

# Inverse the Fourier transform
z3 = np.fft.irfft2(frequency3)
z5 = np.fft.irfft2(frequency5)

# Plot the contour of the frequencies associated
# with m=3
contourf(Longitude, Times, z3)
xlabel('longitude(degrees)')
ylabel('days since Jan. 1 2015')
title('SLP anomaly (hPa), for m=3')
plt.show()

# Plot the contour of the frequencies associated
# with m=5
contourf(Longitude, Times, z5)
xlabel('longitude(degrees)')
ylabel('days since Jan. 1 2015')
title('SLP anomaly (hPa), for m=5')


