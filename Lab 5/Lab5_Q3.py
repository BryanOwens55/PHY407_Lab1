"""
Authors: Bryan Owens, Dharmik Patel
Purpose: To Fourier decompose a 2D list into two Fourier wavenumber 
m = 3 and to m = 5
Collaberation: Code was evenly created and edited by both lab partners
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

'''
for i in range(len(frequency)):
    if i != 3 and i != 5:
        frequency[i] = 0
#plt.plot(abs(frequency))
#plt.xlim(0,10)
#plt.show()
'''




frequency3 = frequency * 0
frequency3[:,3] = frequency[:,3]

print(frequency, frequency3[0])

frequency5 = frequency * 0
frequency5[:,5] = frequency[:,5]

z = np.fft.irfft2(frequency)

z3 = np.fft.irfft2(frequency3)
z5 = np.fft.irfft2(frequency5)

contourf(Longitude, Times, z3)
xlabel('longitude(degrees)')
ylabel('days since Jan. 1 2015')
title('SLP anomaly (hPa), for m=3')

plt.show()
contourf(Longitude, Times, z5)
xlabel('longitude(degrees)')
ylabel('days since Jan. 1 2015')
title('SLP anomaly (hPa), for m=5')
newSLP = np.copy(SLP)

