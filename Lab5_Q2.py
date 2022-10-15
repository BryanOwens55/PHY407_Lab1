# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 13:48:41 2022

@author: bryan
"""


from scipy.io.wavfile import read, write
from numpy import empty
import numpy as np
import matplotlib.pyplot as plt


# read the data into two stereo channels
# sample is the sampling rate, data is the data in each channel,
# dimensions [2, nsamples]
sample, data = read('GraviteaTime.wav')
# sample is the sampling frequency, 44100 Hz
# separate into channels
channel_0 = data[:, 0]
channel_1 = data[:, 1]
N_Points = len(channel_0)
# ... do work on the data...

lst = np.linspace(0, N_Points/sample, len(channel_0))
plt.plot(lst,channel_0)
plt.show()
plt.plot(lst,channel_1)
plt.show()

print(sample, N_Points)
print(N_Points/sample)

'''
# this creates an empty array data_out with the same shape as "data"
# (2 x N_Points) and the same type as "data" (int16)
data_out = empty(data.shape, dtype = data.dtype)
# fill data_out
data_out[:, 0] = channel_0_out
data_out[:, 1] = channel_1_out
write('output_file.wav', sample, data_filt)
'''



plt.plot(lst,channel_0)
plt.xlim(0.02,0.05)
plt.show()

frequency1 = np.fft.rfft(channel_0)
frequency2 = np.fft.rfft(channel_1)



plt.plot(np.real(abs(frequency1)[0:int(len(frequency1)/2)]))
plt.show()

#frequency1[8800:-8799] = 0
#frequency2[8800:-8799] = 0

frequency1[881:] = 0
frequency2[881:] = 0


channel_0_out = np.fft.irfft(frequency1)
channel_1_out = np.fft.irfft(frequency2)

plt.plot(lst, channel_0_out)
plt.show()
plt.plot(lst, channel_1_out)
plt.show()

print(channel_0_out)


data_out = empty(data.shape, dtype = data.dtype)
data_out[:, 0] = channel_0_out
data_out[:, 1] = channel_1_out
write('output_file.wav', sample, data_out)


