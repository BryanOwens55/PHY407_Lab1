# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 13:48:41 2022

@author: bryan
"""

# import scipy, numpy, and matplotlib
from scipy.io.wavfile import read, write
from numpy import empty
import numpy as np
import matplotlib.pyplot as plt


# Import then read the data into two stereo channels
# sample is the sampling rate, data is the data in each channel,
# dimensions [2, nsamples]
sample, data = read('GraviteaTime.wav')
# sample is the sampling frequency, 44100 Hz
# separate into channels
channel_0 = data[:, 0]
channel_1 = data[:, 1]
N_Points = len(channel_0)


# Plot the two channels of the audio file
lst = np.linspace(0, N_Points/sample, len(channel_0))


fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
fig.subplots_adjust(hspace=0.4)
ax1.plot(lst, channel_0)
ax2.plot(lst, channel_1)
ax1.set_ylabel('Frequency (Hz)')
ax1.set_title('Frequency vs time, channel 0')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_title('Frequency vs time, channel 1')
plt.show()


# Calculate the timestep by the given sample rate
dt = 1 / sample
time = np.arange(0, N_Points * dt, dt)



# Plot the channels with bounds of 0.02s to 0.05s
fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
fig.subplots_adjust(hspace=0.4)
ax1.plot(lst, channel_0)
ax2.plot(lst, channel_1)
ax1.set_ylabel('Frequency (Hz)')
ax1.set_title('Frequency vs time, channel 0')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_title('Frequency vs time, channel 1')
ax1.set_xlim(0.02,0.05)
ax2.set_xlim(0.02,0.05)
plt.show()

# Calculate the fourier transfor, then shift the zero value to the middle
frequency1 = np.fft.fftshift(np.fft.fft(channel_0))
frequency2 = np.fft.fftshift(np.fft.fft(channel_1))

# Calculate the x-axis of the fourier transform with the calculated time step
f_axis = np.fft.fftshift(np.fft.fftfreq(len(channel_0), dt))

# Plot the fourier transform of the two channels
plt.plot(f_axis, abs(frequency1))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0)
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
fig.subplots_adjust(hspace=0.4)
ax1.plot(f_axis, abs(frequency1))
ax2.plot(f_axis, abs(frequency2))
ax1.set_title('Frequency vs time, channel 0')
ax1.set_ylabel('Fourier Coefficients')
ax2.set_ylabel('Fourier Coefficients')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_title('Frequency vs time, channel 1')
ax1.set_xlim(0)
ax2.set_xlim(0)
plt.show()





freq_range = (np.where(f_axis > 880)[0][0], np.where(f_axis < -880)[0][-1]+1)

frequency1[freq_range[0]:] = 0
frequency1[:freq_range[1]] = 0

frequency2[freq_range[0]:] = 0
frequency2[:freq_range[1]] = 0

channel_0_out = np.fft.ifft(np.fft.ifftshift(frequency1))
channel_1_out = np.fft.ifft(np.fft.ifftshift(frequency2))



fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
fig.subplots_adjust(hspace=0.4)
ax1.plot(lst, channel_0_out)
ax2.plot(lst, channel_1_out)
ax1.set_ylabel('Frequency (Hz)')
ax1.set_title('Frequency vs time, channel 0')

ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_title('Frequency vs time, channel 1')

plt.show()



data_out = empty(data.shape, dtype = data.dtype)
data_out[:, 0] = channel_0_out
data_out[:, 1] = channel_1_out
write('GraviteaTime_lpf.wav', sample, data_out)


