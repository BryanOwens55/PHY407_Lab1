# Imports
from random import random 
import numpy as np 
import matplotlib.pyplot as plt

# Function being integrated
def f(x):
    return (x**(-0.5))/(1+np.exp(x))


# Method to calculate integral value using mean value method
def Mean_value():
    b = 1
    a = 0
    N = 10000 
    f_avg = 0
    for i in range(N): 
        x = (b-a)*random()
        f_avg += f(x) 
    # Calculate and return integral
    I = (b-a)*f_avg/N 
    return I


# Method to calculate integral value using mean value method
def Imp_Sampling():
    b = 1
    a = 0
    
    # Sub-method to calculate omega
    def w(x):
        return (x**(-0.5))
    
    N = 10000 
    Sigma = 0
    for i in range(N): 
        x = ((b-a)*random())**2
        # Calculate sum
        Sigma += 2*f(x) / w(x)
    # Calculate and return integral
    I = Sigma/N 
    return I


# Set up arrays for both methods
sampling = []
mean = []
# Run both methods 100 times
for i in range(100):
    # Part (a) mean value method
    mean.append(Mean_value())
    # Part (b) importance sampling method
    sampling.append(Imp_Sampling())
    
# Plot the histogram of the two methods and restrict 0.8 to 0.88
plt.hist(sampling, bins=10)
plt.hist(mean, bins=10, zorder=0)
plt.legend(['Importance Sampling Method', 'Mean Value Method'])
plt.title('Importance Sampling vs Mean Value method')
plt.xlabel('Integral Value')
plt.ylabel('Frequency')
plt.xlim(0.80,0.88)
plt.show()


