'''
Authors: Bryan Owens, Dharmik Patel
Purpose: To study the forward and central difference schemes
Collaboration: Code was evenly created and edited by both lab partners
'''
# Import functions numpy, matplotlib, and gaussxw
import numpy as np
import matplotlib.pyplot as plt
from gaussxw import gaussxwab


# Part A

# Method that takes in a range of values x along with the state n and returns
# the value of H_n
def H(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    else:
        H = np.ones((n+1,len(x)))
        H[1] = 2*x
        for i in range(2, n+1):
            H[i] = 2*x*H[i-1] - 2*(i-1)*H[i-2]
    return H[n]


# Method that takes in a range of values x along with the state n and returns
# the value of Psi
def Psi(n, x):
    psi = np.exp(-1*(x**2)/2)*H(n, x) / np.sqrt((2**n)*np.math.factorial(n)*np.sqrt(np.pi))
    return psi


# Plot, using the psi method, the first four states of the function psi (n=0,1,2,3)
# from x =-4 to x=4
x = np.linspace(-4,4,100)
plt.plot(x, Psi(0, x))
plt.plot(x, Psi(1, x))
plt.plot(x, Psi(2, x))
plt.plot(x, Psi(3, x))
plt.legend(['n = 0', 'n = 1', 'n = 2', 'n = 3'])
plt.xlabel('x-Position')
plt.ylabel('Probability density')
plt.title('Probability density vs x-position')
plt.show()


# Part B

# Plot the function psi at the n=30 state from x=-10 to x=10
x = np.linspace(-10,10,1000)
plt.plot(x, Psi(30, x))
plt.xlabel('x-Position')
plt.ylabel('Probability density')
plt.title('Probability density vs x-position')
plt.show()


# Part C

# Method that takes in state number n and range of values of x and returns the integrand
# for the mean-square-position integration
def Integrand_1(n, z):
    x = z/(1-(z**2))
    f = x**2 * abs((Psi(n, x))**2)
    return (1+z**2)*f/(1-z**2)**2 


# Method that integrates Integrand_1 using Gaussian quadrature and returns the mean-square of x
def Mean_x(n, N, x):
    u,w = gaussxwab(N, -1, 1)
    s = 0
    f = Integrand_1(n, u)
    for i in range(N):
        s += w[i]*f[i]
    return s


# Print the square root of the mean-square of the position of n=5 (test to get approx 2.35)
x = np.linspace(-4,4,100)
n = 5
print(np.sqrt(Mean_x(n, 100, x)))


# Method that takes in state number n and range of values of x and returns the integrand
# for the mean-square-momentum integration
def Integrand_2(n, z):
    x = z/(1-(z**2))
    H_1 = H(n, x)
    H_2 = H(n+1, x)
    
    f = np.exp(-1*(x**2)/2) * (x*H_1 - H_2)
    f = f / np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi))
    return f**2

# Method that integrates Integrand_2 using Gaussian quadrature and returns the mean-square of momentum
def Mean_p(n, N, x):
    u,w = gaussxwab(N, -1, 1)
    s = 0
    f = Integrand_2(n, u)
    for i in range(N):
        s += w[i]*f[i]
    return s
    

# Method that takes in the state value n, integration steps N, and an array of x values to calculate
# the value of the energy at state n
def Energy(n, N, x):
    mean_x = Mean_x(n, N, x)
    mean_p = Mean_p(n, N, x)
    return 0.5 * (mean_x + mean_p)
    
    
# 
x = np.linspace(-4,4,100)
n = 1
print(Mean_p(n, 100, x))
print(Energy(n, 100, x))


# Setup lists for uncertainties of x and p, and a list for energy values
x_uncertainty = []
p_uncertainty = []
energy = []

# Loop through values of n and using previous method calculate uncertainty of x, uncertainty of p
# and energy
for i in range(16):
    x_uncertainty.append(np.sqrt(Mean_x(i, 100, x)))
    p_uncertainty.append(np.sqrt(Mean_p(i, 100, x)))
    energy.append(Energy(i, 100, x))


# Print values that were calculated in for-loop
print(x_uncertainty, p_uncertainty)
print(energy)


# Calculate and plot the product of uncertainty of x and uncertainty of p
n = np.arange(16)
product = np.array(x_uncertainty) * np.array(p_uncertainty)

plt.plot(product)
plt.xlabel('n value')
plt.ylabel('x-uncertainty * p-uncertainty')
plt.title('Product of x and p uncertainty vs n')





