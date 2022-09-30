
import numpy as np
import matplotlib.pyplot as plt
from gaussxw import gaussxwab



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



def Psi(n, x):
    psi = np.exp(-1*(x**2)/2)*H(n, x) / np.sqrt((2**n)*np.math.factorial(n)*np.sqrt(np.pi))
    return psi


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

x = np.linspace(-10,10,1000)
plt.plot(x, Psi(30, x))
plt.xlabel('x-Position')
plt.ylabel('Probability density')
plt.title('Probability density vs x-position')
plt.show()




def Integrand_1(n, z):
    x = z/(1-(z**2))
    f = x**2 * abs((Psi(n, x))**2)
    return (1+z**2)*f/(1-z**2)**2 



def Mean_x(n, N, x):
    u,w = gaussxwab(N, -1, 1)
    s = 0
    f = Integrand_1(n, u)
    for i in range(N):
        s += w[i]*f[i]
    return s



x = np.linspace(-4,4,100)
n = 5
print(np.sqrt(Mean_x(n, 100, x)))



def Integrand_2(n, z):
    x = z/(1-(z**2))
    H_1 = H(n, x)
    H_2 = H(n+1, x)
    
    f = np.exp(-1*(x**2)/2) * (x*H_1 - H_2)
    f = f / np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi))
    return f**2


def Mean_p(n, N, x):
    u,w = gaussxwab(N, -1, 1)
    s = 0
    f = Integrand_2(n, u)
    for i in range(N):
        s += w[i]*f[i]
    return s
    
    
def Energy(n, N, x):
    mean_x = Mean_x(n, N, x)
    mean_p = Mean_p(n, N, x)
    return 0.5 * (mean_x + mean_p)
    
    

x = np.linspace(-4,4,100)
n = 1
print(Mean_p(n, 100, x))

print(Energy(n, 100, x))


x_uncertainty = []
p_uncertainty = []
energy = []

for i in range(16):
    x_uncertainty.append(np.sqrt(Mean_x(i, 100, x)))
    p_uncertainty.append(np.sqrt(Mean_p(i, 100, x)))
    energy.append(Energy(i, 100, x))

print(x_uncertainty, p_uncertainty)
print(energy)

n = np.arange(16)
temp = np.array(x_uncertainty) * np.array(p_uncertainty)
print(temp)

plt.plot(temp)
plt.xlabel('n value')
plt.ylabel('x-uncertainty * p-uncertainty')
plt.title('Product of x and p uncertainty vs n')





