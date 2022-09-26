
import numpy as np
import matplotlib.pyplot as plt
from gaussxw import gaussxwab





def H(n, x):
    H = np.ones(max(n+1,2))
    H[1] = 2*x
    if n > 1:
        for i in range(2, n+1):
            H[i] = 2*x*H[i-1] - 2*n*H[i-2]
    return H[n]


def Psi(n, x):
    psi = x*0
    for i in range(len(psi)):
        psi[i] = np.exp(-1*(x[i]**2)/2)*H(n, x[i])
        psi[i] = psi[i] / np.sqrt((2**n)*np.math.factorial(n)*np.sqrt(np.pi))
    return psi


x = np.linspace(-4,4,500)
plt.plot(x, Psi(0, x))
plt.plot(x, Psi(1, x))
plt.plot(x, Psi(2, x))
plt.plot(x, Psi(3, x))
plt.show()

x = np.linspace(-10,10,500)
plt.plot(x, Psi(30, x))
plt.show()


def F(n, z):
    f = (z/(1-z**2))**2 * abs((Psi(n, z/(1-z**2)))**2)
    return (1+z**2)*f/(1-z**2)**2 



def mean_x(n, N, x):
    u,w = gaussxwab(N, -1, 1)
    s = 0
    f = F(n, u)
    for i in range(len(x)):
        s += w[i]*f[i]
    print(s)



x = np.linspace(-4,4,500)
n = 0
mean_x(n, 500, x)




