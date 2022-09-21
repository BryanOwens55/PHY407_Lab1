# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:12:19 2022

@author: bryan
"""
import numpy as np
import matplotlib.pyplot as plt

# PART A


def Q(u):
    q = np.zeros(len(u))
    for i in range(len(u)):
        q[i] = 1 - 8*u[i] + 28*u[i]**2 - 56*u[i]**3 + 70*u[i]**4 - \
            56*u[i]**5 + 28*u[i]**6 - 8*u[i]**7 + u[i]**8
    return q


def P(u):
    p = (1-u)**8
    return p


u = np.linspace(0.98, 1.02, 500)

p = P(u)
q = Q(u)

plt.plot(u, p)
plt.xlabel('P(u)')
plt.ylabel('u')
plt.title('P(u) vs u')
plt.show()

plt.plot(u, q)
plt.xlabel('Q(u)')
plt.ylabel('u')
plt.title('Q(u) vs u')
plt.show()


# PART B

plt.hist(p-q, 90)
plt.show()

sigma_1 = np.std(p-q)
print(sigma_1)



def Sigma(x):
    C = 1e-16
    N = len(x)
    mean_of_square = 0
    for i in range(N):
        mean_of_square += x[i]**2
    mean_of_square = mean_of_square / N
    #print(np.sqrt(mean_of_square))
    return C * np.sqrt(mean_of_square) * np.sqrt(N)


def Sigma2(u):
    C = 1e-16
    N = len(u)
    lst = u*0
    for i in range(N):
        square_mean = ((1**2) + (8*u[i])**2 + (28*u[i]**2)**2 + (56*u[i]**3)**2\
                       + (70*u[i]**4)**2 + (56*u[i]**5)**2 + (28*u[i]**6)**2 + \
                           (8*u[i]**7)**2 + (u[i]**8)**2) / 9
        lst[i] = C * np.sqrt(square_mean) * np.sqrt(9)
    return lst


print(np.mean(Sigma2(u)))


# PART C

u = np.linspace(0.98, 0.984, 500)

f = abs(P(u) - Q(u)) / abs(P(u))


plt.plot(u, f)
plt.show()





u = np.linspace(0.98, 1.02, 500)

def F(u):
    f = np.zeros(len(u))
    for i in range(len(u)):
        f[i] = u[i]**8 / ((u[i]**4)*(u[i]**4))
    return f

plt.plot(u, F(u)-1)
plt.show()







