# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 04:09:20 2022

@author: bryan
"""

import numpy as np
import matplotlib.pyplot as plt



def F(x):
    return np.exp(-1*x**2)

def Forward(x, n):
    h = np.zeros(n)
    answer = h*0
    for i in range(len(h)):
        h[i] = 10**(-16 + i)
        answer[i] = (abs(F(x + h[i]) - F(x))/(x + h[i] - x))
    return answer, h

def Error(answer):
    correct = 0.778800783071    
    return abs(correct - answer)



# Part A


answer, h = Forward(0.5, 17)
err = abs(Error(answer))
print(answer, h, err)



# Part B


plt.loglog(h, err)
plt.ylabel('Error of derivative')
plt.xlabel('Value of h used in derivative calculation')
plt.title('Error vs h Value (using forward derivative)')
plt.show()


C = 1e-16
epsilon = 2*C*abs(F(0.5))/h + 0.5*abs(-0.7788007830714049)*h
plt.loglog(h, epsilon)
plt.ylabel('Error of derivative')
plt.xlabel('Value of h used in derivative calculation')
plt.title('Error vs h Value (using equation 5.91 from textbook)')
plt.show()



# Part C


def Centred(x, n):
    h = np.zeros(n)
    answer = h*0
    for i in range(len(h)):
        h[i] = 10**(-16 + i)
        answer[i] = (abs(F(x + h[i]) - F(x - h[i]))/(x + h[i] - (x - h[i])))
    return answer, h

answer2, h2 = Centred(0.5, 17)
err2 = abs(Error(answer2))

plt.loglog(h, err)
plt.ylabel('Error of derivative')
plt.xlabel('Value of h used in derivative calculation')
plt.title('Error vs h Value (using central and forward derivative)')
plt.loglog(h2, err2)
plt.legend(['Forward difference scheme', 'Central difference scheme'])





