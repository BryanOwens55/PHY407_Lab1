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





answer, h = Forward(0.5, 17)
err = abs(Error(answer))
plt.loglog(h, err)




def Centred(x, n):
    h = np.zeros(n)
    answer = h*0
    for i in range(len(h)):
        h[i] = 10**(-16 + i)
        answer[i] = (abs(F(x + h[i]) - F(x - h[i]))/(x + h[i] - (x - h[i])))
    return answer, h

answer, h = Centred(0.5, 17)
err = abs(Error(answer))
plt.loglog(h, err)






