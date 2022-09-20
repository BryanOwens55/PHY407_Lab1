# -*- coding: utf-8 -*-
import numpy as np
"""
Prupose: To calculate and study the formulae of standard deviation
"""

SoL = np.loadtxt('cdata.txt') # Speed of light data; 10^3 km/s

# PART B
def Mean(lst):
    mean = 0
    for i in range(len(lst)):
        mean = mean + lst[i]
    mean = mean / len(lst)
    return mean


def STD1(lst):
    mean = Mean(lst)
    sum = 0
    for i in range(len(lst)):
        sum += (lst[i] - mean)**2
    return np.sqrt(sum/(len(lst-1)))

def STD2(lst):
    mean = Mean(lst)
    sum = 0
    n = len(lst)
    for i in range(len(lst)):
        sum += lst[i]**2
    sum -= n*mean**2
    return np.sqrt(abs(sum/(len(lst-1))))




def Error(x, y):
    return (x - y) / y

#print(Error(STD2(SoL), np.std(SoL)), Error(STD1(SoL), np.std(SoL)))


# PART C

sequence_1 = np.random.normal(0., 1., 2000)
sequence_2 = np.random.normal(1e7, 1., 2000)

'''
print(Error(STD2(sequence_1), np.std(sequence_1)), Error(STD1(sequence_1), np.std(sequence_1)))

print()

print(Error(STD2(sequence_2), np.std(sequence_2)), Error(STD1(sequence_2), np.std(sequence_2)))
'''


def STD3(lst):
    mean = Mean(lst)
    sum = 0
    n = len(lst)
    for i in range(len(lst)):
        sum += lst[i]**2 - 2*lst[i]*mean
    sum += n*mean**2
    return np.sqrt(abs(sum/(len(lst-1))))





print(Error(STD3(SoL), np.std(SoL)), Error(STD2(SoL), np.std(SoL)))

