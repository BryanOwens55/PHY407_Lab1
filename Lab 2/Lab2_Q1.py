"""
Authors: Bryan Owens, Dharmik Patel
Purpose: To calculate and study the formulae of standard deviation
Collaboration: Code was evenly created and edited by both lab partners
"""

# Import numpy and the speed of light data file
import numpy as np
SoL = np.loadtxt('cdata.txt') # Speed of light data; 10^3 km/s


# PART B

# Method that takes in a list and returns the mean of the list
def Mean(lst):
    mean = 0
    for i in range(len(lst)):
        mean = mean + lst[i]
    mean = mean / len(lst)
    return mean

# Method that takes in a list and returns the standard deviation using equation
# 1 from the lab handout (2-pass method)
def STD1(lst):
    mean = Mean(lst)
    sum1 = 0
    for i in range(len(lst)):
        sum1 += (lst[i] - mean)**2
    return np.sqrt(sum1/(len(lst)-1))

# Method that takes in a list and reutrns the standard deviation using equation
# 2 from the lab handout (1-pass method)
def STD2(lst):
    mean = 0
    sum2 = 0
    n = len(lst)
    for i in range(len(lst)):
        sum2 += lst[i]**2
        mean += lst[i]
    mean = mean/len(lst)
    sum2 -= n*mean**2
    return np.sqrt(abs(sum2/(len(lst)-1)))



# Method that takes in a two values, one the true value (y) and the one you are
# calculating the error for (x)
def Error(x, y):
    return (x - y) / y

# Print standard deviation of speed of light data set using numpy, method 1 and method 2
print('Standard deviation using numpy: ', np.std(SoL, ddof=1), ', Standard deviation using method 1: ', STD1(SoL))
print('Standard deviation using method 2: ', STD2(SoL))

# Print the error of method 1 and 2 using numpy as the true value
print('Error of first method: ', Error(STD1(SoL), np.std(SoL, ddof=1))) 
print('Error of second method: ', Error(STD2(SoL), np.std(SoL, ddof=1)))


# PART C

# Create random sequence with given conditions
sequence_1 = np.random.normal(0., 1., 2000)
sequence_2 = np.random.normal(1e7, 1., 2000)


# Print error of standard deviation of method 1 and 2 using numpy as true value for sequence 1
print('Error using method 2: ', Error(STD2(sequence_1), np.std(sequence_1, ddof=1)))
print('Error using method 1: ', Error(STD1(sequence_1), np.std(sequence_1, ddof=1)))

# Print values of STD using methods 1, 2, and numpy for sequence 1
print('STD method 1: ', STD1(sequence_1))
print('STD method 2: ', STD2(sequence_1))
print('STD numpy: ', np.std(sequence_1, ddof=1))


# Print error of standard deviation of method 1 and 2 using numpy as true value for sequence 2
print('Error using method 2: ', Error(STD2(sequence_2), np.std(sequence_2, ddof=1)))
print('Error using method 1: ', Error(STD1(sequence_2), np.std(sequence_2, ddof=1)))

# Print values of STD using methods 1, 2, and numpy for sequence 2
print('STD method 1: ', STD1(sequence_2))
print('STD method 2: ', STD2(sequence_2))
print('STD numpy: ', np.std(sequence_2, ddof=1))


# Attempt at improving STD2
def STD3(lst):
    mean = 0
    sum2 = 0
    shift = lst[0]
    n = len(lst)
    for i in range(len(lst)):
        sum2 += (lst[i] - shift)**2
        mean += (lst[i] - shift)
    mean = mean/len(lst)
    sum2 = sum2/mean
    sum2 -= n*mean
    return np.sqrt(abs(sum2*mean/(len(lst)-1)))



# Print error of standard deviation of new method using numpy as true value for speed of light data
print('Error using method 3: ', Error(STD3(sequence_2), np.std(sequence_2, ddof=1)))
print('Error using method 2: ', Error(STD2(sequence_2), np.std(sequence_2, ddof=1)))
print('STD method 3: ', STD3(sequence_2))


