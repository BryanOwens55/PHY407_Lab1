'''
Authors: Bryan Owens, Dharmik Patel
Purpose: To study the forward and central difference schemes
Collaboration: Code was evenly created and edited by both lab partners
'''
# Import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt


# Method that takes in x and returns the value of the function f(x)
def F(x):
    return np.exp(-1*x**2)

# Method that takes in x and number of h values and returns a list of derivative values
# at x with different h values using the forward difference scheme
def Forward(x, n):
    h = np.zeros(n)
    answer = h*0
    for i in range(len(h)):
        # Calculate the values of h
        h[i] = 10**(-16 + i)
        # Calcualte the analytical answer
        answer[i] = (abs(F(x + h[i]) - F(x))/(x + h[i] - x))
    return answer, h

# Method that computes the difference between analytical calculation vs theoretical value
def Error(answer):
    # Theoretical value of the derivative at x=0.5
    correct = 0.778800783071    
    return abs(correct - answer)



# Part A

# Call forward function and print the returned values for 17 values of h
answer, h = Forward(0.5, 17)
err = abs(Error(answer))
print(answer, h, err)



# Part B

# Plot the log-log plot of the error of the forward difference scheme with given h values
plt.loglog(h, err)
plt.ylabel('Error of derivative')
plt.xlabel('Value of h used in derivative calculation')
plt.title('Error vs h Value (using forward derivative)')
plt.show()


# Using equation 5.91 from the textbook caluculate and plot the theoretical error of
# the forward difference scheme
C = 1e-16
epsilon = 2*C*abs(F(0.5))/h + 0.5*abs(-0.7788007830714049)*h
plt.loglog(h, epsilon)
plt.ylabel('Error of derivative')
plt.xlabel('Value of h used in derivative calculation')
plt.title('Error vs h Value (using equation 5.91 from textbook)')
plt.show()



# Part C


# Method that takes in x and number of h values and returns a list of derivative values
# at x with different h values using the central difference scheme
def Centred(x, n):
    h = np.zeros(n)
    answer = h*0
    for i in range(len(h)):
        h[i] = 10**(-16 + i)
        answer[i] = (abs(F(x + h[i]) - F(x - h[i]))/(x + h[i] - (x - h[i])))
    return answer, h

# Using the central method, calculate the derivative at x=0.5 with 17 different h values
# and also calculate the error
answer2, h2 = Centred(0.5, 17)
err2 = abs(Error(answer2))

# Plot the error of both the forward and central difference scheme on a log-log plot
plt.loglog(h, err)
plt.ylabel('Error of derivative')
plt.xlabel('Value of h used in derivative calculation')
plt.title('Error vs h Value (using central and forward derivative)')
plt.loglog(h2, err2)
plt.legend(['Forward difference scheme', 'Central difference scheme'])





