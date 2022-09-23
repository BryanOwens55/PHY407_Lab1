'''
Authors: Bryan Owens, Dharmik Patel
Purpose: To study the effect of round off errors in summations and in products
Collaberation: Code was evenly created and edited by both lab partners
'''

# Import Functions
import numpy as np
import matplotlib.pyplot as plt


# PART A

# Method that takes in a value of u and returns Q at that u
def Q(u):
    q = np.zeros(len(u))
    for i in range(len(u)):
        q[i] = 1 - 8*u[i] + 28*u[i]**2 - 56*u[i]**3 + 70*u[i]**4 - \
            56*u[i]**5 + 28*u[i]**6 - 8*u[i]**7 + u[i]**8
    return q

# Method that takes in a value of u and returns P at that u
def P(u):
    p = (1-u)**8
    return p

# Create a list of u's to pass as arguments to P and Q
u = np.linspace(0.98, 1.02, 500)

# Call P and Q functions and save the lists
p = P(u)
q = Q(u)

# Plot P(u) vs u and Q(u) vs u
plt.plot(u, p)
plt.ylabel('P(u)')
plt.xlabel('u')
plt.title('P(u) vs u')
plt.show()

plt.plot(u, q)
plt.ylabel('Q(u)')
plt.xlabel('u')
plt.title('Q(u) vs u')
plt.show()


# PART B
# Plot the difference of p and q vs u to see the round off error
plt.plot(u, p-q)
plt.ylabel('P(u)-Q(u)')
plt.xlabel('u')
plt.title('P(u)-Q(u) vs u')
plt.show()

# Plot the histogram of the differnce of p and q to see the distribution of
# the error
plt.hist(p-q, 90)
plt.ylabel('Frequency')
plt.xlabel('P(u)-Q(u)')
plt.title('Histogram of P(u)-Q(u)')
plt.show()

# Compute and print the standard deivation of the difference of p and q
sigma_1 = np.std(p-q,  ddof=1)
print('Standard deviation of p-q: ', sigma_1)



# Method that takes in a list x and returns the round off error using
# equation 3 from the lab handout
def Sigma(u):
    C = 1e-16
    N = len(u)
    lst = u*0
    for i in range(N):
        square_mean = (P(u[i]) + (1**2) + (8*u[i])**2 + (28*u[i]**2)**2 + (56*u[i]**3)**2\
                       + (70*u[i]**4)**2 + (56*u[i]**5)**2 + (28*u[i]**6)**2 + \
                           (8*u[i]**7)**2 + (u[i]**8)**2) / 10
        lst[i] = C * np.sqrt(square_mean) / np.sqrt(10)
    return lst


# Print the round off error caused by summation
sigma = np.mean(Sigma(u)) 
print('Round off error:  ', sigma)



# PART C



# Finding a value close to 1.0 that with u<1 and print value
u = np.linspace(0.98, 0.9831, 500)
u2 = np.array([0.9817])
print('Fractional error: ', abs(sigma/(P(u2)-Q(u2))[0]))

u = np.linspace(0.98, 0.984, 500)
f = abs(P(u) - Q(u)) / abs(P(u))


# Plot the fractional error of q taking p to be the true value
plt.plot(u, f)
plt.ylabel('|(p-q)/p|')
plt.xlabel('u')
plt.title('Fractional error of q(u)')
plt.show()


# PART D

# Method that takes in a u and returns f(u), equation given in question
def F(u):
    f = np.zeros(len(u))
    for i in range(len(u)):
        f[i] = u[i]**8 / ((u[i]**4)*(u[i]**4))
    return f

# Plot f(u) - 1 to see the round off error
plt.plot(u, F(u)-1)
plt.ylabel('f(u)-1')
plt.xlabel('u')
plt.title('F(u) - 1 vs u')
plt.show()

# Print the standard deviation and the round off error value of f(u)
C = 1e-16
print('Stardard deviation of F(u): ', np.std(F(u),  ddof=1))
print('Round off error caused by product/quotient: ', np.mean(F(u)*C))




