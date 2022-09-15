import numpy as np
import matplotlib.pyplot as plt


# Set variables
delta_t = 0.0001 #0.0001
G = 39.5
M = 1
alpha = 0.01

# Create lists
t = np.arange(0,1,delta_t)
x = 0*t 
y = 0*t
v_x = 0*t
v_y = 0*t
a_x = np.zeros(len(t)-1)
a_y = np.zeros(len(t)-1)
# Set up initial conditions
x[0] = 0.47
v_y[0] = 8.17

# Calculate acceleration values
def acceleration(x_1, x_2):
    # Calculate radial value
    r = (x_1**2 + x_2**2)**0.5
    # Calculate acceleration values
    a_1 = -1*G*M*x_1/(r**3)
    a_2 = -1*G*M*x_2/(r**3)
    return a_1, a_2

# Calculate velocity values
def velocity(v1_old, v2_old, a1, a2):
    v1_new = v1_old + a1 * delta_t
    v2_new = v2_old + a2 * delta_t
    return v1_new, v2_new

# Calculate position values
def position(x1_old, x2_old, v1, v2, a1, a2):
    x1_new = x1_old + v1*delta_t + a1*(delta_t**2)
    x2_new = x2_old + v2*delta_t + a2*(delta_t**2)
    return x1_new, x2_new
    
# Start for loop
for i in range(len(t)-1):
    
    # Calculate acceleration values
    a_x[i], a_y[i] = acceleration(x[i],y[i])
    
    # Calculate Velocity values
    v_x[i+1], v_y[i+1] = velocity(v_x[i], v_y[i], a_x[i], a_y[i])
    
    # Calculate position values    
    x[i+1], y[i+1] = position(x[i], y[i], v_x[i+1], v_y[i+1], a_x[i], a_y[i])

# Plot x vs y, x vs t, y vs t
plt.plot(t, x)
plt.xlabel('Time (Earth-years)')
plt.ylabel('X-position (AU)')
plt.title('Mercury\'s x-position vs time')
plt.show()
plt.plot(t, y)
plt.xlabel('Time (Earth-years)')
plt.ylabel('Y-position (AU)')
plt.title('Mercury\'s y-position vs time')
plt.show()
plt.plot(x, y)
plt.ylabel('X-position (AU)')
plt.ylabel('Y-position (AU)')
plt.title('Phase plot of Mercury\'s orbit')
plt.show()



r = ((x**2) + (y**2))**0.5
v = ((v_x**2) + (v_y**2))**0.5
angular_v = r*v
print(angular_v[0],angular_v[-1])



# PART C

# Create lists
t = np.arange(0,1,delta_t)
x = 0*t 
y = 0*t
v_x = 0*t
v_y = 0*t
a_x = np.zeros(len(t)-1)
a_y = np.zeros(len(t)-1)
# Set up initial conditions
x[0] = 0.47
v_y[0] = 8.17

# Calculate acceleration values
def acceleration(x_1, x_2):
    # Calculate radial value
    r = (x_1**2 + x_2**2)**0.5
    # Calculate acceleration values
    a_1 = -1*G*M*x_1*(1+(alpha/r**2))/(r**3)
    a_2 = -1*G*M*x_2/(r**3)
    return a_1, a_2

# Calculate velocity values
def velocity(v1_old, v2_old, a1, a2):
    v1_new = v1_old + a1 * delta_t
    v2_new = v2_old + a2 * delta_t
    return v1_new, v2_new

# Calculate position values
def position(x1_old, x2_old, v1, v2, a1, a2):
    x1_new = x1_old + v1*delta_t + a1*(delta_t**2)
    x2_new = x2_old + v2*delta_t + a2*(delta_t**2)
    return x1_new, x2_new
    
# Start for loop
for i in range(len(t)-1):
    
    # Calculate acceleration values
    a_x[i], a_y[i] = acceleration(x[i],y[i])
    
    # Calculate Velocity values
    v_x[i+1], v_y[i+1] = velocity(v_x[i], v_y[i], a_x[i], a_y[i])
    
    # Calculate position values    
    x[i+1], y[i+1] = position(x[i], y[i], v_x[i+1], v_y[i+1], a_x[i], a_y[i])

 
# Plot x vs y, x vs t, y vs t
plt.plot(t, x)
plt.xlabel('Time (Earth-years)')
plt.ylabel('X-position (AU)')
plt.title('Mer\'s x-position vs time (with relativistic correction)')
plt.show()
plt.plot(t, y)
plt.xlabel('Time (Earth-years)')
plt.ylabel('Y-position (AU)')
plt.title('Mercury\'s y-position vs time (with relativistic correction)')
plt.show()
plt.plot(x, y)
plt.xlabel('X-position (AU)')
plt.ylabel('Y-position (AU)')
plt.title('Phase plot of Mercury\'s orbit (with relativistic correction)')
plt.show()

