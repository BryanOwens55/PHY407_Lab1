"""
Created on Thu Oct 20 16:58:17 2022

@author: bryanp
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from numpy import arange 
import copy

eps = 1
alpha = 1
m = 1

def f(r,t): 
    x = r[0]
    y = r[1] 
    fx = x*24*eps*((-1*alpha**6/(x**2+y**2)**7) + 2*(alpha**12/(x**2+y**2)**13)) / (m)
    fy = y*24*eps*((-1*alpha**6/(x**2+y**2)**7) + 2*(alpha**12/(x**2+y**2)**13)) / (m)
    return np.array([fx,fy] ,float) 

def Energy(r):
    return

dt = 0.01
time = np.arange(0,1000*dt,dt)



N = 16
Lx = 4
Ly = 4
dx = Lx/sqrt(N)
dy = Ly/sqrt(N)
x_grid = arange(dx/2, Lx, dx)
y_grid = arange(dy/2, Ly, dy)
xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
x_initial = xx_grid.flatten()
y_initial = yy_grid.flatten()


x_pos = np.array(x_initial, float)
y_pos = np.array(y_initial, float)

plt.plot(x_initial, y_initial, '.')

def Nparticles(x_pos,y_pos,t):
    acceleration_x = x_pos*0
    acceleration_y = y_pos*0
    for i in range(len(x_initial)):
        for j in range(len(x_initial)):
            if i != j:
                force = f([x_pos[i]-x_pos[j],y_pos[i]-y_pos[j]],t)
                acceleration_x[i] += force[0]
                acceleration_y[i] += force[1]
    return acceleration_x, acceleration_y

ax = x_initial*0
ay = y_initial*0

vx = x_initial*0
vy = y_initial*0


    
init = Nparticles(x_pos,y_pos,0)
vx = (dt/2) * init[0] 
vy = (dt/2) * init[1]
positions = [[x_initial, y_initial]]


for t in time:
    ax = x_initial*0
    ay = y_initial*0
    
    ax, ay = Nparticles(x_pos, y_pos,t)
    
    x_pos += dt*vx
    y_pos += dt*vy
    
    positions.append([copy.copy(x_pos),copy.copy(y_pos)])
    plt.plot(x_pos, y_pos, '.', color='blue')
    vx += dt*ax
    vy += dt*ay    
plt.show()



for i in range(len(positions)):
    plt.plot(positions[i][0], positions[i][1], '.')
plt.show()
positions = np.array(positions)

legend = []
for i in range(16):
    plt.plot(positions[:,0][:,i], positions[:,1][:,i])
    #legend.append('Particle ' + str(i+1))
plt.xlabel('X-position')
plt.ylabel('Y-position')
plt.title('X-position vs Y-position')
#plt.legend(legend)
