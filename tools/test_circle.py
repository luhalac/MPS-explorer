# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:03:16 2021

@author: Lucia
"""
import numpy as np
import matplotlib.pyplot as plt
import circle_fit

#simulated data points
R = 10
N = 100

deltax = np.random.rand(N)
deltay = np.random.rand(N)

x = np.zeros(N)
y = np.zeros(N)

for i in np.arange(N):
    theta = i*2*np.pi/N
    x[i] = R*np.cos(theta) + deltax[i]
    y[i] = R*np.sin(theta) + deltay[i]
    
xy = np.vstack((x, y)).T
    
circle = circle_fit.least_squares_circle(xy)
xc = circle[0]
yc = circle[1]
Rc = circle[2]

plotcircle = circle_fit.plot_data_circle(x, y, xc, yc, Rc)
    