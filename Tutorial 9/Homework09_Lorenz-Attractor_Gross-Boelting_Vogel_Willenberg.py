# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 09:  Many Species Population Dynamics
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np; import matplotlib.pyplot as plt
import numpy.linalg as linalg


def rk4_step(y0, x0, f, h, f_args={}):
    ''' Simple python implementation for one RK4 step.
        Inputs:
            y_0    - M x 1 numpy array specifying all variables of the ODE at the current time step
            x_0    - current time step
            f      - function that calculates the derivates of all variables of the ODE
            h      - time step size
            f_args - Dictionary of additional arguments to be passed to the function f
        Output:
            yp1 - M x 1 numpy array of variables at time step x0 + h
            xp1 - time step x0+h
    '''
    k1 = h*f(y0, x0, **f_args)
    k2 = h*f(y0+k1/2., x0+h/2., **f_args)
    k3 = h*f(y0+k2/2., x0+h/2., **f_args)
    k4 = h*f(y0+k3, x0+h, **f_args)

    xp1 = x0+h
    yp1 = y0+1./6.*(k1+2.*k2+2.*k3+k4)

    return (yp1,xp1)

def rk4(y0, x0, f, h, n, f_args={}):
    ''' Simple implementation of RK4
        Inputs:
            y_0    - M x 1 numpy array specifying all variables of the ODE at the current time step
            x_0    - current time step
            f      - function that calculates the derivates of all variables of the ODE
            h      - time step size
            n      - number of steps
            f_args - Dictionary of additional arguments to be passed to the function f
        Output:
            yn - N+1 x M numpy array with the results of the integration for every time step (includes y0)
            xn - N+1 x 1 numpy array with the time step value (includes start x0)
    '''
    yn = np.zeros((n+1, y0.shape[0]))
    xn = np.zeros(n+1)
    yn[0,:] = y0
    xn[0] = x0

    for n in np.arange(1,n+1,1):
        yn[n,:], xn[n] = rk4_step(y0 = yn[n-1,:], x0 = xn[n-1], f = f, h = h, f_args = f_args)

    return (yn,xn)


def Lorenz_attractor(y,x,r,b,sigma):
    ''' Three coupled ordinary differential equations of first order '''
    yn = np.ones(3)
    x = y[0]
    y = y[1]
    z = y[2]

    yn[0] = -sigma*(x-y)
    yn[1] = r*x-y-x*z
    yn[2] = x*y-b*z
    return yn


r = 0.5
b = 8./3.
sigma = 10.
initial_condition = np.array([0.1,0.1,0.1])

yn, xn = rk4(initial_condition,0,Lorenz_attractor,
             0.001,int(1000),{'r':r, 'b':b, 'sigma':sigma})
