# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 03:  Numerical Simulation of the Three-Body Problem
                using the 4th Order Runge-Kutta Method
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np
import matplotlib.pyplot as plt

def rk4_step(y0, x0, f, h, f_args = {}):
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
    k1 = h * f(y0, x0, **f_args)
    k2 = h * f(y0 + k1/2., x0 + h/2., **f_args)
    k3 = h * f(y0 + k2/2., x0 + h/2., **f_args)
    k4 = h * f(y0 + k3, x0 + h, **f_args)

    xp1 = x0 + h
    yp1 = y0 + 1./6.*(k1 + 2.*k2 + 2.*k3 + k4)

    return(yp1,xp1)

def rk4(y0, x0, f, h, n, f_args = {}):
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

    return(yn, xn)




def ode_exponential_decay(x,y): # differential equation
    return -x

def exponential_decay(x):   # analytical solution
    return np.exp(-x)

yn, xn = rk4(np.array([1]),0,ode_exponential_decay,0.01,1000)

fig, ax = plt.subplots()
x = np.linspace(-1,11,1001)
ax.plot(xn, yn, 'b.', markersize=1, label=r'4th Order Runge-Kutta')
ax.plot(x, exponential_decay(x), color='red', ls='-', lw=1,
        label=r'Analytical Solution')
ax.set_title('Numerical Solution of the Exponential Decay\n'
            +r'$\dot{x}=-rx$ with $r=1$ and $x(0)=x(t_0)=1$')
ax.set_xlabel(r'time $t$'); ax.set_ylabel(r'$x(t)$')
ax.grid(); ax.legend(loc='best'); ax.set_yscale('log')
ax.set_xlim((-1,11))
fig.savefig('figures/RK4_Exponential-Decay.pdf', format='pdf')
plt.show(); plt.clf(); plt.close()
