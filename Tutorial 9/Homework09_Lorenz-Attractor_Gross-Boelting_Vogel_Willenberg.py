# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 09:  The Lorenz Attractor
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np; import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

def rk4_step(x0, y0, z0, fx, fy, fz, h, fx_args={}, fy_args={}, fz_args={}):
    ''' Simple python implementation for one RK4 step.
        Inputs:
            x0      - M x 1 numpy array specifying all variables of the first ODE at the current time step
            y0      - M x 1 numpy array specifying all variables of the second ODE at the current time step
            z0      - M x 1 numpy array specifying all variables of the third ODE at the current time step
            f       - function that calculates the derivates of all variables of the ODE
            h       - step size
            fx_args - Dictionary of additional arguments to be passed to the function f1
            fy_args - Dictionary of additional arguments to be passed to the function f2
            fz_args - Dictionary of additional arguments to be passed to the function f3
        Output:
            xp1 - M x 1 numpy array of variables of the first ODE at time step x0 + h
            yp1 - M x 1 numpy array of variables of the second ODE at time step x0 + h
            zp1 - M x 1 numpy array of variables of the third ODE at time step x0 + h
            tp1 - time step t0+h
    '''
    k1_x = h*fx(x0, y0, z0, **fx_args)
    k1_y = h*fy(x0, y0, z0, **fy_args)
    k1_z = h*fz(x0, y0, z0, **fz_args)

    k2_x = h*fx(x0+k1_x/2., y0+k1_x/2., z0+k1_x/2., **fx_args)
    k2_y = h*fy(x0+k1_y/2., y0+k1_y/2., z0+k1_y/2., **fy_args)
    k2_z = h*fz(x0+k1_z/2., y0+k1_z/2., z0+k1_z/2., **fz_args)

    k3_x = h*fx(x0+k2_x/2., y0+k2_x/2., z0+k2_x/2, **fx_args)
    k3_y = h*fy(x0+k2_y/2., y0+k2_y/2., z0+k2_y/2, **fy_args)
    k3_z = h*fz(x0+k2_z/2., y0+k2_z/2., z0+k2_z/2, **fz_args)

    k4 = h*fx(x0+k3_x/2., y0+k3_x/2., z0+k3_x/2, **fx_args)
    l4 = h*fy(x0+k3_y/2., y0+k3_y/2., z0+k3_y/2, **fy_args)
    m4 = h*fz(x0+k3_z/2., y0+k3_z/2., z0+k3_z/2, **fz_args)

    xp1 = x0+1./6.*(k1_x+2.*k2_x+2.*k3_x+k4)
    yp1 = y0+1./6.*(k1_y+2.*k2_y+2.*k3_y+k4)
    zp1 = z0+1./6.*(k1_z+2.*k2_z+2.*k3_z+k4)
    return (xp1,yp1,zp1)

def rk4(x0, y0, z0, fx, fy, fz, h, n, fx_args={}, fy_args={}, fz_args={}):
    ''' Simple implementation of RK4
        Inputs:
            x0      - M x 1 numpy array specifying all variables of the ODE at the current time step
            y0      - M x 1 numpy array specifying all variables of the ODE at the current time step
            z0      - M x 1 numpy array specifying all variables of the ODE at the current time step
            fx      - function that calculates the derivates of all variables of the first ODE
            fy      - function that calculates the derivates of all variables of the second ODE
            fz      - function that calculates the derivates of all variables of the  third ODE
            h       - step size
            n       - number of steps
            fx_args - Dictionary of additional arguments to be passed to the function f1
            fy_args - Dictionary of additional arguments to be passed to the function f2
            fz_args - Dictionary of additional arguments to be passed to the function f3
        Output:
            yn - N+1 x M numpy array with the results of the integration for every time step (includes y0)
            xn - N+1 x 1 numpy array with the time step value (includes start x0)
    '''
    xn = np.zeros(n+1); yn = np.zeros(n+1); zn = np.zeros(n+1)
    xn[0] = x0; yn[0] = y0; zn[0] = z0

    for n in np.arange(1,n+1,1):
        xn[n], yn[n], zn[n] = rk4_step(x0=xn[n-1], y0=yn[n-1], z0=zn[n-1],
                                       fx=fx, fy=fy, fz=fz, h=h, fx_args=fx_args,
                                       fy_args=fy_args, fz_args=fz_args)
    return (xn,yn,zn)


# set initial condition
sigma = 10
b = 8/3
r = np.array([1.17,1.3456,25.0,29.0])
a0 = np.sqrt(b*(r-1))

# Lorenz System
def dx(x,y,z,sigma):
    return -sigma*(x-y)
def dy(x,y,z,r):
    return r*x-y-x*z
def dz(x,y,z,b):
    return x*y-b*z


for i in range(0,len(r)-2):
    xn, yn, zn = rk4(x0=1., y0=1., z0=1., fx=dx, fy=dy, fz=dz, h=0.01, n=10000,
                     fx_args={'sigma':sigma}, fy_args={'r':r[i]}, fz_args={'b':b})

    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xn,yn,zn, s=1, c='blue')
    ax.set_title('Title')
    ax.set_xlabel(r'$x$-axis')
    ax.set_ylabel(r'$y$-axis')
    ax.set_zlabel(r'$z$-axis')
    #ax.set_xlim((0,5)); ax.set_ylim((-7.5,12.5))
    #ax.grid(); ax.legend(loc='best', markerscale=8)
    fig.savefig('figures/Plot-{}.pdf'.format(i), format='pdf')

plt.show(); plt.clf(); plt.close()
