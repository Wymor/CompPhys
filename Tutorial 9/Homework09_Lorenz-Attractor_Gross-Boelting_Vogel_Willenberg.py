# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 09:  The Lorenz Attractor
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np; import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rk4_step(x0, y0, z0, fx, fy, fz, h, fx_args={}, fy_args={}, fz_args={}):
    ''' Simple python implementation for one RK4 step.
        Inputs: (i=x,y,z)
            i0 - M x 1 numpy arrays specifying all variables of the three
                 ODE's at the current time step
            f  - function that calculates the derivates of all variables of the ODE
            h  - step size
            fi_args - dictionary of additional arguments to be passed to the function fi
        Output: (i=x,y,z)
            ip1 - M x 1 numpy array of variables of the first ODE at time step i0+h '''
    k1_x = h*fx(x0, y0, z0, **fx_args)
    k1_y = h*fy(x0, y0, z0, **fy_args)
    k1_z = h*fz(x0, y0, z0, **fz_args)

    k2_x = h*fx(x0+k1_x/2., y0+k1_x/2., z0+k1_x/2., **fx_args)
    k2_y = h*fy(x0+k1_y/2., y0+k1_y/2., z0+k1_y/2., **fy_args)
    k2_z = h*fz(x0+k1_z/2., y0+k1_z/2., z0+k1_z/2., **fz_args)

    k3_x = h*fx(x0+k2_x/2., y0+k2_x/2., z0+k2_x/2, **fx_args)
    k3_y = h*fy(x0+k2_y/2., y0+k2_y/2., z0+k2_y/2, **fy_args)
    k3_z = h*fz(x0+k2_z/2., y0+k2_z/2., z0+k2_z/2, **fz_args)

    k4_x = h*fx(x0+k3_x/2., y0+k3_x/2., z0+k3_x/2, **fx_args)
    k4_y = h*fy(x0+k3_y/2., y0+k3_y/2., z0+k3_y/2, **fy_args)
    k4_z = h*fz(x0+k3_z/2., y0+k3_z/2., z0+k3_z/2, **fz_args)

    xp1 = x0+1./6.*(k1_x+2.*k2_x+2.*k3_x+k4_x)
    yp1 = y0+1./6.*(k1_y+2.*k2_y+2.*k3_y+k4_y)
    zp1 = z0+1./6.*(k1_z+2.*k2_z+2.*k3_z+k4_z)
    return (xp1,yp1,zp1)

def rk4(x0, y0, z0, fx, fy, fz, h, n, fx_args={}, fy_args={}, fz_args={}):
    ''' Simple implementation of RK4
        Inputs: (i=x,y,z)
            i0 - M x 1 numpy arrays specifying all variables of the three
                 ODE's at the current time step
            f  - function that calculates the derivates of all variables of the ODE
            h  - step size
            fi_args - dictionary of additional arguments to be passed to the function fi
        Output: (i=x,y,z)
            in - N+1 x M numpy array with the results of the integration for
                 every time step (includes i0) '''
    xn = np.zeros(n+1); yn = np.zeros(n+1); zn = np.zeros(n+1)
    xn[0] = x0; yn[0] = y0; zn[0] = z0

    for n in np.arange(1,n+1,1):
        xn[n], yn[n], zn[n] = rk4_step(x0=xn[n-1], y0=yn[n-1], z0=zn[n-1],
                                       fx=fx, fy=fy, fz=fz, h=h, fx_args=fx_args,
                                       fy_args=fy_args, fz_args=fz_args)
    return (xn,yn,zn)


# set initial conditions
sigma = 10
b = 8/3
r = np.array([0.5,1.17,1.3456,25.0,29.0])
a0 = np.sqrt(b*(r[r>1.]-1))

# Lorenz system: coupled set of equations
def dx(x,y,z,sigma):
    return -sigma*(x-y)
def dy(x,y,z,r):
    return r*x-y-x*z
def dz(x,y,z,b):
    return x*y-b*z


for i in range(0,len(r)):
    # plot the solution (3-D plot)
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ax.set_title(r'Lorenz Attractor for $r=${}'.format(r[i]))

    # choose the initial conditions near one of the fixed points
    if (r[i]<1): x0 = 1. ; y0 = 1. ; z0 = 1.
    else: x0 = a0[i-1]+1.; y0 = a0[i-1]+1. ; z0 = (r[i]-1.)+1.

    # choose number of steps and step size
    if (r[i]==25 or r[i]==29): N = int(1e4); h = 1e-2
    else: N = int(1e4); h = 1e-3

    # solve numerically the above coupled set of equations (using Runge-Kutta-4)
    xn, yn, zn = rk4(x0=x0, y0=y0, z0=z0, fx=dx, fy=dy, fz=dz, h=h, n=N,
                     fx_args={'sigma':sigma}, fy_args={'r':r[i]}, fz_args={'b':b})

    ax.scatter(xn,yn,zn, marker='.', s=1, c='blue')
    ax.scatter([x0],[y0],[z0], marker='x', s=30, c='green')
    ax.scatter([x0-1.],[y0-1.],[z0-1.], marker='x', s=30, c='red')
    if (r[i]==25 or r[i]==29):
         ax.scatter([-a0[i-1]],[-a0[i-1]],[r[i]-1.], marker='x', s=30, c='red')
    ax.set_xlabel(r'$x$-axis'); ax.set_ylabel(r'$y$-axis'); ax.set_zlabel(r'$z$-axis')
    fig.savefig('figures/3D-Plot_Lorenz-Attractor-{}.pdf'.format(i+1), format='pdf')

plt.show(); plt.clf(); plt.close()
