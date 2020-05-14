# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 03:  Numerical Simulation of the Three-Body Problem
                using the 4th Order Runge-Kutta Method
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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


def three_body_problem(y,x,G,m1,m2,m3):
    ''' Twelve coupled ordinary differential equations of first order
        (converted into the standard form) '''
    yn = np.ones(12)
    x1 = y[0];  y1 = y[1]
    x2 = y[4];  y2 = y[5]
    x3 = y[8];  y3 = y[9]

    r12 = np.sqrt((x1-x2)**2+(y1-y2)**2) # distance between body 1 and body 2
    r13 = np.sqrt((x1-x3)**2+(y1-y3)**2) # distance between body 1 and body 3
    r23 = np.sqrt((x2-x3)**2+(y2-y3)**2) # distance between body 2 and body 3

    yn[0] = y[2]    # differential equations for body 1
    yn[1] = y[3]
    yn[2] = (-m2*G/r12**3)*(x1-x2)+(-m3*G/r13**3)*(x1-x3)
    yn[3] = (-m2*G/r12**3)*(y1-y2)+(-m3*G/r13**3)*(y1-y3)

    yn[4] = y[6]    # differential equations for body 2
    yn[5] = y[7]
    yn[6] = (-m1*G/r12**3)*(x2-x1)+(-m3*G/r23**3)*(x2-x3)
    yn[7] = (-m1*G/r12**3)*(y2-y1)+(-m3*G/r23**3)*(y2-y3)

    yn[8] = y[10]   # differential equations for body 3
    yn[9] = y[11]
    yn[10] = (-m1*G/r13**3)*(x3-x1)+(-m2*G/r23**3)*(x3-x2)
    yn[11] = (-m1*G/r13**3)*(y3-y1)+(-m2*G/r23**3)*(y3-y2)

    return yn


class Body:
    def __init__(self,mass,position,velocity=np.array([0.,0.])):
        self.mass = mass            # mass of the body
        self.position = position    # inital position vector
        self.velocity = velocity    # inital velocity vector

def initial_conditions(body1,body2,body3):
    ''' Function to write the inital conditions into an numpy array '''
    return np.array([body1.position[0],body1.position[1],
                     body1.velocity[0],body1.velocity[1],
                     body2.position[0],body2.position[1],
                     body2.velocity[0],body2.velocity[1],
                     body3.position[0],body3.position[1],
                     body3.velocity[0],body3.velocity[1]])


G = 1.  # simplify the system by setting the gravitational constant to G=1
# create the three bodies with their initial conditions (Meissel-Burrau Problem)
body1 = Body(3., np.array([1.,3.]))
body2 = Body(4., np.array([-2.,-1.]))
body3 = Body(5., np.array([1.,-1.]))


# numerical simulation of the gravitational three-body problem
# using the Runge-Kutta-4 integrator
yn, xn = rk4(initial_conditions(body1,body2,body3),0,three_body_problem,2e-5,int(8*1e5),
             {'G':G, 'm1':body1.mass, 'm2':body2.mass, 'm3':body3.mass})

# plot the trajectories of the three bodies
fig, ax = plt.subplots()

ax.set_title('Numerical Simulation of the Gravitational Three-Body Problem (Meissel-Burrau Problem)')
ax.set_xlabel(r'$x$-coordinates'); ax.set_ylabel(r'$y$-coordinates')

line1, = ax.plot([], [], 'r.', ms=30, label='Body 1')
line2, = ax.plot([], [], 'g.', ms=30, label='Body 2')
line3, = ax.plot([], [], 'b.', ms=30, label='Body 3')

ax.legend(loc='upper right', markerscale=0.6)
ax.set_xlim((-6,4)); ax.set_ylim((-20,20))

def trajectories(i):
    index = i*300
    line1.set_data(yn[index-1:index,0], yn[index-1:index,1])
    line2.set_data(yn[index-1:index,4], yn[index-1:index,5])
    line3.set_data(yn[index-1:index,8], yn[index-1:index,9])
    return (line1, line2, line3,)

animate = animation.FuncAnimation(fig, trajectories, frames=int(8*1e5/300),
                                  interval=1, repeat=True, blit=True)
plt.show(); plt.clf(); plt.close()
