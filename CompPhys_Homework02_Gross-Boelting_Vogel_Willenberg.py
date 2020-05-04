# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 02:  Numerical Simulation of the Two-Body Problem
                and Error Analysis of the Euler Scheme
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np
import matplotlib.pyplot as plt


class Body:
    def __init__(self,mass,position,velocity):
        self.mass = mass            # mass of the body
        self.position = position    # inital position vector
        self.velocity = velocity    # inital velocity vector


def two_body_problem(body1,body2,G,R0,dt,N):
    ''' Numerical Simulation of the Two-Body Problem
        Input:  gravitational constant G
                length scale R0
                time steps dt
                number of time steps N '''
    M = body1.mass+body2.mass   # total mass
    V0 = (G*M/R0)**0.5          # velocity scale
    T0 = (R0**3/(G*M))**0.5     # time scale
    h = dt/(R0**3/(G*M))**1.5

    s = np.zeros((int(N),3)); s[0] = (body1.position-body2.position)/R0
    w = np.zeros((int(N),3)); w[0] = (body1.velocity-body2.velocity)/V0

    for i in range(1,int(N)):
        s[i] = s[i-1]+w[i-1]*h
        w[i] = w[i-1]-(s[i-1]/np.linalg.norm(s[i-1])**3)*h

    return (s,w)


v0 = np.sqrt(1*2/1)
body1 = Body(1, np.array([1.0,0.0,0.0]), np.array([0.0,v0,0.0]))
body2 = Body(1, np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]))
s, w = two_body_problem(body1, body2, 1, 1, 1e-3, 1e4)



# Plot the orbits
fig, ax = plt.subplots()
ax.plot(s[:,0], s[:,1], 'b.', label='Body 1')
ax.plot([0],[0], 'rx', label='Body 2')
ax.set_title(r'Numerical Simulation of the 2-Body Problem (Euler Method)')
ax.set_xlabel(r'$x$-axis'); ax.set_ylabel(r'$y$-axis')
ax.grid(); ax.legend(loc='best'); ax.axis('equal')
fig.savefig('figures/Two-Body-Problem.pdf', format='pdf')
plt.show(); plt.clf(); plt.close()
