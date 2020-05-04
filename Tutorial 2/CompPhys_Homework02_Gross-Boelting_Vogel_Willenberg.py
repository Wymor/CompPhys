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


def total_energy(s,w):
    energy = np.zeros(np.shape(s)[0])
    for i in range(0,np.shape(s)[0]):
        energy[i] = (0.5*np.linalg.norm(w[i])**2)-(1/np.linalg.norm(s[i]))
    return energy

def angular_momentum(s,w):
    angular_momentum = np.zeros((np.shape(s)[0],3))
    for i in range(0,np.shape(s)[0]):
        angular_momentum[i] = np.dot(s[i],w[i])
    return angular_momentum

def laplace_runge_lenz(s,w):
    LRL = np.zeros((np.shape(s)[0],3))
    for i in range(0,np.shape(s)[0]):
        LRL[i] = np.dot(w[i], np.dot(s[i],w[i]))-s[i]
    return LRL

def eccentricity(LRL):
    return np.linalg.norm(LRL, axis=1)

def relative_error(energy):
    rel_error = np.zeros(np.shape(s)[0])
    for i in range(0,np.shape(s)[0]):
        rel_error[i] = abs(energy[i]-energy[0])/abs(energy[0])
    return rel_error

def two_body_problem(body1,body2,G,R0,dt,N):
    ''' Numerical Simulation of the Two-Body Problem
        Input:  gravitational constant G
                length scale R0
                time steps dt
                number of time steps N '''
    M = body1.mass+body2.mass   # total mass
    V0 = (G*M/R0)**0.5          # velocity scale
    T0 = (R0**3./(G*M))**0.5     # time scale
    h = dt/(R0**3./(G*M))**1.5

    s = np.zeros((int(N),3)); s[0] = (body1.position-body2.position)/R0
    w = np.zeros((int(N),3)); w[0] = (body1.velocity-body2.velocity)/V0

    for i in range(1,int(N)):
        s[i] = s[i-1]+(w[i-1]*h)
        w[i] = w[i-1]-(s[i-1]/np.linalg.norm(s[i-1])**3.)*h

    return (s,w)

v0 = 0.5*np.sqrt(1.*1./1.)
body1 = Body(1., np.array([1.,0.,0.]), np.array([0.,v0,0.]))
body2 = Body(1., np.array([0.,0.,0.]), np.array([0.,0.,0.]))
s, w = two_body_problem(body1, body2, 1., 1., 1e-3, 1e4)

energy = total_energy(s,w)
angular_momentum = angular_momentum(s,w)
LRL = laplace_runge_lenz(s,w)
eccentricity = eccentricity(LRL)
rel_error = relative_error(energy)

# Plot the orbits
fig, ax = plt.subplots(2,2)
fig.suptitle(r'Forward Euler Method: Numerical Simulation of the Two-Body Problem')

ax[0,0].plot(s[:,0], s[:,1], 'b.', label='Body 1')
ax[0,0].plot([0], [0], 'rx', label='Body 2')
ax[0,0].set_title(r'Orbits')
#ax[0,0].set_xlabel(r'$x$-axis'); ax.set_ylabel(r'$y$-axis')
#ax[0,0].grid(); ax[0,0].legend(loc='best'); ax[0,0].axis('equal')

ax[0,1].plot(range(0,len(eccentricity)), eccentricity, 'b.', label='Eccentricity')
ax[0,1].set_title(r'Eccentricity')
#ax[0,1].set_xlabel(r'Step $i$'); ax.set_ylabel(r'Eccentricity')
#ax[0,1].grid(); ax[0,1].legend(loc='best')

ax[1,0].plot(range(0,len(energy)), energy, 'b.', label='Total Energy')
ax[1,0].set_title(r'Total Energy')
#ax[1,0].set_xlabel(r'Step $i$'); ax.set_ylabel(r'Total Energy $E_i$')
#ax[1,0].grid(); ax[1,0].legend(loc='best')

ax[1,1].plot(range(0,len(rel_error)), rel_error, 'b.', label='Relative Error')
ax[1,1].set_title(r'Relative Error in the Total Energy')
#ax[1,1].set_xlabel(r'Step $i$'); ax.set_ylabel(r'Relative Error')
#ax[1,1].grid(); ax[1,1].legend(loc='best')

fig.savefig('figures/Two-Body-Problem.pdf', format='pdf')
plt.show(); plt.clf(); plt.close()
