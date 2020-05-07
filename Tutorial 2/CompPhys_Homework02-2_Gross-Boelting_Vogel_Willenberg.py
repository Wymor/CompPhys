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
    ''' Function to compute the total energy of the system from the
        dimensionless vectors s (location) and w (velocity) '''
    energy = np.zeros(np.shape(s)[0])
    for i in range(0,np.shape(s)[0]):
        energy[i] = (0.5*np.linalg.norm(w[i])**2)-(1/np.linalg.norm(s[i]))
    return energy

def relative_error(energy):
    ''' Function to compute the relative error in the total energy at
        step i compared to the initial value 0 '''
    rel_error = np.zeros(np.shape(s)[0])
    for i in range(0,np.shape(s)[0]):
        rel_error[i] = abs(energy[i]-energy[0])/abs(energy[0])
    return rel_error

def forward_euler(body1,body2,G,dt,N):
    ''' Numerical Simulation of the Two-Body Problem: This function computes
        the relative motion of two point-like bodies under their mutual
        gravitational influence using a forward Euler integration procedure
        Input:  two bodies of the class "Body", gravitational constant G,
                time steps dt and number of time steps N
        Output: location vector s and velocity vector w '''

    # compute the total mass of the two bodies and set the
    # characteristic length scale as the inital seperation
    M = body1.mass+body2.mass
    R0 = np.linalg.norm(body1.position-body2.position)
    h = dt/np.sqrt(R0**3/(G*M)) # compute the dimensionless step size

    s = np.zeros((int(N),3))    # create array for the location vectors
    s[0] = (body1.position-body2.position)/R0

    w = np.zeros((int(N),3))    # create array for the velocity vectors
    w[0] = (body1.velocity-body2.velocity)/np.sqrt(G*M/R0)

    for i in range(1,int(N)):   # compute the relative motion
        s[i] = s[i-1]+(w[i-1]*dt)
        w[i] = w[i-1]-(s[i-1]/np.linalg.norm(s[i-1])**3.)*dt

    return (s,w)


v0 = [np.sqrt(.5*1.*2./1.), np.sqrt(1.*2./1.)]
dt = np.linspace(1e-4, 1e-2, int(1e2))
rel_error = np.zeros((len(v0),len(dt)))

for i in range(0,len(v0)):
    for j in range(0,len(dt)):
        N = int(2*np.sqrt(1.**3/(1.*2.))/dt[j])
        body1 = Body(1., np.array([1.,0.,0.]), np.array([0.,v0[i],0.]))
        body2 = Body(1., np.array([0.,0.,0.]), np.array([0.,0.,0.]))
        s, w = forward_euler(body1, body2, 1., dt[j], N)
        rel_error[i,j] = relative_error(total_energy(s,w))[N-1]


# Plot the results
fig, ax = plt.subplots()
ax.plot(dt, rel_error[0,:], 'b.',
        label=r'initial velocity $v_0=\sqrt{0,5\cdot GM/r}$')
ax.plot(dt, rel_error[1,:], 'r.',
        label=r'initial velocity $v_0=\sqrt{GM/r}$')
ax.set_title(r'Error Analysis of Euler Scheme')
ax.set_xlabel(r'step size $\mathrm{d}t$'); ax.set_ylabel(r'relative error $\epsilon$')
ax.grid(); ax.legend(loc='best')
#ax.set_xscale('log'); ax.set_yscale('log')
fig.savefig('figures/Euler-Scheme_Error-Analysis.pdf', format='pdf')
plt.show(); plt.clf(); plt.close()
