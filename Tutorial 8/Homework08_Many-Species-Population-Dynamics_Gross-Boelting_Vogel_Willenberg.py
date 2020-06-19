# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 08:  Many Species Population Dynamics
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np; import matplotlib.pyplot as plt
import numpy.linalg as linalg

def initial_state(coeff,ev):
    ''' Function to set the initial state for given coefficients
        Input:  coefficients (coeff), eigenvalues (ew) and eigenvectors (ev)'''
    n = np.zeros(len(ev[:,0]))
    for i in range(0,len(n)): n = n+coeff[i]*ev[:,i]

    # if all imaginary part are zero, then only use the real parts
    complex = np.iscomplex(n)
    if complex.all() == False: n = np.real(n)
    return n

def time_evolution(t,coeff,ew,ev):
    ''' Function to compute the time-dependent evolution of the population
        Input:  time (t), coefficients (coeff), eigenvalues (ew)
                and eigenvectors (ev) '''
    evolution = np.zeros((len(t),len(ew)))
    for i in range(0,len(t)):
        for j in range(0,len(ew)):
            evolution[i,:] = evolution[i,:]+coeff[j]*np.exp(ew[j]*t[i])*ev[:,j]
    return evolution

# set the Jacobi matrix A at the non-trivial fixed point
A = np.array([[-1,0,0,-20,-30,-5],[0,-1,0,-1,-3,-7],[0,0,-1,-4,-10,-20],
              [20,30,35,0,0,0],[3,3,3,0,0,0],[7,8,20,0,0,0]])

# determine the eigenvalues (ew) and eigenvectors (ev) of A
eigenvalues, eigenvectors = linalg.eig(A)

# print the results (eigenvalues and eigenvectors)
for i in range(0,len(eigenvalues)):
    print('Eigenvalue:  {}'.format(eigenvalues[i]))
    print('Eigenvector: {}\n'.format(eigenvectors[:,i]))

# set the initial state with the given coefficients
c = np.array([3,3,1,1,-5,0.1])
print('Initial state: {}'.format(initial_state(c,eigenvectors)))

# compute and plot the time-dependent evolution of the six populations
t = np.linspace(0,40,10000) # time
time_evolution = time_evolution(t,c,eigenvalues,eigenvectors)

#
predator = time_evolution[:,3]+time_evolution[:,4]+time_evolution[:,5]
prey = time_evolution[:,0]+time_evolution[:,1]+time_evolution[:,2]

fig1, ax1 = plt.subplots()
ax1.plot(t,predator, 'b-', linewidth=1, label=r'Predator $P_1+P_2+P_3$')
ax1.plot(t,prey, 'r-', linewidth=1, label=r'Prey $N_1+N_2+N_3$')
ax1.set_xlim((0,40)); ax1.set_ylim((-6,8.5))

fig2, ax2 = plt.subplots()
ax2.plot(t,predator, 'b-', linewidth=1, label=r'Predator $P_1+P_2+P_3$')
ax2.plot(t,prey, 'r-', linewidth=1, label=r'Prey $N_1+N_2+N_3$')
ax2.set_xlim((0,10)); ax2.set_ylim((-6,8.5))

fig3, ax3 = plt.subplots()
ax3.plot(t,predator, 'b-', linewidth=1, label=r'Predator $P_1+P_2+P_3$')
ax3.plot(t,prey, 'r-', linewidth=1, label=r'Prey $N_1+N_2+N_3$')
ax3.set_xlim((4.5,5.5)); ax3.set_ylim((-1,1))

for ax in [ax1,ax2,ax3]:
    ax.set_title('Many Species Population Dynamics')
    ax.set_xlabel(r'time $t$'); ax.set_ylabel(r'population number')
    ax.grid(); ax.legend(loc='upper right')
fig1.savefig('figures/Population-Time-Evolution-01.pdf', format='pdf')
fig2.savefig('figures/Population-Time-Evolution-02.pdf', format='pdf')
fig3.savefig('figures/Population-Time-Evolution-02.pdf', format='pdf')
plt.show(); plt.clf(); plt.close()
