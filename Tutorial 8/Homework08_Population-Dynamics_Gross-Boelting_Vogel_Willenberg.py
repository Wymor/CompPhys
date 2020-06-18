# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 08:  Population Dynamics - Stationary Points
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np; import matplotlib.pyplot as plt
import numpy.linalg as linalg

# set matrix A
A = np.array([[-1,0,0,-20,-30,-5],[0,-1,0,-1,-3,-7],[0,0,-1,-4,-10,-20],
              [20,30,35,0,0,0],[3,3,3,0,0,0],[7,8,20,0,0,0]])

# we determine the eigenvalues and eigenvectors of A
eigenvalues,eigenvectors = linalg.eig(A)

# print the results
for i in range(0,len(eigenvalues)):
    print('Eigenvalue:  {}'.format(eigenvalues[i]))
    print('Eigenvector: {}\n'.format(eigenvectors[:,i]))


def initial_state(coefficients,eigenvalues,eigenvectors):
    ''' Function to set the initial state'''
    n = np.zeros(len(eigenvalues))
    for i in range(0,len(n)):
        n = n+coefficients[i]*eigenvectors[:,i]

    # if all imaginary part are zero, then only use the real parts
    complex = np.iscomplex(n)
    if complex.all() == False:
        n = np.real(n)
    return n

def time_evolution(time,coefficients,eigenvalues,eigenvectors):
    evolution = np.zeros((len(time),len(eigenvalues)))
    for i in range(0,len(time)):
        for j in range(0,len(eigenvalues)):
            evolution[i,:] = evolution[i,:]+coefficients[j]*np.exp(eigenvalues[j]*time[i])*eigenvectors[:,j]
    return evolution


# choose initial state
c = np.array([3,3,1,1,-5,0.1])
n = initial_state(c,eigenvalues,eigenvectors)

t = np.linspace(0,6,10000)
time_evolution = time_evolution(t,c,eigenvalues,eigenvectors)


fig, ax = plt.subplots()
color = ['red','green','blue','red','green','blue']
linestyle = ['-','-','-','--','--','--']

for i in range(0,len(eigenvalues)):
    ax.plot(t,time_evolution[:,i], linestyle=linestyle[i], color=color[i], linewidth=1.5)

ax.set_title('Population Dynamics')
ax.set_xlabel(r'time $t$'); ax.set_ylabel(r'population number')
ax.grid(); ax.legend(loc='best', markerscale=8)
ax.set_xlim((0,4)); ax.set_ylim((-3,8))
fig.savefig('figures/Time_Evolution.pdf', format='pdf')
plt.show(); plt.clf(); plt.close()
