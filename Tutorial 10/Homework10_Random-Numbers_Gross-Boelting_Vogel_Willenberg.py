# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 10:  Determination of Pi with Random Numbers
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np; import matplotlib.pyplot as plt; import scipy.constants as const

def quadrant_function(x):
    ''' Function that describes the unit quarter circle (quadrant) '''
    return np.sqrt(1-x**2)

def pi_Monte_Carlo(N,quadrant):
    ''' Function to determine pi using Monte Carlo integration '''
    random_x = np.random.rand(1,N) # random numbers (random sampling)
    return (4./N)*np.sum(quadrant(random_x))

def pi_rejection_method(N,quadrant):
    ''' Function to determine pi using a rejection method '''

    # create random coordinates within the square
    random_x = np.random.rand(1,N); random_y = np.random.rand(1,N)

    # check whether the coordinate fell into the quadrant
    x_take = random_x[random_y < quadrant(random_x)]
    y_take = random_y[random_y < quadrant(random_x)]

    # check whether the coordinate fell not into the quadrant
    x_reject = random_x[random_y > quadrant(random_x)]
    y_reject = random_y[random_y > quadrant(random_x)]

    return (4.*len(x_take)/N, x_take, y_take, x_reject, y_reject)


## Determination of Pi using Monte Carlo Integration
N = int(6e7) # number of randomly selected points
print('Monte Carlo Integration: {}'.format(pi_Monte_Carlo(N,quadrant_function)))

## Determination of Pi using a Rejection Method
N = int(3e3) # number of randomly selected points
out = pi_rejection_method(N,quadrant_function)

fig, ax = plt.subplots() # plot to illustrate the rejection method
ax.set_title(r'Determination of $\pi$ using a Rejection Method'+'\n'
            +r'($N={}$ random numbers and $\pi\approx{}$)'.format(N,round(out[0],4)))
ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')

x = np.linspace(0,1,1000)
ax.plot(x,quadrant_function(x), 'b-', linewidth=1.5, label=r'$y=\sqrt{1-x^2}$')
ax.plot(out[1], out[2], 'g.', markersize=1.5, label=r'${}$ points'.format(len(out[1])))
ax.plot(out[3], out[4], 'r.', markersize=1.5, label=r'${}$ points'.format(len(out[3])))

ax.grid(); ax.legend(loc='lower left', markerscale=8)
ax.set(xlim=(-0.05,1.05), ylim=(-0.05,1.05)); ax.set_aspect('equal', 'box')
fig.savefig('figures/RN-Rejection-Method_Pi-Determination.pdf', format='pdf')

# accuracy of the result as a function of the number of randomly selected points
N = np.linspace(10,int(1e4),1000)   # number of randomly selected points
rel_error = np.zeros(len(N))        # array for the accuracy of the result

for i in range(0,len(N)):
    N[i] = int(N[i]); pi = pi_rejection_method(int(N[i]),quadrant_function)[0]
    rel_error[i] = abs(pi-const.pi)/abs(const.pi)

fig, ax = plt.subplots()
ax.set_title(r'Rejection Method: Accuracy of the Determination of $\pi$')
ax.set_xlabel(r'number $N$ of randomly selected points')
ax.set_ylabel(r'relative error [$\%$]')
ax.plot(N, 100*rel_error, 'b.', markersize=1.5)
ax.grid(); ax.set_xscale('log'); ax.set_yscale('log')
#ax.set(xlim=(-0.05,1.05), ylim=(-0.05,1.05))
fig.savefig('figures/RN-Rejection-Method_Accuracy-Analysis.pdf', format='pdf')
plt.show(); plt.clf(); plt.close()
