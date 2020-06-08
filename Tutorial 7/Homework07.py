# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 07:  Numerov Algorithm for the Schrödinger Equation
                Neutrons in the Gravitational Field of the Earth
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np; import matplotlib.pyplot as plt
import scipy.constants as const

def y(n):
    return 1/(1+n**-2)

def x(n):
    D = .5
    return D*n*(1.-(1./7.3)*n)

n = np.linspace(-10,10,1000)

fig, ax = plt.subplots()
ax.set_ylabel(r'wavefunction $\psi_n(x)$')
ax.plot(n,y(n), 'b.', markersize=1, label='1')
ax.plot(n,x(n), 'r.', markersize=1, label='2')
ax.set_title('Quantum states and energy eigenvalues\n'+
             'of neutrons in the Earth\'s gravitational field')
ax.set_xlabel(r'dimensionless height $x$')
#ax.grid(); ax.legend(loc='best', markerscale=8)
#ax.set_xlim((int(round(min(x))),int(round(max(x)))))
#fig2.savefig('figures/Eigenvalues_Probability-Amplitude.pdf', format='pdf')
plt.show(); plt.clf(); plt.close()
