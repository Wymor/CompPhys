# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 07:  Population Dynamics - Stationary Points
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np; import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq

def cubic_function(n,coeff):
    a,b,c,d = coeff[0], coeff[1], coeff[2], coeff[3]
    return (a*n**3)+(b*n**2)+(c*n)+d

def pq(arr):
    p = arr[1]/arr[0]; q = arr[2]/arr[0]
    if ((p/2)**2 >= q):
        n2 = -(p/2)+np.sqrt((p/2)**2-q)
        n3 = -(p/2)-np.sqrt((p/2)**2-q)
        return(n2,n3)

def count_roots(arr):
    p = arr[1]/arr[0]; q = arr[2]/arr[0]
    if ((p/2)**2 < q):
        return 1
    elif ((p/2)**2 == q):
        return 2
    else: return 3

D = 0.1
coeff = np.array([1.,-7.3,1+(7.3/D),-7.3])

# compute first root
n1 = brentq(cubic_function,-2.,8.,args=coeff)

# polynomial division
divisor = np.array([1.,-n1])
div = np.polydiv(coeff, divisor)

while count_roots(div[0]) == 1:
    D += 1e-3
    coeff = np.array([1.,-7.3,1+(7.3/D),-7.3])
    # compute first root
    n1 = brentq(cubic_function,-2.,8.,args=coeff)
    # polynomial division
    divisor = np.array([1.,-n1])
    div = np.polydiv(coeff, divisor)

print(D)
print(n1,pq(div[0]))

n = np.linspace(-5,8,1000)
fig, ax = plt.subplots()
#ax.set_ylabel(r'wavefunction $\psi_n(x)$')
ax.plot(n,cubic_function(n,coeff), 'b.', markersize=1, label='1')
ax.plot(n,np.zeros(len(n)), 'r.', markersize=1, label='1')
#ax.set_title('Quantum states and energy eigenvalues\n'+
#             'of neutrons in the Earth\'s gravitational field')
#ax.set_xlabel(r'dimensionless height $x$')
#ax.grid(); ax.legend(loc='best', markerscale=8)
#ax.set_xlim((int(round(min(x))),int(round(max(x)))))
#fig2.savefig('figures/Eigenvalues_Probability-Amplitude.pdf', format='pdf')
plt.show(); plt.clf(); plt.close()
