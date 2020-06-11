# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 07:  Population Dynamics - Stationary Points
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np; import matplotlib.pyplot as plt
from scipy.optimize import brentq

def cubic_function(n,coefficients):
    ''' Cubic function where n represents an unknown (variable)
        and a, b, c and d represent known numbers (coefficients) '''
    a,b,c,d = coefficients[0],coefficients[1],coefficients[2],coefficients[3]
    return (a*n**3)+(b*n**2)+(c*n)+d

def quadratic_formula(coefficients):
    ''' Function to compute the solutions of a reduced quadratic equation
        using the quadratic formula '''
    p = coefficients[1]/coefficients[0]
    q = coefficients[2]/coefficients[0]
    if ((p/2)**2 >= q): # as stationary points only real solutions are valid
        n2 = -(p/2)+np.sqrt((p/2)**2-q)
        n3 = -(p/2)-np.sqrt((p/2)**2-q)
        return (n2,n3)
    else: return None

def count_zeros(coefficients):
    ''' Function to compute the number of real zeros of a quadratic equation
        Input:  NumPy-Array with the coefficients of the quadratic equation
        Output: number of real zeros of the given quadratic equation '''
    p = coefficients[1]/coefficients[0]
    q = coefficients[2]/coefficients[0]
    if ((p/2)**2 < q):
        # in this case the square root becomes imaginary and
        # the quadratic equation has no real zero points
        return 0
    elif ((p/2)**2 == q):
        # in this case the square root becomes zero and
        # the quadratic equation has a real 'double zero point'
        return 1
    else: return 2

def find_solutions(D):
    ''' Function to answer the question: When do one or three real solutions
        exist as a function of the remaining free parameter?
        We vary the free parameter D and check for which value of D we
        get one or three real solutions
        Input:  starting value for the free paramter D '''

    # We use the coefficients of the cubic equation, which
    # we have already derived analytically
    coefficients = np.array([1.,-7.3,1+(7.3/D),-7.3])

    # compute first zero point of the cubic equation numerically
    n1 = brentq(cubic_function, -2., 8., args=coefficients)

    # compute the quadratic function using polynomial division
    quadratic_function = np.polydiv(coefficients, np.array([1.,-n1]))[0]

    D_lower = D; D_upper = D
    while count_zeros(quadratic_function) != 2:
        D += 1e-3   # increase free paramter
        coefficients = np.array([1.,-7.3,1+(7.3/D),-7.3])

        # compute first zero point of the cubic equation numerically
        n1 = brentq(cubic_function,-2.,8.,args=coefficients)

        # compute the quadratic function using polynomial division
        quadratic_function = np.polydiv(coefficients, np.array([1.,-n1]))[0]
        D_lower = D

    while count_zeros(quadratic_function) == 2:
        D_upper = D; D += 1e-3   # increase free paramter
        coefficients = np.array([1.,-7.3,1+(7.3/D),-7.3])

        # compute first zero point of the cubic equation numerically
        n1 = brentq(cubic_function,-2.,8.,args=coefficients)

        # compute the quadratic function using polynomial division
        quadratic_function = np.polydiv(coefficients, np.array([1.,-n1]))[0]

    return (D_lower,D_upper)

# We determine the stationary points for K/A = 7.3
# The stationary points are solutions of a cubic equation; it depends on
# the variable n and the remaining free parameter D
D = find_solutions(0.1)
n = np.linspace(0,5,1000)
print(D)


fig, ax = plt.subplots()
ax.set_title('Titel')
ax.set_xlabel(r'$n$'); ax.set_ylabel(r'$\frac{\mathrm{d}n}{\mathrm{d}\tau}$')

coefficients = np.array([1.,-7.3,1+(7.3/D[0]),-7.3])
ax.plot(n,cubic_function(n,coefficients), 'r.', markersize=1, label='1')
coefficients = np.array([1.,-7.3,1+(7.3/D[1]),-7.3])
ax.plot(n,cubic_function(n,coefficients), 'g.', markersize=1, label='1')
coefficients = np.array([1.,-7.3,1+(7.3/(0.5*(D[0]+D[1]))),-7.3])
ax.plot(n,cubic_function(n,coefficients), 'b.', markersize=1, label='1')

ax.grid(); ax.legend(loc='best', markerscale=8)
ax.set_xlim((0,5)); ax.set_ylim((-7.5,12.5))
#fig.savefig('figures/Eigenvalues_Probability-Amplitude.pdf', format='pdf')
plt.show(); plt.clf(); plt.close()
