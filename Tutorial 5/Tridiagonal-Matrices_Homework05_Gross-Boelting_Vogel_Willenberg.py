# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 05:  Numerical Linear Algebra Methods
                Tridiagonal Matrices and Gaussian Elimination
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np; import matplotlib.pyplot as plt
from scipy.sparse import diags; from copy import deepcopy


def Gaussian_elimination(A,y):
    ''' Numerical subroutine for the iterative expression for
        Gaussian elimination without pivoting '''
    a, b = deepcopy(A), deepcopy(y)
    N = np.shape(a)[0]
    for i in range(N):
        for k in range(i+1,N):
            factor = a[k,i]/a[i,i]
            b[k] -= b[i]*factor
            for j in range(i,N):
                a[k,j] -= a[i,j]*factor
    return (a,b)

def Thomas_algorithm(A,y):
    ''' Numerical subroutine for the Thomas algorithm (a simplified form
        of Gaussian elimination that can be used to solve tridiagonal
        systems of equations) '''
    a, b = deepcopy(A), deepcopy(y)
    N = np.shape(a)[0]
    for i in range(N-1):
        print(i)
        factor = a[i+1,i]/a[i,i]
        a[i+1,i] -= factor*a[i,i]
        a[i+1,i+1] -= factor*a[i,i+1]
        b[i+1] -= factor*b[i]
    return (a,b)

def backward_substitution(A,y):
    ''' Numerical subroutine for the iterative expression for
        backward substitution '''
    N = np.shape(A)[0]
    x = np.zeros(N)

    x[N-1] = y[N-1]/A[N-1,N-1]
    for i in range(N-2,-1,-1):
        x[i] = (y[i]-A[i,i+1]*x[i+1])/A[i,i]
    return x

def solve_tridiagonal_system(a,b,c,y,method):
    ''' Numerical subroutine that finds the solution vector x for a
        tridiagnonal equation system Ax=y '''
    tridiag = diags([a,b,c], [-1,0,1]).toarray() # create tridiagonal matrix
    if (method=='Gauss'):
        out = Gaussian_elimination(tridiag,y)
    elif (method=='Thomas'):
        out = Thomas_algorithm(tridiag,y)
    return (tridiag, backward_substitution(out[0],out[1]))

def relative_error(A,x,y):
    ''' Function that puts the numerical solution x back into the original
        matrix equation Ax=y and finds how much the result deviates from the
        original right-hand-side y '''
    return abs(np.dot(A,x)-y)/abs(y)


N = 10                  # size of the tridiagonal matrix
a = -1.*np.ones(N-1)    # values for the diagonal entries a
b = 3.*np.ones(N)       # values for the diagonal entries b
c = -1.*np.ones(N-1)    # values for the diagonal entries c
y = 0.2*np.ones(N)      # values for the right-hand-side vector y

tridiag, solution = solve_tridiagonal_system(a,b,c,y,method='Gauss')
rel_error = relative_error(tridiag,solution,y)
print('\nGaussian Elimination:\n')
print('Solution vector:\n{}'.format(solution))
print('Relative Error:\n{}'.format(rel_error))

tridiag, solution = solve_tridiagonal_system(a,b,c,y,method='Thomas')
rel_error = relative_error(tridiag,solution,y)
print('\nThomas Algorithm:\n')
print('Solution vector:\n{}'.format(solution))
print('Relative Error:\n{}'.format(rel_error))
