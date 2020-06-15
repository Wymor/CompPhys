# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 08:  Population Dynamics - Stationary Points
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np; import matplotlib.pyplot as plt
import numpy.linalg as linalg

A = np.array([[-1,0,0,-20,-30,-5],[0,-1,0,-1,-3,-7],[0,0,-1,-4,-10,-20],
             [20,30,35,0,0,0],[3,3,3,0,0,0],[7,8,20,0,0,0]])

ew,ev = linalg.eig(A)
print(ew)
print(ev)
