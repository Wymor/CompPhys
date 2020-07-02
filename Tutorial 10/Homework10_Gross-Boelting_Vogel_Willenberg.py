# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 08:  Many Species Population Dynamics
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np; import matplotlib.pyplot as plt
import random

# first method
N = int(1e6)
x = np.array([random.uniform(0,1) for i in range(0,N)])

def function(x):
    return np.sqrt(1-x**2)

pi = 4*(1/N)*np.sum(function(x))
print(pi)

# second method
N = int(1e3)
random_x = np.array([random.uniform(0,1) for i in range(0,N)])
random_y = np.array([random.uniform(0,1) for i in range(0,N)])

x_in = random_x[random_y <= function(random_x)]
y_in = random_y[random_y <= function(random_x)]

x_out = random_x[random_y > function(random_x)]
y_out = random_y[random_y > function(random_x)]

print(4*len(x_in)/N)


fig, ax = plt.subplots()
ax.set_title(r'Determination of $\pi$ using Random Numbers')
ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')

x = np.linspace(0,1,1000)
ax.plot(x,function(x), 'b-', linewidth=1.5, label=r'$y=\sqrt{1-x^2}$')
ax.plot(x_in, y_in, 'g.', label=r'$N_+=${}'.format(len(x_in)))
ax.plot(x_out, y_out, 'r.', label=r'$N_-=${}'.format(len(x_out)))

ax.grid(); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=3); ax.set_aspect('equal', 'box')
ax.set(xlim=(-0.05,1.05), ylim=(-0.05,1.05))
fig.savefig('figures/Random-Number_pi-Determination.pdf', format='pdf')
plt.show(); plt.clf(); plt.close()
