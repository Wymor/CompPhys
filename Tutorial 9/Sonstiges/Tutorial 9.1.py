import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

#Initial Values
sig = 10
b = 8/3

def P(lam,r):
    return lam**3+(1+b+sig)*lam**2+b*(sig+r)*lam+2*sig*b*(r-1)

r = np.array([0, 0.5, 1, 1.8])

plt.figure(figsize=(10,6))
plt.axhline(0, ls='--', color='lightgray')
lam = np.linspace(-10,10,1000)
for i in range(0, len(r)):
    plt.plot(lam, P(lam, r[i]), label=r"$r=${}".format(r[i]))
    zero = fsolve(P, args=(r[i]), x0=[-2,2])
    plt.plot(zero, np.zeros(len(zero)),'r.')
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$P(\lambda)$")
plt.legend()
plt.show()

#Suche nach komplexen Nullstellen mit Sympy
'''
from sympy import Symbol, nsolve , I
lam = Symbol('lam')
for i in range(0, len(r)):
    print(nsolve(lam**3+(1+b+sig)*lam**2+b*(sig+r[i])*lam+2*sig*b*(r[i]-1), (-1-I,1+I)))
'''
#Suche nach komplexen Nullstellen mit mpmath
import mpmath
for i in range(0, len(r)):
    print('Für r = {} erhalten wir die Nullstellen:'.format(r[i]))
    P = lambda lam: lam**3+(1+b+sig)*lam**2+b*(sig+r[i])*lam+2*sig*b*(r[i]-1)
    print(mpmath.findroot(P, x0=0, solver='muller', verbose=True))
