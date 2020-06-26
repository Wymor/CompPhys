import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#Initial Values
sig = 10
b = 8/3
r = np.array([0.5,1.17, 1.3456, 25.0, 29.0])
a0 = np.sqrt(b*(r.astype('complex')-1))

#Lorenz System
def dx(x,y,z,sig):
    return -sig*(x-y)
def dy(x,y,z,r):
    return r*x-y-x*z
def dz(x,y,z,b):
    return x*y-b*z

def rk4_step(x0, y0, z0, f1, f2, f3, h, f1_args = {}, f2_args = {}, f3_args = {}):
    ''' Simple python implementation for one RK4 step.
        Inputs:
            x_0     - M x 1 numpy array specifying all variables of the first ODE at the current time step
            y_0     - M x 1 numpy array specifying all variables of the second ODE at the current time step
            z_0     - M x 1 numpy array specifying all variables of the third ODE at the current time step
            f       - function that calculates the derivates of all variables of the ODE
            h       - step size
            f1_args - Dictionary of additional arguments to be passed to the function f1
            f2_args - Dictionary of additional arguments to be passed to the function f2
            f3_args - Dictionary of additional arguments to be passed to the function f3
        Output:
            xp1 - M x 1 numpy array of variables of the first ODE at time step x0 + h
            yp1 - M x 1 numpy array of variables of the second ODE at time step x0 + h
            zp1 - M x 1 numpy array of variables of the third ODE at time step x0 + h
            tp1 - time step t0+h
    '''
    k1 = h * f1(x0, y0, z0, **f1_args)
    l1 = h * f2(x0, y0, z0, **f2_args)
    m1 = h * f3(x0, y0, z0, **f3_args)

    k2 = h * f1(x0 + k1/2., y0 + k1/2., z0 + k1/2., **f1_args)
    l2 = h * f2(x0 + l1/2., y0 + l1/2., z0 + l1/2., **f2_args)
    m2 = h * f3(x0 + m1/2., y0 + m1/2., z0 + m1/2., **f3_args)

    k3 = h * f1(x0 + k2/2., y0 + k2/2., z0+ k2/2, **f1_args)
    l3 = h * f2(x0 + l2/2., y0 + l2/2., z0+ l2/2, **f2_args)
    m3 = h * f3(x0 + m2/2., y0 + m2/2., z0+ m2/2, **f3_args)

    k4 = h * f1(x0 + k3/2., y0 + k3/2., z0+ k3/2, **f1_args)
    l4 = h * f2(x0 + l3/2., y0 + l3/2., z0+ l3/2, **f2_args)
    m4 = h * f3(x0 + m3/2., y0 + m3/2., z0+ m3/2, **f3_args)

    xp1 = x0 + 1./6.*(k1 + 2.*k2 + 2.*k3 + k4)
    yp1 = y0 + 1./6.*(l1 + 2.*l2 + 2.*l3 + l4)
    zp1 = z0 + 1./6.*(m1 + 2.*m2 + 2.*m3 + m4)

    return(xp1, yp1, zp1)

def rk4(x0, y0, z0, f1, f2, f3, h, n, f1_args = {}, f2_args = {}, f3_args = {}):
    ''' Simple implementation of RK4
        Inputs:
            x_0     - M x 1 numpy array specifying all variables of the ODE at the current time step
            y_0     - M x 1 numpy array specifying all variables of the ODE at the current time step
            z_0     - M x 1 numpy array specifying all variables of the ODE at the current time step
            f1      - function that calculates the derivates of all variables of the first ODE
            f1      - function that calculates the derivates of all variables of the second ODE
            f1      - function that calculates the derivates of all variables of the  third ODE
            h       - step size
            n       - number of steps
            f1_args - Dictionary of additional arguments to be passed to the function f1
            f2_args - Dictionary of additional arguments to be passed to the function f2
            f3_args - Dictionary of additional arguments to be passed to the function f3
        Output:
            yn - N+1 x M numpy array with the results of the integration for every time step (includes y0)
            xn - N+1 x 1 numpy array with the time step value (includes start x0)
    '''
<<<<<<< HEAD
    xn = np.zeros(n+1)
    yn = np.zeros(n+1)
    zn = np.zeros(n+1)
    xn[0] = x0
    yn[0] = y0
    zn[0] = z0

=======
    xn = np.zeros(n+1); xn[0] = x0
    yn = np.zeros(n+1); yn[0] = y0
    zn = np.zeros(n+1); zn[0] = z0
    
>>>>>>> 6d97ec93d2ecac30c938a56777c01267bf83b7e8
    for n in np.arange(1,n+1,1):
        xn[n], yn[n], zn[n], = rk4_step(x0 = xn[n-1], y0 = yn[n-1], z0 = zn[n-1], f1 = f1, f2 = f2, f3 = f3, h = h, f1_args = f1_args, f2_args = f2_args, f3_args = f3_args)
    return(xn, yn, zn)

plt.figure()
ax = plt.axes(projection="3d")
for i in range(3,4):
    if r[i] <= 1:
        ax.scatter3D(*rk4(x0=1, y0=1, z0=1, f1=dx, f2=dy, f3=dz, h=0.1, n=100000, f1_args={'sig':sig}, f2_args={'r':r[i]}, f3_args={'b':b}), s=1, label='r={}'.format(r[i]))
    else:
        ax.scatter3D(*rk4(x0=a0[i]+20, y0=a0[i]+20, z0=r[i]+19, f1=dx, f2=dy, f3=dz, h=0.01, n=1000, f1_args={'sig':sig}, f2_args={'r':r[i]}, f3_args={'b':b}), s=1, label='r={}'.format(r[i]))
plt.legend()
plt.show()
