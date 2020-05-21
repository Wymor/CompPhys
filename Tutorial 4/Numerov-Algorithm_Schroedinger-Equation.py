# -*- coding: utf-8 -*-
"""
Introduction to Computational Physics
- Exercise 04:  Numerov Algorithm for the Schrödinger Equation
                Neutrons in the Gravitational Field of the Earth
- Group: Simon Groß-Bölting, Lorenz Vogel, Sebastian Willenberg
"""

import numpy as np; import matplotlib.pyplot as plt
import scipy.constants as const

class Particle:
    def __init__(self,mass,energy=0):
        ''' Input:  mass [kg] and energy [eV] of the particle '''
        self.mass = mass        # mass [kg] of the particle
        self.energy = energy    # energy [eV] of the particle

def calculate_wavefunction(particle,zrange,N):
    ''' Function to compute the wavefunction for a particle in the
        gravitational field of the Earth using the Numerov algorithm
        Input:  particle of the class "Particle" with mass [kg] and energy [eV]
        Output: dimensionless length range and wavefunction '''

    # compute the characteristic length scale
    z0 = (const.hbar**2/(2*particle.mass**2*const.g))**(1/3)

    # compute the scaled length and energy units (dimensionless)
    x = zrange/z0; epsilon = particle.energy*const.e/(particle.mass*const.g*z0)

    k = epsilon-x
    h_sq = (abs(max(x)-min(x))/N)**2    # squared step size
    psi = np.zeros(N)                   # empty array for the wavefunction
    psi[0] = 0; psi[1] = 1e-20          # set conditions at the mirror

    for i in range(2,N): # compute the wavefunction using the Numerov method
        psi[i] = (2.*(1.-(5./12.)*h_sq*k[i-1])*psi[i-1]
                -(1.+(1./12.)*h_sq*k[i-2])*psi[i-2])/(1.+(1./12.)*h_sq*k[i])
    return (x,psi,epsilon)

def find_eigenvalue(particle,energy_step,precision):
    ''' Function to compute the energy eigenvalues using sign changes
        of the asymptotic behavior for different trial energies
        Input:  particle of the class "Particle" with a trial energy [eV],
                energy step size [eV] and precision [eV]
        Output: energy eigenvalue [eV] '''
    psi1 = calculate_wavefunction(particle, z, N)[1][N-1]

    while abs(energy_step) > precision:
        particle.energy += energy_step
        psi2 = calculate_wavefunction(neutron, z, N)[1][N-1]

        if psi1*psi2<0: # check for sign change
            energy_step = -energy_step/2    # reduce energy step
        psi1 = psi2
    return particle.energy


# Exercise 1: We use the Numerov algorithm to solve the differential equation
N = int(2e4)                            # set number of iterations
z = 1e-6*np.linspace(0,40,N)            # set z-range [0 to 50 micrometre]
energy = 1e-12*np.array([2.4,2.5])      # set energy values
color = ['blue','red']                  # set color of the graphs

# plot the corresponding wavefunctions for two energy values:
# one with positive and one with negative asymptotic behavior
fig, ax = plt.subplots()
ax.set_title('Asymptotic behavior of the Numerov algorithm solutions\n'+
             'for neutrons in the Earth\'s gravitational field')
ax.set_xlabel(r'dimensionless height $x$')
ax.set_ylabel(r'solution $\psi(x)$')

for i in range(0,len(energy)):
    neutron = Particle(const.neutron_mass, energy[i])
    x, psi, eps = calculate_wavefunction(neutron, z, N)
    ax.plot(x[::6], psi[::6], '.', markersize=1, color=color[i],
            label=r'$\epsilon={eps}$, $E={e}\,\mathrm{{peV}}$'
            .format(e=round(neutron.energy*1e12,2),eps=round(eps,2)))

ax.grid(); ax.legend(loc='best', markerscale=8)
ax.set_xlim((int(round(min(x))),int(round(max(x)))))
fig.savefig('figures/Asymptotic-Behavior.pdf',format='pdf')


# Exercise 2: We compute the stationary states of neutrons in the gravitational
# field of the Earth and the energy eigenvalues of the first three bound states
N = int(2e4)                                # set number of iterations
z = 1e-6*np.linspace(0,50,N)                # set z-range [0 to 50 micrometre]
trial_energy = 1e-12*np.array([1.,2.,3.])   # set trial energy values
color = ['blue','green','red']              # set color of the graphs
energy_step = 0.25e-12                      # set energy step
precision = 1e-18                           # set precision limit

# compute the energy eigenvalues and plot the corresponding wavefunctions
# as well as the probability amplitude
fig1, ax1 = plt.subplots(); fig2, ax2 = plt.subplots()
ax1.set_ylabel(r'wavefunction $\psi_n(x)$')
ax2.set_ylabel(r'probability amplitude $|\psi_n(x)|^2$')

for i in range(0,len(trial_energy)):
    neutron = Particle(const.neutron_mass, trial_energy[i])
    neutron.energy = find_eigenvalue(neutron,energy_step,precision)
    x, psi, eps = calculate_wavefunction(neutron, z, N)
    ax1.plot(x[::6], psi[::6], '.', markersize=1, color=color[i],
             label=r'$n={n}$, $\epsilon={eps}$, $E_{n}={e}\,\mathrm{{peV}}$'
             .format(n=i+1,e=round(neutron.energy*1e12,2),eps=round(eps,2)))
    ax2.plot(x[::6], abs(psi[::6])**2, '.', markersize=1, color=color[i],
             label=r'$n={n}$, $\epsilon={eps}$, $E_{n}={e}\,\mathrm{{peV}}$'
             .format(n=i+1,e=round(neutron.energy*1e12,2),eps=round(eps,2)))

for ax in [ax1,ax2]:
    ax.set_title('Quantum states and energy eigenvalues\n'+
                 'of neutrons in the Earth\'s gravitational field')
    ax.set_xlabel(r'dimensionless height $x$')
    ax.grid(); ax.legend(loc='best', markerscale=8)
    ax.set_xlim((int(round(min(x))),int(round(max(x)))))

fig1.savefig('figures/Eigenvalues_Wavefunction.pdf', format='pdf')
fig2.savefig('figures/Eigenvalues_Probability-Amplitude.pdf', format='pdf')
plt.show(); plt.clf(); plt.close()
