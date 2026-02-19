#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:17:31 2022

@author: fn
2DOF linearized model greitzer
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import fsolve


# Preamble: customization of matplotlib
# Configuration for plots
plt.rc('text', usetex=False)      
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.rcParams['font.size'] = 14
format_fig = (12,8)


#%%
#Greitzer coefficients span to understand B behavior
B = np.linspace(0,10,10)
G = 2

#Greitzer coefficients calculated for a certain compressor (se aumenti B ottiene il ciclo limite)
N = 10e3                                    #[rpm] --> large N enlarges the cirle
r_ref = 100e-3                              #radius [m]
omega = 2*np.pi*N/60                        #[rad/s]
U_ref = omega*r_ref                         #reference speed of compressor[m/s]
a = 340                                     #speed of sound [m/s]
Vp = 0.000075                               #plenum volume [m3]
dc = 10e-3                                  #inlet diameter [m]
Ac = (np.pi*dc**2)/4                        #inlet area [m2]
Lc = 0.1                                    #inlet length [m]
dt = 10e-3                                  #throttle diameter [m]
At = (np.pi*dt**2)/4                        #throttle area [m2]
Lt = 0.1                                    #throttle length [m]
B_real = (U_ref/(2*a))*np.sqrt(Vp/(Ac*Lc))  #B parameter of the described compressor
G_real = Lt*Ac/(Lc*At)                      #G parameter of the described compressor
k_valve = 3 
# G_real = 1000

#%%
#compressor points
phi_p = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
psi_p = np.array([0.5, 0.8, 0.95, 1.0, 0.8])

#polynomial interpolation for the compressor curve
z_coeff = np.polyfit(phi_p, psi_p, 3)
phi = np.linspace(0,1,100)
psi_c = np.polyval(z_coeff,phi)

#throttle valve curve
psi_v = k_valve*phi**2

#plot the compressor curve
plt.figure(figsize=format_fig)
plt.scatter(phi_p, psi_p)
plt.plot(phi,psi_c,label='Compressor')
plt.plot(phi,psi_v,'--r',label='Throttle')
plt.ylim(0,1.5)
plt.ylabel('$\Psi$')
plt.xlabel('$\Phi$')
plt.legend()

#find the intersection flow coefficient between compressor and throttle
def func_work_coefficient(phi):
    return np.polyval(z_coeff,phi) - k_valve*phi**2

phi_eq = fsolve(func_work_coefficient,0.7)[0]               #equilibrium flow coefficient
psi_eq = k_valve*phi_eq**2                                  #equilibrium work coefficient

#calculate derivatives of the compressor and throttle characteristic at equilibrium point
delta_phi = phi_eq*0.001
phi_left = phi_eq - delta_phi
phi_right = phi_eq + delta_phi
psi_c_right = np.polyval(z_coeff,phi_right)
psi_c_left = np.polyval(z_coeff,phi_left)
psi_c_prime = (psi_c_right - psi_c_left) / (2*delta_phi)
psi_v_prime = 2*k_valve*phi_eq

#%% solve the reduced differential system of equations (as defined by Sundstrom)
def Greitzer_2DOF(y, t, B, G, k_valve, closure_coeff, psi_c_prime, psi_v_prime):
    """
    Defines the differential equations for the Greitzer model
    (as found in the thesis of Sündstrom).

    Arguments:
        y :  vector of the state variables
        t :  time
        B :  B parameter
        G :  G parameter
        
    State variables:
        x1 : compressor flow coefficient fluctuation from equilibrium value
        x2 : work coefficient fluctuation from equilibrium value
    """
    x1, x2 = y
    dydt = [B * (psi_c_prime*x1 - x2),
            (x1 - x2/psi_v_prime)/B ]
            # (x1 - x2/psi_v_prime)/B + np.cos(1100*omega*t)]
    return dydt

t = np.linspace(0,20,100000)


y0 = [0.01, 
      0.01] #initial conditions
IC_x = [y0[0]]
IC_y = [y0[1]]

from scipy.integrate import odeint
closure_coeff = 1
sol = odeint(Greitzer_2DOF, y0, t, args=(B_real, G_real, k_valve, closure_coeff, psi_c_prime, psi_v_prime))
initialDerivative = Greitzer_2DOF(y0, t[0], B_real, G_real, k_valve, closure_coeff, psi_c_prime, psi_v_prime)

#temporal evolution plots
fig, axes = plt.subplots(2,1, figsize=format_fig)
axes[0].set_ylabel(r'$\phi_{c} $')
axes[0].plot(t,sol[:,0])
axes[1].set_ylabel(r'$\psi  $')
axes[1].plot(t,sol[:,1])
axes[1].set_xlabel(r'$\xi $')
fig.suptitle('Transient after initial perturbation')

#plots in the phase space
fig, axes = plt.subplots(1, figsize=format_fig)
axes.set_ylabel(r'$\psi$')
axes.set_xlabel(r'$\phi_{c}$')
axes.plot(sol[:,0], sol[:,1],linewidth=0.8)
axes.plot(IC_x, IC_y, marker='o',color='b', label='IC')
fig.suptitle('Phase Space trajectories')

# #plot the compressor curve
# plt.figure(figsize=format_fig)
# # plt.scatter(phi_p, psi_p, 'v')
# plt.plot(phi,psi_c,label='Compressor')
# plt.plot(phi,psi_v,'--k',label='Throttle Initial')
# plt.plot(phi,psi_v*closure_coeff,'--r',label='Throttle Final')
# plt.plot(sol[:,0],sol[:,1],linewidth=0.8, label = 'Transient')
# plt.plot(IC_x, IC_y, 'ko', label='Initial condition')
# plt.ylabel(r'$\Psi$')
# plt.xlabel(r'$\Phi$')
# plt.legend()
# plt.ylim(0,2)
# plt.title('Initial operating point')
 
#%% root locus plots
from numpy.lib.scimath import sqrt as csqrt
def roots_equation(B):
    alfa = -0.5* (1/(B*psi_v_prime) - B*psi_c_prime) 
    beta = 0.5*csqrt((1/(B*psi_v_prime) - B*psi_c_prime)**2 -4*(1-psi_c_prime/psi_v_prime))
    beta_real = beta.real
    beta_imag = beta.imag
    alfa_sum1 = alfa + beta_real
    alfa_sum2 = alfa - beta_real
    return alfa_sum1, alfa_sum2, beta_imag
    
B_locus = np.linspace(0.01,10,100000)
real1, real2, imag1 = roots_equation(B_locus)

plt.figure()
plt.plot(real1,imag1)
plt.plot(real2,-imag1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




