#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:17:31 2022

@author: fn

debug Tax
"""

import matplotlib.pyplot as plt
import numpy as np
from functions import *
import os



# Preamble: customization of matplotlib
# Configuration for plots
plt.rc('text', usetex=False)      
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.rcParams['font.size'] = 12
format_fig = (16,9)

#create directory for pictures
path = "pics"
isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)

#%%INPUT DATA
DeltaX = np.linspace(0,2,100)
Vx1 = 0.34 #non dimensional background axial flow velocity at inlet
Vy1 = 0 #non dimensional background azimuthal flow velocity at outlet
DeltaX = 0.3 #gap between rotor and stator

#rotor parameters (pag. 147)
beta1 = -71.1*np.pi/180 #relative inlet swirl
alfa1 = 0*np.pi/180 #absolute inlet swirl
beta2 = -35*np.pi/180 #relative outlet swirl
alfa2 = 65.7*np.pi/180 #absolute outlet swirl
dLr_dPhi = -0.6938 #steady state rotor loss derivative at background condition
dLr_dTanb = dLr_dPhi/((np.tan(alfa1)-np.tan(beta1))**2) #steady state rotor loss derivative at background condition
c_r = 0.135 #blade chord
gamma_r = -50.2*np.pi/180 #stagger angle rotor blades
lambda_r = 0.212 #inertia parameter rotor

#stator parameters (pag. 147)
beta3 = -35*np.pi/180 #relative inlet swirl
alfa3 = 65.7*np.pi/180 #absolute inlet swirl
beta4 = -71.1*np.pi/180 #relative outlet swirl
alfa4 = 0.0*np.pi/180 #absolute outlet swirl
dLs_dTana = 0.0411 #steady state stator loss at inlet condition of the stator
c_s = 0.121 #blade chord
gamma_s = 61.8*np.pi/180 #stagger angle rotor blades
lambda_s = 0.256 #inertia parameter rotor

#axial cordinates
x1 = 0
x2 = c_r*np.cos(gamma_r)


#velocities across the stages
Vx2 = Vx1
Vy2 = Vx2*np.tan(alfa2)
Vx3 = Vx1
Vy3 = Vy2
Vx4 = Vx1
Vy4 = 0

#%%
s = 1+1j
n = 2
theta = 0
Dict = {}
x = np.linspace(0,3,1000) #x plays the role of the DeltaX
for ii in range(0,len(x)):
    x3 = x[ii]
    x4 = x3 + c_s*np.cos(gamma_s)
    m1 = np.linalg.inv(Tax_n(x4, s, n, Vx4, Vy4, theta=theta))
    m2 = Bsta_n(s, n, Vx3, Vy3, Vy4, alfa3, alfa4, lambda_s, dLs_dTana, theta=theta)
    m3 = Bgap_n(x2, x3, s, n, Vx2, Vy2, theta=theta)
    m4 = Brot_n(s, n, Vx1, Vy1, Vy2, alfa1, beta1, beta2, lambda_r, dLr_dTanb, theta=theta)
    m5 = Tax_n(x1, s, n, Vx1, Vy1, theta=theta)
    m6 = np.linalg.multi_dot([m1,m2,m3,m4,m5])
    EC = np.array([[1,0,0]])
    IC = np.array([[0,1,0],
                    [0,0,1]])
    Y = np.concatenate((np.matmul(EC,m6),IC))
    Dict[x[ii]] = m6
    
fig, axes = plt.subplots(3,3, figsize=format_fig)
fig.suptitle('X, s='+str(s.real)+'+'+str(s.imag)+'j')
axes[0,0].set_ylabel(r'$X_{11}$')
axes[0,1].set_ylabel(r'$X_{12}$')
axes[0,2].set_ylabel(r'$X_{13}$')
axes[1,0].set_ylabel(r'$X_{21}$')
axes[1,1].set_ylabel(r'$X_{22}$')
axes[1,2].set_ylabel(r'$X_{23}$')
axes[2,0].set_ylabel(r'$X_{31}$')
axes[2,1].set_ylabel(r'$X_{32}$')
axes[2,2].set_ylabel(r'$X_{33}$')
T11 = np.zeros(len(x), dtype = complex)
T12 = np.zeros(len(x), dtype = complex)
T13 = np.zeros(len(x), dtype = complex)
T21 = np.zeros(len(x), dtype = complex)
T22 = np.zeros(len(x), dtype = complex)
T23 = np.zeros(len(x), dtype = complex)
T31 = np.zeros(len(x), dtype = complex)
T32 = np.zeros(len(x), dtype = complex)
T33 = np.zeros(len(x), dtype = complex)
for ii in range(0,len(x)):
    T11[ii] = Dict[x[ii]][0,0]
    T12[ii] = Dict[x[ii]][0,1]
    T13[ii] = Dict[x[ii]][0,2]
    T21[ii] = Dict[x[ii]][1,0]
    T22[ii] = Dict[x[ii]][1,1]
    T23[ii] = Dict[x[ii]][1,2]
    T31[ii] = Dict[x[ii]][2,0]
    T32[ii] = Dict[x[ii]][2,1]
    T33[ii] = Dict[x[ii]][2,2]
axes[0,0].plot(x, T11.real, label='real')
axes[0,0].plot(x, T11.imag, label='imaginary')
axes[0,1].plot(x, T12.real)
axes[0,1].plot(x, T12.imag)
axes[0,2].plot(x, T13.real)
axes[0,2].plot(x, T13.imag)
axes[1,0].plot(x, T21.real)
axes[1,0].plot(x, T21.imag)
axes[1,1].plot(x, T22.real)
axes[1,1].plot(x, T22.imag)
axes[1,2].plot(x, T23.real)
axes[1,2].plot(x, T23.imag)
axes[2,0].plot(x, T31.real)
axes[2,0].plot(x, T31.imag)
axes[2,1].plot(x, T32.real)
axes[2,1].plot(x, T32.imag)
axes[2,2].plot(x, T33.real)
axes[2,2].plot(x, T33.imag)
axes[2,0].set_xlabel(r'$x_{gap}$')
axes[2,1].set_xlabel(r'$x_{gap}$')
axes[2,2].set_xlabel(r'$x_{gap}$')
fig.legend()
fig.savefig(path+'/X_rot_stat.png')


