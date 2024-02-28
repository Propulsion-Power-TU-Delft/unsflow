#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:17:31 2022

@author: fn

debug Bvlsd
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

#%% exercise rotor stator pre-stall modes

#Geometrical and compressor parameters (pag. 147)
Q = 0.215 #source term of the swirling flow
GAMMA = 0.7032 #rotational term of the swirling flow
N = 20 #number of blades
Vx = 0.34 #non dimensional background axial flow velocity at inlet
Vy = 0 #non dimensional background azimuthal flow velocity at outlet
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
  

#%%debug parameters
r = 1
r0 = 0.9
n = 1
s = 1+1j
theta = np.pi/4
x = 0.3
x1 = 0
x2 = 0.3
Vx1 = 1
Vx2 = 0.9
Vr1 = 0.7
Vr2 = 0.5
Vy1 = 0.3
Vy2 = 0.2
r1 = 0.3
r2 = 0.8
rho1 = 1.014
rho2 = 1.3
A1 = 1
A2 = 0.8
s_i = 0.8
s_dif = 0.7
dLi_dTanb = -0.9
dLd_dTana = -0.5

#%%
s = 1-3j
n = 3
Dict = {}
R2 = 1.0
R3 = 3.0
x = np.linspace(R2,R3,1000)
for ii in range(0,len(x)):
    Dict[x[ii]] = Bvlsd_n(s, n, R2, x[ii], R2, Q, GAMMA)
    
fig, axes = plt.subplots(3,3, figsize=format_fig)
fig.suptitle('Bvlsd, s='+str(s.real)+'+'+str(s.imag)+'j')
axes[0,0].set_ylabel(r'$B_{vlsd_{11}}$')
axes[0,1].set_ylabel(r'$B_{vlsd_{12}}$')
axes[0,2].set_ylabel(r'$B_{vlsd_{13}}$')
axes[1,0].set_ylabel(r'$B_{vlsd_{21}}$')
axes[1,1].set_ylabel(r'$B_{vlsd_{22}}$')
axes[1,2].set_ylabel(r'$B_{vlsd_{23}}$')
axes[2,0].set_ylabel(r'$B_{vlsd_{31}}$')
axes[2,1].set_ylabel(r'$B_{vlsd_{32}}$')
axes[2,2].set_ylabel(r'$B_{vlsd_{33}}$')
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
axes[2,0].set_xlabel(r'$r_2/r_1$')
axes[2,1].set_xlabel(r'$r_2/r_1$')
axes[2,2].set_xlabel(r'$r_2/r_1$')
fig.legend()
fig.savefig(path+'/Bvlsd.png')


