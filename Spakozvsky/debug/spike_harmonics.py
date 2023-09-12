#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 10:07:54 2023

@author: fneri

Study how many harmonics we need to consider if want to sudy the stability of spikes disturbances. Spike disturbance
here is considered as a sinusoidal wave in the pressure field which spans two blade pitches (consider the paper:
"Origins and Structure of Spike-Type Rotating Stall, Pullan et Al")
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

# Preamble: customization of matplotlib
# Configuration for plots
plt.rc('text', usetex=False)      
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.rcParams['font.size'] = 14
format_fig = (10,6)

N = 12 #number of blades
sample_points = 10000 #sample points along one azimuthal period
sample_spacing = 2*np.pi/sample_points #sample spacing in theta
theta = np.linspace(0,2*np.pi,sample_points) #theta  cordinate
delta_theta = (2*np.pi)*(2/N) #theta span occupied by 2 blades, which are the one undergoing spike disturbance
p = np.zeros(len(theta)) #pressure signal
num_points = int(len(theta)*delta_theta/(2*np.pi)) #number of points along which the spike is present
A = 1 #amplitude spike
noise = A/50 #amplitude noise 
for i in range(int(len(p)/2-num_points/2),int(len(p)/2+num_points/2)):
    p[i] = A*np.sin(theta[i]*N/2) #add spike signal to mean pressure
for i in range(0,len(p)):
    p[i] = p[i] + np.random.uniform(-noise,noise) #add noise to pressure signal
    
plt.figure(figsize=format_fig)
plt.plot(theta,p, linewidth=1)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\Delta p / \Delta p_{max}$')
plt.title('Spike disturbance')

pf = fft(p) #fourier transform of pressure
thetaf = fftfreq(sample_points, 2*np.pi)[:sample_points//2] #obtaine the frequencies 1/rad

final_harm = 100
n = np.linspace(1,final_harm,final_harm, dtype=int)
limit = np.ones(final_harm)*0.1
plt.figure(figsize=format_fig)
# plt.plot(np.abs(pf[0:final_harm])/np.max(np.abs(pf)),'-o',label='Spectral Content') #plot as a function of azimuthal harmonic
plt.stem(n, np.abs(pf[0:final_harm])/np.max(np.abs(pf)), 'b', \
         markerfmt="b.", basefmt="-b", label='Spectral Content')
plt.plot(limit,'--r',linewidth='1', label='10%')
plt.xlabel(r'$n$')
plt.ylabel(r'$\Delta \hat{p} / \Delta \hat{p}_{max}$')
plt.legend()
plt.title('Spike spectral content')
