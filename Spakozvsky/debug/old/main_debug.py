#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:17:31 2022

@author: fn

test file for functions.py
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(1, '../../') #to add function folder
import Spakozvsky
from functions import *
import os 

# Preamble: customization of matplotlib
# Configuration for plots
plt.rc('text', usetex=False)      
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.rcParams['font.size'] = 14
format_fig = (12,8)

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

#build all the functions written in the functions file, to see it if they work
swirler = Trad_n(r, r, n, s, Q, GAMMA)  
axial = Tax_n(x, s, n, Vx1, Vy1)  
stator = Bsta_n(s, n, Vx, Vy1, Vy2, alfa1, alfa2, lambda_s, dLs_dTana, theta=1)
rotor = Brot_n(s, n, Vx1, Vy1, Vy2, alfa1, beta1, beta2, lambda_r, dLr_dTanb, theta=0.5) 
impeller = Bimp_n(s, n, Vx1, Vr2, Vy1, Vy2, alfa1, beta1, beta2, r1, r2, rho1, rho2, A1, A2, s_i, dLi_dTanb, theta=1)
diffuser = Bdif_n(s, n, Vr1, Vr2, Vy1, Vy2, alfa1, beta1, alfa2, r1, r2, rho1, rho2, A1, A2, s_dif, dLd_dTana, theta=0.1)
vaneless_diffuser = Bvlsd_n(s, n, r1,r2, r1, Q, GAMMA, theta=1)  
gap = Bgap_n(x1, x2, s, n, Vx, Vy, theta=1)

axial = Tax_n(0, 1, 1, 1, 1)
swirler = Trad_n(1, 1, 1, 1, 1, 1)

r = 1
r0 = 1

#%%Poles of a test function  

def test_function(s,n):
    #test function to debug the shot gun method. 7 poles around unitary circle
    return s**7+1

#% pre-stall waves in an isolated rotor 5.4.2
def rotor_row_matrix(s, n=1, theta=0):
    #function of the system rotor stator interaction 5.4.3
    A = np.zeros((3,3),dtype=complex)
    A[0,0] = 1
    A[0,1] = 0
    A[0,2] = -1
    A[1,0] = 0
    A[1,1] = 1
    A[1,2] = -1/Vx + 1j*s/(n*Vx)
    A[2,0] = dLr_dTanb*(1j-np.tan(beta1))/Vx**2 + s*(lambda_r*n +1)/(n*Vx) + (1j*n*lambda_r-np.tan(beta2))/Vx
    A[2,1] = 1j-np.tan(alfa2)
    A[2,2]= 1+np.tan(alfa2)*(np.tan(alfa2)-1j*s/(n*Vx))
    return A

domain = [0,0.5,0,4]
grid = [100,100]
poles_analytic = []
poles = SVD_Method(rotor_row_matrix, domain, grid, n=1)
poles = SVD_Method(rotor_row_matrix, domain, grid, n=2)
poles = SVD_Method(rotor_row_matrix, domain, grid, n=3)
poles = SVD_Method(rotor_row_matrix, domain, grid, n=4)
poles = SVD_Method(rotor_row_matrix, domain, grid, n=5)
poles = SVD_Method(rotor_row_matrix, domain, grid, n=6)






#%%
plt.figure(figsize=format_fig)
#poles = Shot_Gun(test_function, domain, grid)
plt.plot(poles.real,-poles.imag,'o', label='shot-gun')
for k in range(0,7):
    poles_analytic.append(np.exp(1j*(np.pi+2*k*np.pi/7)))
poles_analytic = np.array(poles_analytic, dtype=complex)
plt.plot(poles_analytic.real,poles_analytic.imag,'kx', label = 'Analytic')
plt.legend()
plt.xlabel(r'$\sigma_{n}$')
plt.ylabel(r'$j \omega_{n}$')
plt.title(r'$s^7 = -1$')
plt.savefig(path+'/poles_test.png')



#%% pre-stall waves in an isolated stator
def stator_row(s, n, theta=0):
    #function of the stator 5.4.3
    A = np.zeros((3,3),dtype=complex)
    A[0,0] = 1
    A[0,1] = -1
    A[0,2] = -1
    A[1,0] = 0
    A[1,1] = -1j
    A[1,2] = -1j*s/(n*Vx)
    A[2,0] = dLs_dTana*(1j-np.tan(alfa1))/Vx**2 + s*(lambda_s*n +1)/(n*Vx)
    A[2,1] = s/(n*Vx)
    A[2,2]= 1
    return np.linalg.det(A)

domain = [-2,2,-2,2]
grid = [2,2]
n=np.arange(1,7)
poles_list_analytic_stat = []
plt.figure(figsize=format_fig)
for nn in n:
    poles = Shot_Gun(stator_row, domain, grid, n=nn)
    plt.plot(poles.real,-poles.imag,'o', label='n='+str(nn))
    poles_list_analytic_stat.append(complex(nn*Vx)) #not physical poles
    poles_list_analytic_stat.append(complex((dLs_dTana*np.tan(alfa1)/Vx-Vx)/(lambda_s+2/nn),
                                        (dLs_dTana/Vx)/(lambda_s+2/nn)))
poles_list_analytic_stat = np.array(poles_list_analytic_stat, dtype=complex)
plt.plot(poles_list_analytic_stat.real,poles_list_analytic_stat.imag,'kx', label = 'Analytic')
plt.legend()
plt.xlabel(r'$\sigma_{n}$')
plt.ylabel(r'$j \omega_{n}$')
plt.title('Poles of an isolated stator')
plt.savefig(path+'/poles_stator.png')


#%% pre-stall waves in an isolated rotor 5.4.2
def rotor_row(s, n, theta=0):
    #function of the system rotor stator interaction 5.4.3
    A = np.zeros((3,3),dtype=complex)
    A[0,0] = 1
    A[0,1] = 0
    A[0,2] = -1
    A[1,0] = 0
    A[1,1] = 1
    A[1,2] = -1/Vx + 1j*s/(n*Vx)
    A[2,0] = dLr_dTanb*(1j-np.tan(beta1))/Vx**2 + s*(lambda_r*n +1)/(n*Vx) + (1j*n*lambda_r-np.tan(beta2))/Vx
    A[2,1] = 1j-np.tan(alfa2)
    A[2,2]= 1+np.tan(alfa2)*(np.tan(alfa2)-1j*s/(n*Vx))
    return np.linalg.det(A)

domain = [-2,2,-2,2]
grid=[2,2]
n=np.arange(1,7)
poles_analytic_rot = []
plt.figure(figsize=format_fig)
for nn in n:
    poles = Shot_Gun(rotor_row, domain, grid, n=nn)
    plt.plot(poles.real,-poles.imag,'o', label='n '+str(nn))
    poles_analytic_rot.append(complex(((np.tan(beta2)+dLr_dTanb*np.tan(beta1)/Vx-Vx*(1+np.tan(alfa2)**2)+np.tan(alfa2))/(lambda_r+2/nn)),
                                        (dLr_dTanb/Vx + nn*lambda_r+1)/(lambda_r+2/nn)))
poles_analytic_rot = np.array(poles_analytic_rot, dtype=complex)
plt.plot(poles_analytic_rot.real,poles_analytic_rot.imag,'kx', label = 'Analytic')
plt.legend()
plt.xlabel(r'$\sigma_{n}$')
plt.ylabel(r'$j \omega_{n}$')
plt.title('Poles of an isolated rotor')
plt.savefig(path+'/poles_rotor.png')

#%%plot of radial functions
from functions import *

r = np.linspace(1,1.5,10000)
r0 = r[0]
Rn = np.zeros((len(r)),dtype=complex)
Rn_prime = np.zeros((len(r)),dtype=complex)
Rn_second_ = np.zeros((len(r)),dtype=complex)
#proof value
n = 20
s = -20j
Q = 1
GAMMA = 1
for i in range(0,len(r)):
    Rn[i], Rn_prime[i] = Rad_fun(r[i], r0,n,s,Q,GAMMA)
    Rn_second_[i] = Rn_second(r[i], r0, n,s,Q,GAMMA)

fig, axes = plt.subplots(3,1, figsize=(12,9))
fig.suptitle('Radial functions, s='+str(s.real)+'+'+str(s.imag)+'j')
axes[0].set_ylabel(r'$R_n$')
axes[1].set_ylabel(r'$\frac{dR_n}{dr}$')
axes[2].set_ylabel(r'$\frac{d^2 R_n} {dr^2}$')
axes[2].set_xlabel(r'$r $')
axes[0].plot(r, Rn.real, label='real')
axes[0].plot(r, Rn.imag, label='imag')
axes[1].plot(r, Rn_prime.real)
axes[1].plot(r, Rn_prime.imag)
axes[2].plot(r, Rn_second_.real)
axes[2].plot(r, Rn_second_.imag)
fig.legend()

fig, axes = plt.subplots(2,1, figsize=(7,10))
fig.suptitle('Radial functions, s='+str(s.real)+'+'+str(s.imag)+'j')
axes[0].set_ylabel(r'$Re[R_n]$')
axes[1].set_ylabel(r'$Im[R_n]$')
axes[1].set_xlabel(r'$r $')
axes[0].plot(r, Rn.real)
axes[1].plot(r, Rn.imag)


#first derivative by finite difference
Rn_prime2 = np.zeros((len(r)),dtype=complex)
N = len(r)-1
for i in range(0,N):
    Rn_prime2[i] = (Rn[i+1]-Rn[i])/(r[i+1]-r[i])
Rn_prime2[N] = Rn_prime2[N-1]
# axes[2].plot(r, Rn_prime2)


#second derivative by finite difference
Rn_second2 = np.zeros((len(r)),dtype=complex)
N = len(r)-1
for i in range(0,N):
    Rn_second2[i] = (Rn_prime[i+1]-Rn_prime[i])/(r[i+1]-r[i])
Rn_second2[N] = Rn_second2[N-1]
# axes[3].plot(r, Rn_second2)

plt.show()











