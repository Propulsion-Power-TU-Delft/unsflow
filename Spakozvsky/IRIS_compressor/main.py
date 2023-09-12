#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:41:32 2023

@author: Francesco Neri, TU Delft

Exercise on poles location for the IRIS centrifugal compressor described in "The Effect of Size and Working Fluid on the
Multi-Objective Design of High-Speed Centrifugal Compressors" by Andrea Giuffre. The compressor selected is the one
picked from the pareto front.
"""

import matplotlib.pyplot as plt
import numpy as np
# from "../src/functions" import 
import sys

sys.path.insert(1, '../src/')  # to add function folder
from functions import *

# Preamble: customization of matplotlib
# Configuration for plots
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rcParams['font.size'] = 10
format_fig = (18, 8)

# Relevant geometric parameters for the compressor selected by Andrea on the Pareto front. All the variables that begin with
# capital letters are dimensional. Otherwise they have been non-dimensionalized

total_blades = 14  # number of blades (normal +split)
main_blades = 7  # main blades
splitter_blades = 7  # splitter blades
n_max = total_blades // 4 + 1  # maximum harmonic to take in consideration (2 blades in a half-wavelength)

# COMPRESSOR DESIGN SELECTED ALONG THE PARETO FRONT (SI units) - INPUT DATA
fluid = 'R1233zd(E)'  # fluid
Omega_range = np.array([70, 79, 88, 93, 97]) * 1e3 * 2 * np.pi / 60  # angular velocities [rad/s]
R1s = 15.2 * 1e-3  # shroud radius inlet [m]
R1h = 3.4 * 1e-3  # hub radius inlet [m]
R2 = 22.8 * 1e-3  # impeller exit radius [m]
R3 = 35.2 * 1e-3  # diffuser outlet radius [m]
H2 = 2.3 * 1e-3  # blade heigth exit impeller [m]
H3 = 1.6 * 1e-3  # diffuser height [m]
H4 = H3  # diffuser outlet height [m]
Lax = 16 * 1e-3  # axial length [m]
R4 = 49.3 * 1e-3  # external diameter compressor [m]
Ts = 0.3 * 1e-3  # blade trailing edge at shroud [m]
Th = 0.6 * 1e-3  # blade trailing edge at hub [m]
Tte_mean = 0.5 * (Ts + Th)  # blade trailing edge assumed at mid span [m]
R1 = (R1s + R1h) / 2  # radius at impeller inlet [m]
R_Ref = R2  # Reference parameters for non-dimensionalization

# STA locations non dimensionalized
x1 = 0  # impeller inlet
x2 = Lax / R_Ref  # impeller outlet/diffuser inlet
# x3 = x2
x4 = x2  # diffuser outlet
r1 = R1 / R_Ref
r2 = R2 / R_Ref
r3 = R3 / R_Ref
r4 = R4 / R_Ref

# Cross sections
A1 = np.pi * (R1s ** 2 - R1h ** 2)  # cross section [m2]
A2 = 2 * np.pi * R2 * H2  # cross section [m2]
A4 = 2 * np.pi * R4 * H4  # cross section [m2]
A1_blade = A1 / main_blades  # cross section of one sector at impeller inlet [2]
A2_blade = (A2 - total_blades * H2 * Tte_mean) / (total_blades)  # cross section of one sector at impeller outlet [m2]
s_i = np.sqrt(Lax ** 2 + (R2 - R1) ** 2) * 1.3  # approximation of the meridional path length along the impeller [m]

# %%IMPORT DATA FROM DATA FOLDER (IRIS COMPRESSOR ANDREA)
# note: my STA numbers are shifted (1 mine = 0 Andrea, 2 mine = 1 Andrea, 4 mine = 2 Andrea)
import pickle
data_folder = "data/IRIS_single_stage/design0_beta_3.450/operating_map/"  # path to folder

# in the following arrays the zeros means that the working point was out of range (choked)
# every row is a different speedline, given in the RPM vector
# every column is a different mass flow rate

with open(data_folder + 'alpha2.pkl', 'rb') as f:
    Alpha4 = pickle.load(f)
with open(data_folder + 'beta0.pkl', 'rb') as f:
    Beta1 = pickle.load(f)
with open(data_folder + 'beta1.pkl', 'rb') as f:
    Beta2 = pickle.load(f)
with open(data_folder + 'P0.pkl', 'rb') as f:
    P1 = pickle.load(f)  # static pressure
with open(data_folder + 'P1.pkl', 'rb') as f:
    P2 = pickle.load(f)
with open(data_folder + 'P2.pkl', 'rb') as f:
    P4 = pickle.load(f)
with open(data_folder + 'Vm2.pkl', 'rb') as f:
    Vm4 = pickle.load(f)
with open(data_folder + 'Vt2.pkl', 'rb') as f:
    Vt4 = pickle.load(f)
with open(data_folder + 'Wm0.pkl', 'rb') as f:
    Wm1 = pickle.load(f)
with open(data_folder + 'Wm1.pkl', 'rb') as f:
    Wm2 = pickle.load(f)
with open(data_folder + 'Wt0.pkl', 'rb') as f:
    Wt1 = pickle.load(f)
with open(data_folder + 'Wt1.pkl', 'rb') as f:
    Wt2 = pickle.load(f)
with open(data_folder + 'D0.pkl', 'rb') as f:
    Rho1 = pickle.load(f)  # density
with open(data_folder + 'D1.pkl', 'rb') as f:
    Rho2 = pickle.load(f)
with open(data_folder + 'D2.pkl', 'rb') as f:
    Rho4 = pickle.load(f)
with open(data_folder + 'eta_tt.pkl', 'rb') as f:
    eta_tt = pickle.load(f)
with open(data_folder + 'eta_ts.pkl', 'rb') as f:
    eta_ts = pickle.load(f)
with open(data_folder + 'mass_flow.pkl', 'rb') as f:
    mass_flow = pickle.load(f)
with open(data_folder + 'beta_ts.pkl', 'rb') as f:
    beta_ts = pickle.load(f)  # total to static pressure ratio
with open(data_folder + 'beta_tt.pkl', 'rb') as f:
    beta_tt = pickle.load(f)
with open(data_folder + 'rpm.pkl', 'rb') as f:
    rpm = pickle.load(f)

# %%PREPROCESSING OF THE DATA, IN ORDER TO HAVE INPUT DATA READY FOR THE TRANSFER FUNCTIONS

speedline = 2  # choose the speedline to be used
print("Selected speedline : %2d rpm" % (rpm[speedline]))
index_max = np.where(mass_flow[speedline, :] == 0)
index_max = index_max[0]
index_max = index_max[0]
index_max -= 1  # index max in order to avoid the choked data

Omega = rpm[speedline] * 2 * np.pi / 60
U_Ref = Omega * R_Ref  # the reference velocity is the outlet impeller peripheral speed
A_Ref = R_Ref ** 2
p_ratio_tt = beta_tt[speedline, 0:index_max]  # across the whole compressor
mdot = mass_flow[speedline, 0:index_max]
plt.figure(figsize=format_fig)
plt.plot(mdot, p_ratio_tt)
plt.ylabel(r'$\beta_{tt}$')
plt.xlabel(r'$\dot{m}$')
# plt.title('Pressure ratio')


# axial velocities
vx1 = Wm1[speedline, 0:index_max] / U_Ref
vx2 = np.zeros(index_max)
# vx3 = 0 
vx4 = np.zeros(index_max)

# azimuthal absolute velocities
vy1 = np.zeros(index_max)
vy2 = (-Wt2[speedline, 0:index_max] + Omega * R1) / U_Ref  # attention to the sign
# vy3 = 
vy4 = Vt4[speedline, 0:index_max] / U_Ref

# radial velocities
vr1 = np.zeros(index_max)
vr2 = Wm2[speedline, 0:index_max] / U_Ref
# vr3 = 
vr4 = Vm4[speedline, 0:index_max] / U_Ref

alpha1 = np.arctan(vy1 / vx1)  # inlet absolute flow angle

# static pressures [Pa]
p1 = P1[speedline, 0:index_max]
p1_t = p1 + 0.5 * Rho1[speedline, 0:index_max] * (
            (vx1 * U_Ref) ** 2 + (vy1 * U_Ref) ** 2 + (vr1 * U_Ref) ** 2)  # incompressible approximation
p2 = P2[speedline, 0:index_max]
p2_t = p2 + 0.5 * Rho2[speedline, 0:index_max] * ((vx2 * U_Ref) ** 2 + (vy2 * U_Ref) ** 2 + (vr2 * U_Ref) ** 2)
# p3 = not needed for the moment
p4 = P4[speedline, 0:index_max]
p4_t = p4 + 0.5 * Rho4[speedline, 0:index_max] * ((vx4 * U_Ref) ** 2 + (vy4 * U_Ref) ** 2 + (vr4 * U_Ref) ** 2)

# static density [kg/m3]
rho1 = Rho1[speedline, 0:index_max]
rho2 = Rho2[speedline, 0:index_max]
# rho3 = 
rho4 = Rho4[speedline, 0:index_max]
rho_imp = (rho1 + rho2) / 2  # average density in the impeller

# now this is a bit problematic. I assume that the global efficiency is also respected in the impeller alone
# delta_htt_real = Omega*(R2*vy2*U_Ref - R1*vy1*U_Ref) #enthalpy increase experienced in the real impeller (it must be higher than the ideal)
# delta_htt_ideal = delta_htt_real*eta_tt[speedline,0:index_max] #ideal enthalpy increase experienced in the real impeller
# psi_tt_real = delta_htt_real/U_Ref**2 #real impeller work coefficient
# psi_tt_ideal = delta_htt_ideal/U_Ref**2 #ideal impeller work coefficient
# delta_hts_real = Omega*(R2*vy2*U_Ref - R1*vy1*U_Ref) - 0.5*(vy2*U_Ref)**2 #total to static, taking out exit kin. energy
# delta_hts_ideal = delta_hts_real*eta_ts[speedline,0:index_max]
# L_imp = (delta_hts_real - delta_hts_ideal)/U_Ref**2

# plt.figure()
# plt.plot(psi_tt_ideal)
# plt.plot(psi_tt_real)
psi_ts_real = (p2 - p1_t) / (rho_imp * U_Ref ** 2)
psi_ts_ideal = psi_ts_real / eta_ts[speedline, 0:index_max]  # ideal is assumed proportional to total to static efficiency
L_imp = (psi_ts_ideal - psi_ts_real)
phi = mdot / (rho1 * U_Ref * A1)  # inlet flow coefficient for the impeller
dLimp_dphi = np.gradient(L_imp, phi)
dLi_dTanb = np.gradient(L_imp, np.tan(Beta1[speedline, 0:index_max]))

plt.figure(figsize=format_fig)
plt.plot(phi, L_imp)
plt.ylabel(r'$L_{imp}$')
plt.xlabel(r'$\phi$')
plt.title('impeller loss')
plt.figure(figsize=format_fig)
plt.plot(phi, dLimp_dphi)
plt.ylabel(r'$dL_{imp} / d \phi$')
plt.xlabel(r'$\phi$')
plt.title('impeller loss derivative')
plt.figure(figsize=format_fig)
plt.plot(np.tan(Beta1[speedline, 0:index_max]), dLi_dTanb)
plt.ylabel(r'$dL_{imp} / d \tan{\beta_1}$')
plt.xlabel(r'$\beta_1$')
plt.title('impeller loss derivative')

# %% construct dynamic transfer function of the system
Q = 2 * np.pi * r2 * vr2  # non dimensional source term at station 2
GAMMA = 2 * np.pi * r2 * vy2  # non dimensional circulation term at station 2
beta1 = Beta1[speedline, 0:index_max]
beta2 = Beta2[speedline, 0:index_max]
alpha4 = Alpha4[speedline, 0:index_max]

wpoint = index_max // 4
wpoint = 15  # working point selected
working_points = [3, 4, 5, 6, 8, 10, 12, 14]
# working_points = [0,2]

poles_global = {}  # dictionary for the whole set of poles
for wpoint in working_points:
    print('Working Point: ' + str(wpoint) + ' of ' + str(working_points[-1]))


    def centrifugal_vaneless(s, n, theta=0):
        # m1 = np.linalg.inv(Trad_n(r4, r4, n, s, Q[wpoint], GAMMA[wpoint], 0))
        m2 = Bvlsd_n(s, n, r2, r4, r2, Q[wpoint], GAMMA[wpoint], 0)
        m3 = Bimp_n(s, n, vx1[wpoint], vr2[wpoint], vy1[wpoint], vy2[wpoint], alpha1[wpoint], beta1[wpoint],
                    beta2[wpoint], r1, r2, rho1[wpoint], rho2[wpoint], A1_blade, A2_blade, s_i, dLi_dTanb[wpoint], 0, 0)
        m4 = Tax_n(x1, s, n, vx1[0], vy1[0], 0)
        m_res = np.linalg.multi_dot([m2, m3, m4])
        # EC = np.array([[Trad_n(r4, r4, n, s, Q[wpoint], GAMMA[wpoint], 0)[2,0],
        #                 Trad_n(r4, r4, n, s, Q[wpoint], GAMMA[wpoint], 0)[2,1],
        #                 Trad_n(r4, r4, n, s, Q[wpoint], GAMMA[wpoint], 0)[2,2]]])
        EC = np.array([[0, 0, 1]])
        IC = np.array([[0, 1, 0],
                       [0, 0, 1]])
        Y = np.concatenate((np.matmul(EC, m_res), IC))
        return np.linalg.det(Y)


    domain = [-3, 1.5, -10, 10]
    grid = [1, 1]
    n = np.arange(1, 5)
    poles = {}  # dictionary of poles for a single working point
    fig, ax = plt.subplots(1, 2, figsize=format_fig)

    for nn in n:
        print('Harmonic Number: ' + str(nn) + ' of ' + str(n[-1]))
        poles[nn] = Shot_Gun(centrifugal_vaneless, domain, grid, n=nn, attempts=30, N=30)
        plt.plot(poles[nn].real, -poles[nn].imag, 'o', label='n ' + str(nn))
    poles_global[wpoint] = poles  # for every working point attach the poles to the big dictionary of all poles
    real_axis_x = np.linspace(domain[0], domain[1], 100)
    real_axis_y = np.zeros(len(real_axis_x))
    imag_axis_y = np.linspace(domain[2], domain[3], 100)
    imag_axis_x = np.zeros(len(imag_axis_y))
    ax[1].plot(real_axis_x, real_axis_y, '--k', linewidth=0.5)
    ax[1].plot(imag_axis_x, imag_axis_y, '--k', linewidth=0.5)
    ax[1].set_xlim([domain[0], domain[1]])
    ax[1].set_ylim([domain[2], domain[3]])
    ax[1].legend()
    ax[1].set_xlabel(r'$\sigma_{n}$')
    ax[1].set_ylabel(r'$j \omega_{n}$')
    ax[1].set_title('root locus, operating point: ' + str(wpoint))
    # plt.savefig('pics/poles_iris_compressor_sl_'+str(speedline)+'_'+str(wpoint)+'.png')

    # speedline plot
    ax[0].plot(phi, beta_ts[speedline, 0:index_max], label='rpm ' + str(int(rpm[speedline])))
    ax[0].plot(phi[wpoint], beta_ts[speedline, wpoint], 'ro', label='op. point: ' + str(wpoint))
    ax[0].set_ylabel(r'$\beta_{ts}$')
    ax[0].set_xlabel(r'$\phi$')
    ax[0].set_title('operating point: ' + str(wpoint))
    fig.savefig('pics/root_locus_' + str(speedline) + '_op_' + str(wpoint) + '.pdf', bbox_inches='tight')

# %% General plots
# plot of characteristics
fig, ax = plt.subplots(1, figsize=(8, 6))
for s in range(0, len(rpm)):
    speedline = s  # choose the speedline to be used
    index_max = np.where(mass_flow[speedline, :] == 0)
    index_max = index_max[0]
    index_max = index_max[0]
    index_max = index_max - 1  # index max in order to avoid the choked data
    ax.plot(mass_flow[speedline, 0:index_max], beta_ts[speedline, 0:index_max], label='%0d krpm' % (rpm[speedline] / 1000))
ax.set_ylabel(r'$\beta_{ts}$')
ax.set_xlabel(r'$\dot{m}$')
ax.set_title('compressor characteristics')
ax.plot(mass_flow[:, 0], beta_ts[:, 0], 'k^', label='Senoo')
ax.plot(mass_flow[:, 10], beta_ts[:, 10], 'ko', label='Spakovszky')  # instability point, visually located
ax.legend()
fig.savefig('pics/compressor_characteristics.png')

# plot of efficiency
fig, ax = plt.subplots(1, figsize=(8, 6))
for s in range(0, len(rpm)):
    speedline = s  # choose the speedline to be used
    index_max = np.where(mass_flow[speedline, :] == 0)
    index_max = index_max[0]
    index_max = index_max[0]
    index_max = index_max - 1  # index max in order to avoid the choked data
    ax.plot(mass_flow[speedline, 0:index_max], eta_ts[speedline, 0:index_max], label='%0d krpm' % (rpm[speedline] / 1000))
ax.set_ylabel(r'$\eta_{ts}$')
ax.set_xlabel(r'$\dot{m}$')
ax.set_title('compressor characteristics')
ax.plot(mass_flow[:, 0], eta_ts[:, 0], 'k^', label='Senoo')
ax.plot(mass_flow[:, 10], eta_ts[:, 10], 'ko', label='Spakovszky')  # instability point, visually located
ax.legend()
fig.savefig('pics/compressor_efficiencies.png')

# %%plot the characteristics for impeller and diffuser to analyze the slopes
OMEGA = 2 * np.pi * rpm / 60
U1 = OMEGA * R1
U2 = OMEGA * R2
phi_all = np.zeros(mass_flow.shape)
p1_t_all = np.zeros(mass_flow.shape)
p2_t_all = np.zeros(mass_flow.shape)
p4_t_all = np.zeros(mass_flow.shape)
PSI_ts_imp = np.zeros(mass_flow.shape)
PSI_ss_diff = np.zeros(mass_flow.shape)
PSI_tt_diff = np.zeros(mass_flow.shape)
PSI_tt_imp = np.zeros(mass_flow.shape)
for i in range(0, len(U)):
    phi_all[i, :] = Wm1[i, :] / U2[i]
    p1_t_all[i, :] = P1[i, :] + 0.5 * Rho1[i, :] * (Wm1[i, :] ** 2 + (Wt1[i, :] + U1[i]) ** 2)
    p2_t_all[i, :] = P2[i, :] + 0.5 * Rho2[i, :] * (Wm2[i, :] ** 2 + (Wt2[i, :] + U2[i]) ** 2)
    p4_t_all[i, :] = P4[i, :] + 0.5 * Rho4[i, :] * (Vm4[i, :] ** 2 + Vt4[i, :])
    PSI_ts_imp[i, :] = (P2[i, :] - p1_t_all[i, :]) / ((Rho1[i, :] + Rho2[i, :]) / 2 * U2[i] ** 2)
    PSI_tt_imp[i, :] = (p2_t_all[i, :] - p1_t_all[i, :]) / ((Rho1[i, :] + Rho2[i, :]) / 2 * U2[i] ** 2)
    PSI_ss_diff[i, :] = (P4[i, :] - P2[i, :]) / ((Rho2[i, :] + Rho4[i, :]) / 2 * U2[i] ** 2)
    PSI_ts_diff[i, :] = (P4[i, :] - p2_t_all[i, :]) / ((Rho2[i, :] + Rho4[i, :]) / 2 * U2[i] ** 2)
    PSI_tt_diff[i, :] = (p4_t_all[i, :] - p2_t_all[i, :]) / ((Rho2[i, :] + Rho4[i, :]) / 2 * U2[i] ** 2)
    # the warning occurs because we are dividing by density, which is zero after choking conditions, but is not important
    # since we are plotting before that.

# plot of characteristics for impeller and diffuser
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for s in range(0, len(rpm)):
    speedline = s
    index_max = np.where(mass_flow[speedline, :] == 0)
    index_max = index_max[0]
    index_max = index_max[0]
    index_max = index_max - 1  # index max in order to avoid the choked data
    ax[0].plot(phi_all[speedline, 0:index_max], PSI_tt_imp[speedline, 0:index_max] + PSI_tt_diff[speedline, 0:index_max])
    ax[1].plot(phi_all[speedline, 0:index_max], PSI_tt_imp[speedline, 0:index_max])
    ax[2].plot(phi_all[speedline, 0:index_max], PSI_tt_diff[speedline, 0:index_max], label='%0d krpm' % (rpm[speedline] / 1000))
ax[0].plot(phi_all[:, 0], PSI_tt_imp[:, 0] + PSI_tt_diff[:, 0], 'k^', label='Senoo')
ax[0].plot(phi_all[:, 10], PSI_tt_imp[:, 10] + PSI_tt_diff[:, 10], 'ko', label='Spakovszky')
ax[1].plot(phi_all[:, 0], PSI_tt_imp[:, 0], 'k^')
ax[1].plot(phi_all[:, 10], PSI_tt_imp[:, 10], 'ko')
ax[2].plot(phi_all[:, 0], PSI_tt_diff[:, 0], 'k^')
ax[2].plot(phi_all[:, 10], PSI_tt_diff[:, 10], 'ko')
ax[0].set_ylabel(r'$\psi_{tt}$')
ax[0].set_xlabel(r'$\phi$')
ax[0].set_title('full stage')
ax[1].set_ylabel(r'$\psi_{tt}$')
ax[1].set_xlabel(r'$\phi$')
ax[1].set_title('impeller')
ax[2].set_ylabel(r'$\psi_{tt}$')
ax[2].set_xlabel(r'$\phi$')
ax[2].set_title('diffuser')
fig.legend()
fig.savefig('pics/stage_tt_characteristics.png')
