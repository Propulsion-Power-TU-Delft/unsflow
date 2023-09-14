#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:07:05 2023
@author: F. Neri, TU Delft
"""

import numpy as np
from numpy import sqrt
from .styles import *
import matplotlib.path as mplpath
from scipy.ndimage import gaussian_filter
from scipy.interpolate import Rbf
import pickle


class MeridionalProcessGroup:
    """
    Group of meridional Process object, used only to plot stuff together
    """

    def __init__(self):
        self.group = []




    def add_to_group(self, meridional_obj):
        self.group.append(meridional_obj)

    def assemble_fields(self):
        self.z_grid = self.group[0].z_grid
        self.r_grid = self.group[0].r_grid
        self.rho = self.group[0].rho
        self.ur = self.group[0].ur
        self.ut = self.group[0].ut
        self.uz = self.group[0].uz
        self.p = self.group[0].p
        self.T = self.group[0].T
        self.s = self.group[0].s
        self.M = self.group[0].M

        for obj in self.group[1:]:
            self.z_grid = np.concatenate((self.z_grid, obj.z_grid), axis=0)
            self.r_grid = np.concatenate((self.r_grid, obj.r_grid), axis=0)
            self.rho = np.concatenate((self.rho, obj.rho), axis=0)
            self.ur = np.concatenate((self.ur, obj.ur), axis=0)
            self.ut = np.concatenate((self.ut, obj.ut), axis=0)
            self.uz = np.concatenate((self.uz, obj.uz), axis=0)
            self.p = np.concatenate((self.p, obj.p), axis=0)
            self.T = np.concatenate((self.T, obj.T), axis=0)
            self.s = np.concatenate((self.s, obj.s), axis=0)
            self.M = np.concatenate((self.M, obj.M), axis=0)

    def contour(self, save_filename=None):

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.rho*self.group[0].data.rho_ref, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\rho \ \mathrm{[kg/m^3]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_rho.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.ur*self.group[0].data.u_ref, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$u_r \ \mathrm{[m/s]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_ur.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, np.abs(self.ut)*self.group[0].data.u_ref, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$u_{\theta} \ \mathrm{[m/s]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_ut.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.uz*self.group[0].data.u_ref, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$u_{z} \ \mathrm{[m/s]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_uz.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.p*self.group[0].data.p_ref, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$p \ \mathrm{[Pa]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_p.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.T * self.group[0].data.T_ref, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$T \ \mathrm{[K]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_T.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.s * self.group[0].data.s_ref, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$s \ \mathrm{[kJ/kgK]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_s.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.M, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$M \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_M.pdf', bbox_inches='tight')


