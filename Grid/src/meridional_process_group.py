#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:07:05 2023
@author: F. Neri, TU Delft
"""

import numpy as np
from .styles import *
import pickle


class MeridionalProcessGroup:
    """
    Group of meridional Process object, used to plot the full machine data together
    """

    def __init__(self):
        self.group = []


    def add_to_group(self, meridional_obj):
        """
        add component to the group, follow streamwise order
        """
        self.group.append(meridional_obj)


    def assemble_fields(self):
        """
        assemble together the fields contained in all the blocks
        """
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



    def assemble_field_gradients(self):
        """
        assemble together the gradients of the various blocks
        """
        self.drho_dr = self.group[0].drho_dr
        self.drho_dz = self.group[0].drho_dz
        self.dur_dr = self.group[0].dur_dr
        self.dur_dz = self.group[0].dur_dz
        self.dut_dr = self.group[0].dut_dr
        self.dut_dz = self.group[0].dut_dz
        self.duz_dr = self.group[0].duz_dr
        self.duz_dz = self.group[0].duz_dz
        self.dp_dr = self.group[0].dp_dr
        self.dp_dz = self.group[0].dp_dz

        for obj in self.group[1:]:
            self.drho_dr = np.concatenate((self.drho_dr, obj.drho_dr), axis=0)
            self.drho_dz = np.concatenate((self.drho_dz, obj.drho_dz), axis=0)
            self.dur_dr = np.concatenate((self.dur_dr, obj.dur_dr), axis=0)
            self.dur_dz = np.concatenate((self.dur_dz, obj.dur_dz), axis=0)
            self.dut_dr = np.concatenate((self.dut_dr, obj.dut_dr), axis=0)
            self.dut_dz = np.concatenate((self.dut_dz, obj.dut_dz), axis=0)
            self.duz_dr = np.concatenate((self.duz_dr, obj.duz_dr), axis=0)
            self.duz_dz = np.concatenate((self.duz_dz, obj.duz_dz), axis=0)
            self.dp_dr = np.concatenate((self.dp_dr, obj.dp_dr), axis=0)
            self.dp_dz = np.concatenate((self.dp_dz, obj.dp_dz), axis=0)



    def contour_fields(self, save_filename=None):
        """
        contour of the fields. Dimensional quantities
        """

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



    def contour_field_gradients(self, save_filename=None):
        """
        contours of the gradients, non-dimensional quantities
        """

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.drho_dr, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial \rho / \partial r \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_drho_dr.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.drho_dz, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial \rho / \partial {z} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_drho_dz.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.dur_dr, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_r / \partial r \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_dur_dr.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.dur_dz, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_r / \partial {z} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_dur_dz.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.dut_dr, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_{\theta} / \partial r \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_dut_dr.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.dut_dz, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_{\theta} / \partial {z} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_dut_dz.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.duz_dr, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_z / \partial r \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_duz_dr.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.duz_dz, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_z / \partial {z} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_duz_dz.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.dp_dr, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial p / \partial r \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_dp_dr.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_grid, self.r_grid, self.dp_dz, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial p / \partial {z} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_dp_dz.pdf', bbox_inches='tight')


