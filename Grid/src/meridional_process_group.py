#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:07:05 2023
@author: F. Neri, TU Delft
"""

import numpy as np
from .styles import *
from scipy.ndimage import gaussian_filter
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
        self.z_cg = self.group[0].z_cg
        self.r_cg = self.group[0].r_cg
        self.rho = self.group[0].rho
        self.ur = self.group[0].ur
        self.ut = self.group[0].ut
        self.uz = self.group[0].uz
        self.p = self.group[0].p
        self.T = self.group[0].T
        self.s = self.group[0].s
        self.M = self.group[0].M

        for obj in self.group[1:]:
            self.z_cg = np.concatenate((self.z_cg, obj.z_cg), axis=0)
            self.r_cg = np.concatenate((self.r_cg, obj.r_cg), axis=0)
            self.rho = np.concatenate((self.rho, obj.rho), axis=0)
            self.ur = np.concatenate((self.ur, obj.ur), axis=0)
            self.ut = np.concatenate((self.ut, obj.ut), axis=0)
            self.uz = np.concatenate((self.uz, obj.uz), axis=0)
            self.p = np.concatenate((self.p, obj.p), axis=0)
            self.T = np.concatenate((self.T, obj.T), axis=0)
            self.s = np.concatenate((self.s, obj.s), axis=0)
            self.M = np.concatenate((self.M, obj.M), axis=0)



    def assemble_fields_2(self):
        """
        assemble together the fields contained in all the blocks, but superposing the cordinates and fields,
        to avoid having two points coincidents, with two different field values
        """
        self.z_cg = self.group[0].z_cg[0:-1, :]
        self.r_cg = self.group[0].r_cg[0:-1, :]
        self.rho = self.group[0].rho[0:-1, :]
        self.ur = self.group[0].ur[0:-1, :]
        self.ut = self.group[0].ut[0:-1, :]
        self.uz = self.group[0].uz[0:-1, :]
        self.p = self.group[0].p[0:-1, :]
        self.T = self.group[0].T[0:-1, :]
        self.s = self.group[0].s[0:-1, :]
        self.M = self.group[0].M[0:-1, :]

        for obj in self.group[1:]:
            self.z_cg = np.concatenate((self.z_cg, obj.z_cg[0:-1, :]), axis=0)
            self.r_cg = np.concatenate((self.r_cg, obj.r_cg[0:-1, :]), axis=0)
            self.rho = np.concatenate((self.rho, obj.rho[0:-1, :]), axis=0)
            self.ur = np.concatenate((self.ur, obj.ur[0:-1, :]), axis=0)
            self.ut = np.concatenate((self.ut, obj.ut[0:-1, :]), axis=0)
            self.uz = np.concatenate((self.uz, obj.uz[0:-1, :]), axis=0)
            self.p = np.concatenate((self.p, obj.p[0:-1, :]), axis=0)
            self.T = np.concatenate((self.T, obj.T[0:-1, :]), axis=0)
            self.s = np.concatenate((self.s, obj.s[0:-1, :]), axis=0)
            self.M = np.concatenate((self.M, obj.M[0:-1, :]), axis=0)


    def gauss_filtering(self):
        self.rho = self.apply_gaussian_filter(self.rho)
        self.ur = self.apply_gaussian_filter(self.ur)
        self.ut = self.apply_gaussian_filter(self.ut)
        self.uz = self.apply_gaussian_filter(self.uz)
        self.p = self.apply_gaussian_filter(self.p)
        self.T = self.apply_gaussian_filter(self.T)
        self.s = self.apply_gaussian_filter(self.s)
        self.M = self.apply_gaussian_filter(self.M)

    def gauss_filtering_gradients(self):
        self.drho_dr = self.apply_gaussian_filter(self.drho_dr)
        self.drho_dz = self.apply_gaussian_filter(self.drho_dz)
        self.dur_dr = self.apply_gaussian_filter(self.dur_dr)
        self.dur_dz = self.apply_gaussian_filter(self.dur_dz)
        self.dut_dr = self.apply_gaussian_filter(self.dut_dr)
        self.dut_dz = self.apply_gaussian_filter(self.dut_dz)
        self.duz_dr = self.apply_gaussian_filter(self.duz_dr)
        self.duz_dz = self.apply_gaussian_filter(self.duz_dz)
        self.dp_dr = self.apply_gaussian_filter(self.dp_dr)
        self.dp_dz = self.apply_gaussian_filter(self.dp_dz)



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


    def assemble_field_gradients_2(self):
        """
        assemble the gradients field, superposing the coincident nodes
        """
        self.drho_dr = self.group[0].drho_dr[0:-1, :]
        self.drho_dz = self.group[0].drho_dz[0:-1, :]
        self.dur_dr = self.group[0].dur_dr[0:-1, :]
        self.dur_dz = self.group[0].dur_dz[0:-1, :]
        self.dut_dr = self.group[0].dut_dr[0:-1, :]
        self.dut_dz = self.group[0].dut_dz[0:-1, :]
        self.duz_dr = self.group[0].duz_dr[0:-1, :]
        self.duz_dz = self.group[0].duz_dz[0:-1, :]
        self.dp_dr = self.group[0].dp_dr[0:-1, :]
        self.dp_dz = self.group[0].dp_dz[0:-1, :]

        for obj in self.group[1:]:
            self.drho_dr = np.concatenate((self.drho_dr, obj.drho_dr[0:-1, :]), axis=0)
            self.drho_dz = np.concatenate((self.drho_dz, obj.drho_dz[0:-1, :]), axis=0)
            self.dur_dr = np.concatenate((self.dur_dr, obj.dur_dr[0:-1, :]), axis=0)
            self.dur_dz = np.concatenate((self.dur_dz, obj.dur_dz[0:-1, :]), axis=0)
            self.dut_dr = np.concatenate((self.dut_dr, obj.dut_dr[0:-1, :]), axis=0)
            self.dut_dz = np.concatenate((self.dut_dz, obj.dut_dz[0:-1, :]), axis=0)
            self.duz_dr = np.concatenate((self.duz_dr, obj.duz_dr[0:-1, :]), axis=0)
            self.duz_dz = np.concatenate((self.duz_dz, obj.duz_dz[0:-1, :]), axis=0)
            self.dp_dr = np.concatenate((self.dp_dr, obj.dp_dr[0:-1, :]), axis=0)
            self.dp_dz = np.concatenate((self.dp_dz, obj.dp_dz[0:-1, :]), axis=0)



    def contour_fields(self, save_filename=None):
        """
        contour of the fields. Dimensional quantities
        """

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.rho*self.group[0].rho_ref, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\rho \ \mathrm{[kg/m^3]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_rho.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.ur*self.group[0].u_ref, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$u_r \ \mathrm{[m/s]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_ur.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, np.abs(self.ut)*self.group[0].u_ref, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$u_{\theta} \ \mathrm{[m/s]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_ut.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.uz*self.group[0].u_ref, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$u_{z} \ \mathrm{[m/s]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_uz.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.p*self.group[0].p_ref, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$p \ \mathrm{[Pa]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_p.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.T * self.group[0].T_ref, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$T \ \mathrm{[K]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_T.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.s * self.group[0].s_ref, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$s \ \mathrm{[kJ/kgK]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_s.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.M, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$M \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_M.pdf', bbox_inches='tight')

    def show_grid(self, save_filename=None):
        """
        contour of the grid. Non-Dimensional quantities
        """

        self.AR = (np.max(self.r_cg) - np.min(self.r_cg)) / \
                  (np.max(self.z_cg) - np.min(self.z_cg))
        self.picture_size = (7, 7 * self.AR)
        self.nstream = np.shape(self.z_cg)[0]
        self.nspan = np.shape(self.z_cg)[1]

        plt.figure(figsize=self.picture_size)
        for istream in range(0, self.nstream):
            plt.plot(self.z_cg[istream, :], self.r_cg[istream, :], lw=light_line_width, c='black')
        for ispan in range(0, self.nspan):
            plt.plot(self.z_cg[:, ispan], self.r_cg[:, ispan], lw=light_line_width, c='black')
        plt.xlabel(r'$z \ \mathrm{[-]}$')
        plt.ylabel(r'$r \ \mathrm{[-]}$')
        plt.title(r'$(%d \times %d)$' % (self.nstream, self.nspan))
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_grid.pdf', bbox_inches='tight')



    def contour_field_gradients(self, save_filename=None):
        """
        contours of the gradients, non-dimensional quantities
        """

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.drho_dr, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial \rho / \partial r \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_drho_dr.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.drho_dz, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial \rho / \partial {z} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_drho_dz.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.dur_dr, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_r / \partial r \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_dur_dr.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.dur_dz, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_r / \partial {z} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_dur_dz.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.dut_dr, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_{\theta} / \partial r \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_dut_dr.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.dut_dz, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_{\theta} / \partial {z} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_dut_dz.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.duz_dr, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_z / \partial r \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_duz_dr.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.duz_dz, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_z / \partial {z} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_duz_dz.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.dp_dr, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial p / \partial r \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_dp_dr.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.z_cg, self.r_cg, self.dp_dz, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial p / \partial {z} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_dp_dz.pdf', bbox_inches='tight')

    @staticmethod
    def apply_gaussian_filter(field, sigma=1.5):
        """
        Gaussian filtering of a 2D field, with a specified deviation (sigma). 2 was a good value
        """
        smoothed_array = np.copy(field)
        smoothed_array = gaussian_filter(smoothed_array, sigma=sigma)
        return smoothed_array



    def store_pickle(self, file_name=None, folder=None):
        """
        store the object conent in a pickle
        Args:
            file_name: name to store. if None, default one is selected
            folder: location to store. if None, default one is selected
        """
        if folder is None:
            folder = folder_meta_data_default
        if file_name is None:
            file_name = 'meridional_process_%d_%d.pickle' % (self.nstream, self.nspan)

        with open(folder + file_name + '.pickle', "wb") as file:
            pickle.dump(self, file)



