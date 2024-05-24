#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:07:05 2023
@author: F. Neri, TU Delft
"""
import os.path

import numpy as np
import matplotlib.pyplot as plt
from Utils.styles import *
from scipy.ndimage import gaussian_filter
import pickle
from Sun.src.general_functions import print_banner_begin, print_banner_end
from Grid.src.functions import compute_picture_size, print_object_memory_info
from Grid.src.polynomial_ls_regression import *


class MeridionalProcessGroup:
    """
    Group of meridional Process object, used to plot the full machine data together
    """

    def __init__(self, config):
        """
        Construct the object contaning all the meridional objects related to the subdomains.
        """
        self.config = config
        self.group = []
        self.domains_type = []
        self.nstream = 0
        self.nspan = 0
        self.items_number = 0

    def add_to_group(self, meridional_obj, delete_cfd_data=True):
        """
        add component to the group, following streamwise order.
        :param meridional_obj: meridional object to add
        :param delete_cfd_data: delete the cfd dataset to free up memory in the final pickle object
        """
        if delete_cfd_data:
            del meridional_obj.data
        self.group.append(meridional_obj)

        if len(self.group) == 0:
            self.nstream += meridional_obj.nstream
        else:
            self.nstream += meridional_obj.nstream - 1

        if self.nspan == 0:
            self.nspan = meridional_obj.nspan
        else:
            if self.nspan != meridional_obj.nspan:
                raise ValueError("Blocks with different number of spanwise points!")
            else:
                pass
        self.items_number += 1

    def assemble_fields(self):
        """
        Assemble together the fields contained in all the blocks.
        """
        self.GAMMA = self.config.get_fluid_gamma()

        print_banner_begin('GLOBAL REFERENCE QUANTITIES')
        print(f"{'Shaft Omega [rad/s]:':<{total_chars_mid}}{self.config.get_omega_shaft():>{total_chars_mid}.3f}")
        print(f"{'Reference Omega [rad/s]:':<{total_chars_mid}}{self.config.get_reference_omega():>{total_chars_mid}.3f}")
        print(f"{'Reference Density [kg/m3]:':<{total_chars_mid}}{self.config.get_reference_density():>{total_chars_mid}.3f}")
        print(f"{'Reference Length [m]:':<{total_chars_mid}}{self.config.get_reference_length():>{total_chars_mid}.3f}")
        print(f"{'Reference Velocity [m/s]:':<{total_chars_mid}}{self.config.get_reference_velocity():>{total_chars_mid}.3f}")
        print(f"{'Reference Pressure [Pa]:':<{total_chars_mid}}{self.config.get_reference_pressure():>{total_chars_mid}.3f}")
        print(f"{'Reference Time [s]:':<{total_chars_mid}}{self.config.get_reference_time():>{total_chars_mid}.6f}")
        print(f"{'Reference Temperature [K]:':<{total_chars_mid}}{self.config.get_reference_temperature():>{total_chars_mid}.3f}")
        print(f"{'Reference Entropy [J/kgK]:':<{total_chars_mid}}{self.config.get_reference_entropy():>{total_chars_mid}.3f}")
        print_banner_end()

        self.z_cg = self.group[0].z_cg
        self.r_cg = self.group[0].r_cg
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
        self.stwl = self.group[0].stream_line_length
        self.spwl = self.group[0].span_wise_length

        for kk , obj in enumerate(self.group[1:]):
            self.z_cg = np.concatenate((self.z_cg[1:, :], obj.z_cg[1:, :]), axis=0)
            self.r_cg = np.concatenate((self.r_cg[1:, :], obj.r_cg[1:, :]), axis=0)
            self.z_grid = np.concatenate((self.z_grid[1:, :], obj.z_grid[1:, :]), axis=0)
            self.r_grid = np.concatenate((self.r_grid[1:, :], obj.r_grid[1:, :]), axis=0)
            self.rho = np.concatenate((self.rho[1:, :], obj.rho[1:, :]), axis=0)
            self.ur = np.concatenate((self.ur[1:, :], obj.ur[1:, :]), axis=0)
            self.ut = np.concatenate((self.ut[1:, :], obj.ut[1:, :]), axis=0)
            self.uz = np.concatenate((self.uz[1:, :], obj.uz[1:, :]), axis=0)
            self.p = np.concatenate((self.p[1:, :], obj.p[1:, :]), axis=0)
            self.T = np.concatenate((self.T[1:, :], obj.T[1:, :]), axis=0)
            self.s = np.concatenate((self.s[1:, :], obj.s[1:, :]), axis=0)
            self.M = np.concatenate((self.M[1:, :], obj.M[1:, :]), axis=0)

            for ii in range(obj.z_cg.shape[0]):
                obj.stream_line_length[ii, :] = obj.stream_line_length[ii, :] + self.stwl[-1, :]

            self.stwl = np.concatenate((self.stwl[1:, :], obj.stream_line_length[1:, :]), axis=0)
            self.spwl = np.concatenate((self.spwl[1:, :], obj.span_wise_length[1:, :]), axis=0)

    def assemble_body_force_fields(self):
        """
        Assemble together the fields contained in all the blocks
        """
        self.S00 = self.group[0].S00
        self.S01 = self.group[0].S01
        self.S02 = self.group[0].S02
        self.S03 = self.group[0].S03
        self.S04 = self.group[0].S04

        self.S10 = self.group[0].S10
        self.S11 = self.group[0].S11
        self.S12 = self.group[0].S12
        self.S13 = self.group[0].S13
        self.S14 = self.group[0].S14

        self.S20 = self.group[0].S20
        self.S21 = self.group[0].S21
        self.S22 = self.group[0].S22
        self.S23 = self.group[0].S23
        self.S24 = self.group[0].S24

        self.S30 = self.group[0].S30
        self.S31 = self.group[0].S31
        self.S32 = self.group[0].S32
        self.S33 = self.group[0].S33
        self.S34 = self.group[0].S34

        self.S40 = self.group[0].S40
        self.S41 = self.group[0].S41
        self.S42 = self.group[0].S42
        self.S43 = self.group[0].S43
        self.S44 = self.group[0].S44

        for obj in self.group[1:]:
            self.S00 = np.concatenate((self.S00, obj.S00), axis=0)
            self.S01 = np.concatenate((self.S01, obj.S01), axis=0)
            self.S02 = np.concatenate((self.S02, obj.S02), axis=0)
            self.S03 = np.concatenate((self.S03, obj.S03), axis=0)
            self.S04 = np.concatenate((self.S04, obj.S04), axis=0)

            self.S10 = np.concatenate((self.S10, obj.S10), axis=0)
            self.S11 = np.concatenate((self.S11, obj.S11), axis=0)
            self.S12 = np.concatenate((self.S12, obj.S12), axis=0)
            self.S13 = np.concatenate((self.S13, obj.S13), axis=0)
            self.S14 = np.concatenate((self.S14, obj.S14), axis=0)

            self.S20 = np.concatenate((self.S20, obj.S20), axis=0)
            self.S21 = np.concatenate((self.S21, obj.S21), axis=0)
            self.S22 = np.concatenate((self.S22, obj.S22), axis=0)
            self.S23 = np.concatenate((self.S23, obj.S23), axis=0)
            self.S24 = np.concatenate((self.S24, obj.S24), axis=0)

            self.S30 = np.concatenate((self.S30, obj.S30), axis=0)
            self.S31 = np.concatenate((self.S31, obj.S31), axis=0)
            self.S32 = np.concatenate((self.S32, obj.S32), axis=0)
            self.S33 = np.concatenate((self.S33, obj.S33), axis=0)
            self.S34 = np.concatenate((self.S34, obj.S34), axis=0)

            self.S40 = np.concatenate((self.S40, obj.S40), axis=0)
            self.S41 = np.concatenate((self.S41, obj.S41), axis=0)
            self.S42 = np.concatenate((self.S42, obj.S42), axis=0)
            self.S43 = np.concatenate((self.S43, obj.S43), axis=0)
            self.S44 = np.concatenate((self.S44, obj.S44), axis=0)

    def gauss_filtering(self):
        """
        Artificially filter the fields contained in all the blocks
        """
        print("WARNING: the fields have been artificially filtered!")
        self.rho = self.apply_gaussian_filter(self.rho)
        self.ur = self.apply_gaussian_filter(self.ur)
        self.ut = self.apply_gaussian_filter(self.ut)
        self.uz = self.apply_gaussian_filter(self.uz)
        self.p = self.apply_gaussian_filter(self.p)
        self.T = self.apply_gaussian_filter(self.T)
        self.s = self.apply_gaussian_filter(self.s)
        self.M = self.apply_gaussian_filter(self.M)

    def gauss_filtering_gradients(self):
        """
        Artificially filter the field gradients contained in all the blocks
        """
        print("WARNING: the gradients have been artificially filtered!")
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
        Assemble together the gradients of the various blocks
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
        self.ds_dr = self.group[0].ds_dr
        self.ds_dz = self.group[0].ds_dz

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
            self.ds_dr = np.concatenate((self.ds_dr, obj.ds_dr), axis=0)
            self.ds_dz = np.concatenate((self.ds_dz, obj.ds_dz), axis=0)

    def contour_fields(self, save_filename=None, folder_name=None):
        """
        Contour of all the fields. Plotted as dimensional for convenience.
        :param save_filename: if you wish specify the prefix-names of the figure that will be saved.
        """

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_grid, self.r_grid, self.rho * self.config.get_reference_density(), cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\rho \ \mathrm{[kg/m^3]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_rho.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.ur * self.config.get_reference_velocity(), cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$u_r \ \mathrm{[m/s]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_ur.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.ut * self.config.get_reference_velocity(), cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$u_{\theta} \ \mathrm{[m/s]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_ut.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.uz * self.config.get_reference_velocity(), cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$u_{z} \ \mathrm{[m/s]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_uz.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.p * self.config.get_reference_pressure(), cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$p \ \mathrm{[Pa]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_p.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.T * self.config.get_reference_temperature(), cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$T \ \mathrm{[K]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_T.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.s * self.config.get_reference_entropy(), cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$s \ \mathrm{[J/kgK]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_s.pdf', bbox_inches='tight')
            # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.M, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$M \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_M.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.stwl, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$s_{stwl} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_stwl.pdf', bbox_inches='tight')

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.spwl, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$s_{spwl} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_spwl.pdf', bbox_inches='tight')

    def show_grid(self, save_filename=None, grid_centers=False, folder_name=None):
        """
        Show the outer grid lines. Non-Dimensional quantities.
        :param save_filename: if you wish specify the prefix-names of the figure that will be saved.
        :param grid_centers: if True shows also the grid central nodes
        """

        self.picture_size_blank, self.picture_size_contour = compute_picture_size(self.z_cg, self.r_cg)
        nstream = np.shape(self.z_grid)[0]
        nspan = np.shape(self.z_grid)[1]

        plt.figure(figsize=self.picture_size_blank)

        # external grid
        for istream in range(0, nstream):
            plt.plot(self.z_grid[istream, :], self.r_grid[istream, :], lw=light_line_width, c='black')
        for ispan in range(0, nspan):
            plt.plot(self.z_grid[:, ispan], self.r_grid[:, ispan], lw=light_line_width, c='black')

        # grid centers
        if grid_centers:
            for istream in range(0, nstream - 3):
                plt.plot(self.z_cg[istream, :], self.r_cg[istream, :], 'r.')
            for ispan in range(0, nspan - 3):
                plt.plot(self.z_cg[:, ispan], self.r_cg[:, ispan], 'r.')

        plt.xlabel(r'$z \ \mathrm{[-]}$')
        plt.ylabel(r'$r \ \mathrm{[-]}$')
        plt.title(r'$(%d \times %d)$' % (self.nstream, self.nspan))
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_grid.pdf', bbox_inches='tight')
            plt.close()

    def contour_field_gradients(self, save_filename=None, folder_name=None):
        """
        Contours of the gradients, multiplied by the dimensional factors.
        :param save_filename: if you wish specify the prefix-names of the figure that will be saved.
        """

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg,
                     self.drho_dr * self.config.get_reference_density() / self.config.get_reference_length(), cmap=color_map,
                     levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial \rho / \partial r \ \mathrm{[kg/m^4]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_drho_dr.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg,
                     self.drho_dz * self.config.get_reference_density() / self.config.get_reference_length(), cmap=color_map,
                     levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial \rho / \partial {z} \ \mathrm{[kg/m^4]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_drho_dz.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg,
                     self.dur_dr * self.config.get_reference_velocity() / self.config.get_reference_length(), cmap=color_map,
                     levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_r / \partial r \ \mathrm{[s^{-1}]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_dur_dr.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg,
                     self.dur_dz * self.config.get_reference_velocity() / self.config.get_reference_length(), cmap=color_map,
                     levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_r / \partial {z} \ \mathrm{[s^{-1}]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_dur_dz.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg,
                     self.dut_dr * self.config.get_reference_velocity() / self.config.get_reference_length(), cmap=color_map,
                     levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_{\theta} / \partial r \ \mathrm{[s^{-1}]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_dut_dr.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg,
                     self.dut_dz * self.config.get_reference_velocity() / self.config.get_reference_length(), cmap=color_map,
                     levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_{\theta} / \partial {z} \ \mathrm{[s^{-1}]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_dut_dz.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg,
                     self.duz_dr * self.config.get_reference_velocity() / self.config.get_reference_length(), cmap=color_map,
                     levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_z / \partial r \ \mathrm{[s^{-1}]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_duz_dr.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg,
                     self.duz_dz * self.config.get_reference_velocity() / self.config.get_reference_length(), cmap=color_map,
                     levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial u_z / \partial {z} \ \mathrm{[s^{-1}]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_duz_dz.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.dp_dr * self.config.get_reference_pressure() / self.config.get_reference_length(),
                     cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial p / \partial r \ \mathrm{[kg/m^2 s^2]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_dp_dr.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.dp_dz * self.config.get_reference_pressure() / self.config.get_reference_length(),
                     cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial p / \partial {z} \ \mathrm{[kg/m^2 s^2]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_dp_dz.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.ds_dr * self.config.get_reference_entropy() / self.config.get_reference_length(),
                     cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial s / \partial r \ \mathrm{[m/s^2 K]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_ds_dr.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.ds_dz * self.config.get_reference_entropy() / self.config.get_reference_length(),
                     cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\partial s / \partial {z} \ \mathrm{[m/s^2 K]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_ds_dz.pdf', bbox_inches='tight')  # plt.close()

    @staticmethod
    def apply_gaussian_filter(field, sigma=1.5):
        """
        Gaussian filtering of a 2D field.
        :param field: field to filter
        :param sigma: std deviation to filter with.
        """
        smoothed_array = np.copy(field)
        smoothed_array = gaussian_filter(smoothed_array, sigma=sigma)
        return smoothed_array

    def store_pickle(self, file_name=None, folder_name=None):
        """
        Store the object content in a pickle.
        :param file_name: name to store. if None, default one is selected
        :param folder_name: location to store. if None, default one is selected
        """

        if folder_name is None:
            folder_name = 'pickle'

        if file_name is None:
            file_name = 'meridional_field_%d_%d' % (self.nstream, self.nspan)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(os.path.join(folder_name, file_name) + '.pickle', "wb") as file:
            pickle.dump(self, file)

    # def compute_streamline_length(self):
    #     """
    #     Compute the length along each streamline. The dimensions are the same of the ones in the single meridional blocks.
    #     """
    #     self.nstream = self.z_grid.shape[0]
    #     self.nspan = self.z_grid.shape[1]
    #     self.stream_line_length = np.zeros_like(self.z_grid)
    #     for ispan in range(0, self.nspan):
    #         z = self.z_cg[:, ispan]
    #         r = self.r_cg[:, ispan]
    #         tmp_len = 0
    #         for istream in range(1, self.nstream):
    #             tmp_len += np.sqrt((z[istream] - z[istream - 1]) ** 2 + (r[istream] - r[istream - 1]) ** 2)
    #             self.stream_line_length[istream, ispan] = tmp_len
    #
    #     # reports to zero the first length of the bladed zone
    #     self.middle_line_length = self.stream_line_length[:, self.nspan // 2]
    #     self.middle_line_length /= self.group[1].stream_line_length[-1, self.group[1].nspan // 2]

    def compute_streamwise_normalized_length(self):
        """
        from the stwl data, compute the normalized length, from
        """
        self.nspan = self.z_cg.shape[1]
        self.stwl_norm = np.zeros_like(self.stwl)
        for jj in range(self.nspan):
            self.stwl_norm[:, jj] = self.stwl[:, jj]/self.stwl[:, jj].max()

    def compute_spanwise_normalized_length(self):
        """
        from the spwl data, compute the normalized length, from
        """
        self.nstream = self.z_cg.shape[0]
        self.spwl_norm = np.zeros_like(self.spwl)
        for ii in range(self.nstream):
            self.spwl_norm[ii, :] = self.spwl[ii, :]/self.spwl[ii, :].max()


    def plot_stream_line(self, field, n, save_filename=None, folder_name=None):
        """
        For the streamline n, plot the evolution of the flow field.
        :param field: field to plot
        :param n: streamline number
        :param save_filename: name of the figure to save
        """
        sl_max = self.stream_line_length[:, n].max()
        fig, ax = plt.subplots()
        if field == 'rho':
            ax.plot(self.stream_line_length[:, n] / sl_max, self.rho[:, n], '--s')
            ax.set_ylabel(r'$\rho \ \mathrm{[-]}$')
        elif field == 'ur':
            ax.plot(self.stream_line_length[:, n] / sl_max, self.ur[:, n], '--s')
            ax.set_ylabel(r'$u_r \ \mathrm{[-]}$')
        elif field == 'ut':
            ax.plot(self.stream_line_length[:, n] / sl_max, self.ut[:, n], '--s')
            ax.set_ylabel(r'$u_t \ \mathrm{[-]}$')
        elif field == 'uz':
            ax.plot(self.stream_line_length[:, n] / sl_max, self.uz[:, n], '--s')
            ax.set_ylabel(r'$u_z \ \mathrm{[-]}$')
        elif field == 'p':
            ax.plot(self.stream_line_length[:, n] / sl_max, self.p[:, n], '--s')
            ax.set_ylabel(r'$p \ \mathrm{[-]}$')
        elif field == 'T':
            ax.plot(self.stream_line_length[:, n] / sl_max, self.T[:, n], '--s')
            ax.set_ylabel(r'$T \ \mathrm{[-]}$')
        elif field == 's':
            ax.plot(self.stream_line_length[:, n] / sl_max, self.s[:, n], '--s')
            ax.set_ylabel(r'$s \ \mathrm{[-]}$')
        else:
            raise ValueError("Field name unknown!")

        ax.grid(alpha=0.3)
        ax.set_xlabel(r'$l \ \mathrm{[-]}$')
        if save_filename is not None:
            fig.savefig(folder_name + '/' + save_filename + '.pdf', bbox_inches='tight')
            plt.close()

    def plot_averaged_fluxes(self, field, save_filename=None, folder_name=None):
        """
        Plots the averaged fluxes along the streamline positions.
        :param field: field
        :param save_filename: name of the figures to be saved
        """
        old_value = 0
        reference_point = self.middle_line_length[self.group[0].nstream]  # initial reference is the leading edge of the blade
        fig, ax = plt.subplots()
        for obj in self.group:
            begin = old_value  # begin index
            end = begin + obj.nstream  # end index
            x = self.middle_line_length[begin:end] - reference_point  # 0 is at leading edge of blade
            old_value = end

            # plots
            if field == 'rho':
                ax.plot(x, obj.rho_flux * self.config.get_reference_density(), '--s', linewidth=light_line_width,
                        markersize=marker_size_small)
                ax.set_ylabel(r'$\rho \ \mathrm{[kg/m^3]}$')
            elif field == 'ur':
                ax.plot(x, obj.ur_flux * self.config.get_reference_velocity(), '--s', linewidth=light_line_width,
                        markersize=marker_size_small)
                ax.set_ylabel(r'$u_r \ \mathrm{[m/s]}$')
            elif field == 'ut':
                ax.plot(x, obj.ut_flux * self.config.get_reference_velocity(), '--s', linewidth=light_line_width,
                        markersize=marker_size_small)
                ax.set_ylabel(r'$u_t \ \mathrm{[m/s]}$')
            elif field == 'uz':
                ax.plot(x, obj.uz_flux * self.config.get_reference_velocity(), '--s', linewidth=light_line_width,
                        markersize=marker_size_small)
                ax.set_ylabel(r'$u_z \ \mathrm{[m/s]}$')
            elif field == 'p':
                ax.plot(x, obj.p_flux * self.config.get_reference_pressure(), '--s', linewidth=light_line_width,
                        markersize=marker_size_small)
                ax.set_ylabel(r'$p \ \mathrm{[Pa]}$')
            elif field == 'T':
                ax.plot(x, obj.T_flux * self.config.get_reference_temperature(), '--s', linewidth=light_line_width,
                        markersize=marker_size_small)
                ax.set_ylabel(r'$T \ \mathrm{[K]}$')
            elif field == 's':
                ax.plot(x, obj.s_flux * self.config.get_reference_entropy(), '--s', linewidth=light_line_width,
                        markersize=marker_size_small)
                ax.set_ylabel(r'$s \ \mathrm{[J/kgK]}$')
            elif field == 'p_tot':
                ax.plot(x, obj.p_tot_flux * self.config.get_reference_pressure(), '--s', linewidth=light_line_width,
                        markersize=marker_size_small)
                ax.set_ylabel(r'$p_t \ \mathrm{[Pa]}$')
            elif field == 'T_tot':
                ax.plot(x, obj.T_tot_flux * self.config.get_reference_temperature(), '--s', linewidth=light_line_width,
                        markersize=marker_size_small)
                ax.set_ylabel(r'$T_t \ \mathrm{[K]}$')
            elif field == 'M':
                ax.plot(x, obj.M_flux, '--s', linewidth=light_line_width, markersize=marker_size_small)
                ax.set_ylabel(r'$M \ \mathrm{[-]}$')
            elif field == 'M_rel':
                ax.plot(x, obj.M_rel_flux, '--s', linewidth=light_line_width, markersize=marker_size_small)
                ax.set_ylabel(r'$M_{rel} \ \mathrm{[-]}$')
            else:
                raise ValueError("Field name unknown!")

            ax.grid(alpha=0.3)
            ax.set_xlabel(r'$l \ \mathrm{[-]}$')
            if save_filename is not None:
                fig.savefig(folder_name + '/' + save_filename + '_flux_' + field + '.pdf', bbox_inches='tight')  # plt.close()

    def contour_plot(self, field, save_filename=None, quiver=False, folder_name='pictures'):
        """
        Contour plot of a 2D field.
        :param field: field to plot
        :param save_filename: if specified, saves the figure
        :param quiver: if True, superposes the quiver plots of the meridional velocity
        """
        fig, ax = plt.subplots(figsize=self.picture_size_contour)

        if field == 'rho':
            cs = ax.contourf(self.z_cg, self.r_cg, self.rho, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{\rho}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ur':
            cs = ax.contourf(self.z_cg, self.r_cg, self.ur, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{u}_r$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ut':
            cs = ax.contourf(self.z_cg, self.r_cg, self.ut, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{u}_{\theta}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'uz':
            cs = ax.contourf(self.z_cg, self.r_cg, self.uz, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{u}_z$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'p':
            cs = ax.contourf(self.z_cg, self.r_cg, self.p, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{p}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 's':
            cs = ax.contourf(self.z_cg, self.r_cg, self.s, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{s}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'T':
            cs = ax.contourf(self.z_cg, self.r_cg, self.T, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{T}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'M':
            cs = ax.contourf(self.z_cg, self.r_cg, self.M, N_levels, cmap=color_map)
            ax.set_title(r'$M$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'um':
            cs = ax.contourf(self.z_cg, self.r_cg, self.u_meridional, N_levels, cmap=color_map)
            ax.set_title(r'$u_m$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'drho_dr':
            cs = ax.contourf(self.z_cg, self.r_cg, self.drho_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{\rho} / \partial \hat{r}$')
            ax.contour(self.z_cg, self.r_cg, self.drho_dr, levels=[0], colors='white', linestyles='dashed', linewidths=2)
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'drho_dz':
            cs = ax.contourf(self.z_cg, self.r_cg, self.drho_dz, N_levels, cmap=color_map)
            ax.contour(self.z_cg, self.r_cg, self.drho_dz, levels=[0], colors='white', linestyles='dashed', linewidths=2)
            ax.set_title(r'$\partial \hat{\rho} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dur_dr':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dur_dr, N_levels, cmap=color_map)
            ax.contour(self.z_cg, self.r_cg, self.dur_dr, levels=[0], colors='white', linestyles='dashed', linewidths=2)
            ax.set_title(r'$\partial \hat{u}_r / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dur_dz':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dur_dz, N_levels, cmap=color_map)
            ax.contour(self.z_cg, self.r_cg, self.dur_dz, levels=[0], colors='white', linestyles='dashed', linewidths=2)
            ax.set_title(r'$\partial \hat{u}_r / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dut_dr':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dut_dr, N_levels, cmap=color_map)
            ax.contour(self.z_cg, self.r_cg, self.dut_dr, levels=[0], colors='white', linestyles='dashed', linewidths=2)
            ax.set_title(r'$\partial \hat{u}_{\theta} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dut_dz':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dut_dz, N_levels, cmap=color_map)
            ax.contour(self.z_cg, self.r_cg, self.dut_dz, levels=[0], colors='white', linestyles='dashed', linewidths=2)
            ax.set_title(r'$\partial \hat{u}_{\theta} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'duz_dr':
            cs = ax.contourf(self.z_cg, self.r_cg, self.duz_dr, N_levels, cmap=color_map)
            ax.contour(self.z_cg, self.r_cg, self.duz_dr, levels=[0], colors='white', linestyles='dashed', linewidths=2)
            ax.set_title(r'$\partial \hat{u}_{z} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'duz_dz':
            cs = ax.contourf(self.z_cg, self.r_cg, self.duz_dz, N_levels, cmap=color_map)
            ax.contour(self.z_cg, self.r_cg, self.duz_dz, levels=[0], colors='white', linestyles='dashed', linewidths=2)
            ax.set_title(r'$\partial \hat{u}_{z} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dp_dr':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dp_dr, N_levels, cmap=color_map)
            ax.contour(self.z_cg, self.r_cg, self.dp_dr, levels=[0], colors='white', linestyles='dashed', linewidths=2)
            ax.set_title(r'$\partial \hat{p} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dp_dz':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dp_dz, N_levels, cmap=color_map)
            ax.contour(self.z_cg, self.r_cg, self.dp_dz, levels=[0], colors='white', linestyles='dashed', linewidths=2)
            ax.set_title(r'$\partial \hat{p} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ds_dr':
            cs = ax.contourf(self.z_cg, self.r_cg, self.ds_dr, N_levels, cmap=color_map)
            ax.contour(self.z_cg, self.r_cg, self.ds_dr, levels=[0], colors='white', linestyles='dashed', linewidths=2)
            ax.set_title(r'$\partial \hat{s} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ds_dz':
            cs = ax.contourf(self.z_cg, self.r_cg, self.ds_dz, N_levels, cmap=color_map)
            ax.contour(self.z_cg, self.r_cg, self.ds_dz, levels=[0], colors='white', linestyles='dashed', linewidths=2)
            ax.set_title(r'$\partial \hat{s} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ut_rel':
            cs = ax.contourf(self.z_cg, self.r_cg, self.ut_rel, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{w}_{\theta}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ut_drag':
            cs = ax.contourf(self.z_cg, self.r_cg, self.ut_drag, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{v}_{\theta}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'k':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_cg, self.r_cg, self.k, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{k}$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'F_ntheta':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_cg, self.r_cg, self.F_ntheta, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{F}_{n \theta}$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'F_nr':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_cg, self.r_cg, self.F_nr, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{F}_{n r}$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'F_nz':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_cg, self.r_cg, self.F_nz, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{F}_{n z}$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'a1':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_cg, self.r_cg, self.a1, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{a}_1$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'a2':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_cg, self.r_cg, self.a2, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{a}_2$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'a3':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_cg, self.r_cg, self.a3, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{a}_3$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'streamline length':
            cs = ax.contourf(self.z_cg, self.r_cg, self.stream_line_length,
                             levels=N_levels, cmap=color_map)
            ax.set_title(r'streamline length')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'mu':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_cg, self.r_cg, self.mu,
                                 levels=N_levels, cmap=color_map)
                ax.set_title(r'$\hat{\mu}$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'F_t':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_cg, self.r_cg, self.F_t, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{F}_t$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'F_t quiver':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_cg, self.r_cg, self.F_t, N_levels, cmap=color_map)
                ax.quiver(self.z_cg, self.r_grid, -self.uz, -self.ur)
                ax.set_title(r'$\hat{F}_t$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'F_n':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_cg, self.r_cg, np.sqrt(self.F_nr ** 2 + self.F_ntheta ** 2 +
                                                               self.F_nz ** 2), N_levels, cmap=color_map)
                ax.set_title(r'$\hat{F}_n$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'p_tot':
            cs = ax.contourf(self.z_cg, self.r_cg, self.p_tot, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{p}_{t}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'T_tot':
            cs = ax.contourf(self.z_cg, self.r_cg, self.T_tot, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{T}_{t}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'p_tot_bar':
            cs = ax.contourf(self.z_cg, self.r_cg, self.p_tot_bar, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{\bar{p}}_{t}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        else:
            raise Exception('Choose a valid contour plot data!')
        # cb = fig.colorbar(cs)
        ax.set_xlabel(r'$\hat{z} \ \mathrm{[-]}$')
        ax.set_ylabel(r'$\hat{r} \ \mathrm{[-]}$')
        axx = fig.gca()
        axx.set_aspect('equal')
        if quiver:
            ax.quiver(self.z_cg, self.r_cg, self.uz, self.ur)
        if save_filename is not None:
            fig.savefig(folder_name + '/' + save_filename + '.pdf', bbox_inches='tight')

    def compute_performance(self):
        """
        Compute pressure ratio and efficiency given the averaged fluxes of total quantities.
        """
        self.nstream = np.shape(self.z_cg)[0]
        self.nspan = np.shape(self.z_cg)[1]
        GMMA = self.GAMMA

        self.T1 = self.group[0].T_flux[0]
        self.Tt1 = self.group[0].T_tot_flux[0]
        self.T2 = self.group[-1].T_flux[-1]
        self.Tt2 = self.group[-1].T_tot_flux[-1]

        self.P1 = self.group[0].p_flux[0]
        self.Pt1 = self.group[0].p_tot_flux[0]
        self.P2 = self.group[-1].p_flux[-1]
        self.Pt2 = self.group[-1].p_tot_flux[-1]

        self.beta_ss = self.P2 / self.P1
        self.beta_ts = self.P2 / self.Pt1
        self.beta_tt = self.Pt2 / self.Pt1

        self.eta_ts = ((self.P2 / self.Pt1) ** ((GMMA - 1) / GMMA) - 1) / (self.Tt2 / self.Tt1 - 1)
        self.eta_tt = ((self.Pt2 / self.Pt1) ** ((GMMA - 1) / GMMA) - 1) / (self.Tt2 / self.Tt1 - 1)

    def print_performance(self):
        """
        print on terminal the performance of the machine. only total to total
        """
        print_banner_begin('MACHINE PERFORMANCE')
        print(f"{'Inlet P [Pa]:':<{total_chars_mid}}{self.P1 * self.config.get_reference_pressure():>{total_chars_mid}.2f}")
        print(f"{'Inlet Pt [Pa]:':<{total_chars_mid}}{self.Pt1 * self.config.get_reference_pressure():>{total_chars_mid}.2f}")
        print(f"{'Outlet P [Pa]:':<{total_chars_mid}}{self.P2 * self.config.get_reference_pressure():>{total_chars_mid}.2f}")
        print(f"{'Outlet Pt [Pa]:':<{total_chars_mid}}{self.Pt2 * self.config.get_reference_pressure():>{total_chars_mid}.2f}")
        print(f"{'Inlet T [K]:':<{total_chars_mid}}{self.T1 * self.config.get_reference_temperature():>{total_chars_mid}.2f}")
        print(f"{'Inlet Tt [K]:':<{total_chars_mid}}{self.Tt1 * self.config.get_reference_temperature():>{total_chars_mid}.2f}")
        print(f"{'Outlet T [K]:':<{total_chars_mid}}{self.T2 * self.config.get_reference_temperature():>{total_chars_mid}.2f}")
        print(f"{'Outlet Tt [K]:':<{total_chars_mid}}{self.Tt2 * self.config.get_reference_temperature():>{total_chars_mid}.2f}")
        print(f"{'Beta_ts:':<{total_chars_mid}}{self.beta_ts:>{total_chars_mid}.2f}")
        print(f"{'Beta_tt [-]:':<{total_chars_mid}}{self.beta_tt:>{total_chars_mid}.2f}")
        print(f"{'Eta_ts:':<{total_chars_mid}}{self.eta_ts:>{total_chars_mid}.2f}")
        print(f"{'Eta_tt [-]:':<{total_chars_mid}}{self.eta_tt:>{total_chars_mid}.2f}")
        print_banner_end()

    # def compose_global_sun_Omega_tau(self):
    #     """
    #     given Omega and tau of the Sun Model of the whole domain, build the global matrices, enlarged by a factor 5, that
    #     will be needed in the sun Model. The order of the point is for istream -> for ispan
    #     """
    #
    #     def enlarge_matrix_for_sun(Z):
    #         nstream = np.shape(Z)[0]
    #         nspan = np.shape(Z)[1]
    #
    #         nrows = nstream*nspan*5
    #         ncols = nrows
    #         enlarged_matrix = np.zeros((nrows, ncols))
    #
    #         irow = 0
    #         for i in range(nstream):
    #             for j in range(nspan):
    #                 block = np.zeros((5, ncols)) + Z[i, j]
    #                 enlarged_matrix[irow:irow+5, :] = block
    #                 irow +=5
    #
    #         return enlarged_matrix
    #
    #     # enlarge_matrix_for_sun(self.Omega)
    #     self.Omega_sun = enlarge_matrix_for_sun(self.Omega)
    #     self.tau_sun = enlarge_matrix_for_sun(self.tau)

    def shock_smoothing(self, i_shock, extension_points=4, blending_function='linear'):
        """
        Apply shock smoothing method to the interface between 2 blocks.
        :param i_shock: streamwise index of the shock/discontinuity position (upstream point of the discontinuity)
        :param extension_points: specify how many points upstream and downstream of the interface you want to 
        smooth, blended accordingly to the blending function
        :param blending_function: specify they type of blending functions to use
        """
        # compute the blending function, linear ramp from S = 1 at the leading edge, to S = 0 at 4 indexes 
        # upstream/downstream
        self.compute_blending_function(i_shock, extension_points, blending_function)

        # smooth the primary fields
        print("Smoothing the flow fields...")
        self.rho = self.smooth_field(self.rho)
        self.ur = self.smooth_field(self.ur)
        self.ut = self.smooth_field(self.ut)
        self.uz = self.smooth_field(self.uz)
        self.p = self.smooth_field(self.p)
        self.T = self.smooth_field(self.T)
        self.s = self.smooth_field(self.s)

        # smooth the gradients
        print("Smoothing the gradients...")
        self.drho_dr = self.smooth_field(self.drho_dr)
        self.dur_dr = self.smooth_field(self.dur_dr)
        self.dur_dz = self.smooth_field(self.dur_dz)
        self.duz_dr = self.smooth_field(self.duz_dr)
        self.duz_dz = self.smooth_field(self.duz_dz)
        self.dut_dr = self.smooth_field(self.dut_dr)
        self.dut_dz = self.smooth_field(self.dut_dz)
        self.dp_dr = self.smooth_field(self.dp_dr)
        self.dp_dz = self.smooth_field(self.dp_dz)

        # smooth the body force fields
        print("Smoothing the body force fields...")
        self.S00 = self.smooth_field(self.S00)
        self.S01 = self.smooth_field(self.S01)
        self.S02 = self.smooth_field(self.S02)
        self.S03 = self.smooth_field(self.S03)
        self.S04 = self.smooth_field(self.S04)

        self.S10 = self.smooth_field(self.S10)
        self.S11 = self.smooth_field(self.S11)
        self.S12 = self.smooth_field(self.S12)
        self.S13 = self.smooth_field(self.S13)
        self.S14 = self.smooth_field(self.S14)

        self.S20 = self.smooth_field(self.S20)
        self.S21 = self.smooth_field(self.S21)
        self.S22 = self.smooth_field(self.S22)
        self.S23 = self.smooth_field(self.S23)
        self.S24 = self.smooth_field(self.S24)

        self.S30 = self.smooth_field(self.S30)
        self.S31 = self.smooth_field(self.S31)
        self.S32 = self.smooth_field(self.S32)
        self.S33 = self.smooth_field(self.S33)
        self.S34 = self.smooth_field(self.S34)

        self.S40 = self.smooth_field(self.S40)
        self.S41 = self.smooth_field(self.S41)
        self.S42 = self.smooth_field(self.S42)
        self.S43 = self.smooth_field(self.S43)
        self.S44 = self.smooth_field(self.S44)

    def compute_blending_function(self, i_shock, extension_points, blending_function):
        """
        :param i_shock: streamwise index of the shock/discontinuity position
        :param extension_points: specify how many points upstream and downstream of the interface you want to 
        smooth, blended accordingly to the blending function
        :param blending_function: specify they type of blending functions to use
        """
        print_banner_begin("SMOOTHING PROCESS")
        print(f"{'Extension of smoothing:':<{total_chars_mid}}{extension_points:>{total_chars_mid}}")
        print(f"{'Blending function:':<{total_chars_mid}}{blending_function:>{total_chars_mid}}")
        print_banner_end()
        self.blending_function = np.zeros_like(self.rho)
        if blending_function == 'linear':
            for jj in range(self.nspan):
                self.blending_function[i_shock - extension_points:i_shock + 1, jj] = np.linspace(0, 1, extension_points + 1)
                self.blending_function[i_shock + 1:i_shock + 1 + extension_points + 1, jj] = np.linspace(1, 0,
                                                                                                         extension_points + 1)
        else:
            raise ValueError("Blending function not recognized.")

    def smooth_field(self, f, c=0.1, Nsc=15):
        """
        Smoothing Algorithm.
        :param f: field to smooth
        :param c: smoothing coefficient. Suggested value=0.1 from He et al. and Crouch et al.
        :param Nsc: number of smoothing cycles. Suggested value=15 from He et al.
        """
        f_original = f.copy()
        islice = slice(1, self.nstream - 1)
        islicep = slice(2, self.nstream)
        islicem = slice(0, self.nstream - 2)
        for _ in range(Nsc):
            f[islice, :] = f[islice, :] + 0.5 * c * (f[islicep, :] - 2 * f[islice, :] + f[islicem, :])
        f = (1 - self.blending_function) * f_original + self.blending_function * f
        return f

    def print_memory_info(self):
        """
        Print the information related to the memory needed by the object
        """
        print_object_memory_info(self)

    def contour_bfm_matrices(self, save_filename=None, folder_name=None):
        """
        Plots the S matrix elements of the BFM.
        :param save_filename: name of the figures to be saved
        """
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.S11, cmap=color_map, levels=N_levels)
        plt.title(r'$S_{11}$')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_S11_.pdf', bbox_inches='tight')  # plt.close()
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.S12, cmap=color_map, levels=N_levels)
        plt.title(r'$S_{12}$')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_S12_.pdf', bbox_inches='tight')  # plt.close()
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.S13, cmap=color_map, levels=N_levels)
        plt.title(r'$S_{13}$')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_S13_.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.S21, cmap=color_map, levels=N_levels)
        plt.title(r'$S_{21}$')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_S21_.pdf', bbox_inches='tight')  # plt.close()
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.S22, cmap=color_map, levels=N_levels)
        plt.title(r'$S_{22}$')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_S22_.pdf', bbox_inches='tight')  # plt.close()
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.S23, cmap=color_map, levels=N_levels)
        plt.title(r'$S_{23}$')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_S23_.pdf', bbox_inches='tight')  # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.S31, cmap=color_map, levels=N_levels)
        plt.title(r'$S_{31}$')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_S31_.pdf', bbox_inches='tight')  # plt.close()
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.S32, cmap=color_map, levels=N_levels)
        plt.title(r'$S_{32}$')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_S32_.pdf', bbox_inches='tight')  # plt.close()
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.S33, cmap=color_map, levels=N_levels)
        plt.title(r'$S_{33}$')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_S33_.pdf', bbox_inches='tight')  # plt.close()

    def compute_regressed_fields(self, order=4):
        """
        Compute the fourth order polynomial regressed fields, as described in the original papers
        :param order: order of the regression. 4 is the values used in the literature.
        """
        print("Regression of the Flow Fields, order: %i" % (order))
        if order != 4:
            raise ValueError("Choose the regression order equal to 4!")
        self.W = basis_function_matrix(self.z_cg, self.r_cg, order=order)
        self.W_dz, self.W_dr = basis_function_matrix_derivatives(self.W, self.z_cg, self.r_cg)

        self.rho, self.drho_dr, self.drho_dz = self.polynomial_regression_solution(self.rho)
        self.ur, self.dur_dr, self.dur_dz = self.polynomial_regression_solution(self.ur)
        self.ut, self.dut_dr, self.dut_dz = self.polynomial_regression_solution(self.ut)
        self.uz, self.duz_dr, self.duz_dz = self.polynomial_regression_solution(self.uz)
        self.p, self.dp_dr, self.dp_dz = self.polynomial_regression_solution(self.p)
        self.T, self.dT_dr, self.dT_dz = self.polynomial_regression_solution(self.T)
        self.s, self.ds_dr, self.ds_dz = self.polynomial_regression_solution(self.s)

    def polynomial_regression_solution(self, field):
        """
        Given a 2D field, and the weight vector coefficients, compute the values of the regressed field and derivatives.
        :param field: 2D array storing the values of the field to be regressed.
        """
        Nz = np.shape(self.z_cg)[0]
        Nr = np.shape(self.r_cg)[1]
        coeff_vector = least_square_regression(self.W, field)
        W = self.W
        W_dz, W_dr = self.W_dz, self.W_dr
        regr_field = regression_evaluation(W, coeff_vector, Nz, Nr)
        regr_field_dz = regression_evaluation(W_dz, coeff_vector, Nz, Nr)
        regr_field_dr = regression_evaluation(W_dr, coeff_vector, Nz, Nr)
        return regr_field, regr_field_dr, regr_field_dz
