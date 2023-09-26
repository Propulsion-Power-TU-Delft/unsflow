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


class MeridionalProcess:
    """
    class that contains a multiblock grid object, and the CFD data results. It performs the circumferential averaging.
    Important note: if the cfd data has been normalized, all the quantities are already non-dimensional except for the
    cordinates variables (related to r,z). In the future we will non-dimensionalize everything from the beginning.
    """

    def __init__(self, data, block=None, blade=None, verbose=False):
        self.data = data
        self.block = block
        if blade is not None:
            self.blade = blade
        self.nstream = block.nstream
        self.nspan = block.nspan
        self.nAxialNodes = block.nstream
        self.nRadialNodes = block.nspan
        self.verbose = verbose
        self.z_grid = block.z_grid_points  # primary grid points
        self.r_grid = block.r_grid_points
        self.z_grid_secondary = block.z_grid_centers  # secondary grid points
        self.r_grid_secondary = block.r_grid_centers

        self.rho_ref = data.rho_ref
        self.u_ref = data.u_ref
        self.T_ref = data.T_ref
        self.s_ref = data.s_ref
        self.x_ref = data.x_ref
        self.omega_ref = data.omega_ref
        self.p_ref = data.p_ref

    def circumferential_average(self, mode='rectangular', fix_borders=True, bfm=None, gauss_filter=True):
        """
        perform circumferential averages
        Args:
            mode: type of algorithm.
                rectangular: take all the points in the rectangle identified by the secondary grid.
                circular: take all the points inside a circle
            fix_borders: if True, the values on the borders are copied from the values in the inner nodes
            bfm: if True, enables calculation of BFM related quantities (depending on the type of BFM specified)
            gauss_filter: if True enables gauss filtering of the 2D fields, to smooth it down
        """
        self.bfm = bfm
        self.instantiate_2d_fields()
        if self.bfm == 'radial':
            self.instantiate_2d_bfm_fields()

        if self.verbose:
            print('performing circumferential averages...')

        # loop over all the elements in the meridional grid
        for istream in range(0, self.nstream):
            for ispan in range(0, self.nspan):
                idx = None
                if mode == 'rectangular':  # default mode of indexing
                    # use the secondary grid to identifying the scattered points in the meridional plane
                    n_elem = 0  # number of elements found in the rectangle. initialization
                    i = 0  # cycle counter
                    while n_elem == 0:
                        quadrilateral_path = self.find_rectangle(istream, ispan, i)
                        idx = np.where(quadrilateral_path.contains_points(np.column_stack((self.data.z, self.data.r))))
                        n_elem = len(idx[0])  # update n_elem
                        i += 1  # update cycle number

                elif mode == 'circular':
                    # use a circle to identifying the scattered points in the meridional plane
                    n_elem = 0  # number of elements found in the rectangle. initialization
                    i = 0  # cycle counter
                    while n_elem == 0:
                        idx = self.find_points_inside_circle(istream, ispan, i)
                        n_elem = len(idx[0])  # update n_elem
                        i += 1  # update cycle number

                else:
                    raise ValueError('Unknown type of circumferential averaging procedure!')

                # main quantities
                self.rho[istream, ispan] = self.mass_average(self.data.rho, idx)
                self.ur[istream, ispan] = self.mass_average(self.data.ur, idx)
                self.ut[istream, ispan] = self.mass_average(self.data.ut, idx)
                self.uz[istream, ispan] = self.mass_average(self.data.uz, idx)
                self.p[istream, ispan] = self.mass_average(self.data.p, idx)
                self.T[istream, ispan] = self.mass_average(self.data.T, idx)
                self.s[istream, ispan] = self.mass_average(self.data.s, idx)
                self.u_mag[istream, ispan] = self.mass_average(self.data.u_mag, idx)
                self.u_mag_rel[istream, ispan] = self.mass_average(self.data.u_mag_rel, idx)

                # gradients
                self.drho_dr[istream, ispan] = self.mass_average(self.data.drho_dr, idx)
                self.drho_dtheta[istream, ispan] = self.mass_average(self.data.drho_dtheta, idx)
                self.drho_dz[istream, ispan] = self.mass_average(self.data.drho_dz, idx)
                self.dur_dr[istream, ispan] = self.mass_average(self.data.dur_dr, idx)
                self.dur_dtheta[istream, ispan] = self.mass_average(self.data.dur_dtheta, idx)
                self.dur_dz[istream, ispan] = self.mass_average(self.data.dur_dz, idx)
                self.dut_dr[istream, ispan] = self.mass_average(self.data.dut_dr, idx)
                self.dut_dtheta[istream, ispan] = self.mass_average(self.data.dut_dtheta, idx)
                self.dut_dz[istream, ispan] = self.mass_average(self.data.dut_dz, idx)
                self.duz_dr[istream, ispan] = self.mass_average(self.data.duz_dr, idx)
                self.duz_dtheta[istream, ispan] = self.mass_average(self.data.duz_dtheta, idx)
                self.duz_dz[istream, ispan] = self.mass_average(self.data.duz_dz, idx)
                self.dp_dr[istream, ispan] = self.mass_average(self.data.dp_dr, idx)
                self.dp_dtheta[istream, ispan] = self.mass_average(self.data.dp_dtheta, idx)
                self.dp_dz[istream, ispan] = self.mass_average(self.data.dp_dz, idx)
                self.ds_dr[istream, ispan] = self.mass_average(self.data.ds_dr, idx)
                self.ds_dtheta[istream, ispan] = self.mass_average(self.data.ds_dtheta, idx)
                self.ds_dz[istream, ispan] = self.mass_average(self.data.ds_dz, idx)

                # body force model quantities
                if bfm == 'radial':
                    self.k[istream, ispan] = self.mass_average(self.data.k, idx)
                    self.F_ntheta[istream, ispan] = self.mass_average(self.data.F_ntheta, idx)
                    self.F_nr[istream, ispan] = self.mass_average(self.data.F_nr, idx)
                    self.F_nz[istream, ispan] = self.mass_average(self.data.F_nz, idx)
                    self.a1[istream, ispan] = self.mass_average(self.data.a1, idx)
                    self.a2[istream, ispan] = self.mass_average(self.data.a2, idx)
                    self.a3[istream, ispan] = self.mass_average(self.data.a3, idx)
                    self.Fn_prime_ss_00[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_00, idx)
                    self.Fn_prime_ss_01[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_01, idx)
                    self.Fn_prime_ss_02[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_02, idx)
                    self.Fn_prime_ss_10[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_10, idx)
                    self.Fn_prime_ss_11[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_11, idx)
                    self.Fn_prime_ss_12[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_12, idx)
                    self.Fn_prime_ss_20[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_20, idx)
                    self.Fn_prime_ss_21[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_21, idx)
                    self.Fn_prime_ss_22[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_22, idx)
                    self.Ft_prime_ss_00[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_00, idx)
                    self.Ft_prime_ss_01[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_01, idx)
                    self.Ft_prime_ss_02[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_02, idx)
                    self.Ft_prime_ss_10[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_10, idx)
                    self.Ft_prime_ss_11[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_11, idx)
                    self.Ft_prime_ss_12[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_12, idx)
                    self.Ft_prime_ss_20[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_20, idx)
                    self.Ft_prime_ss_21[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_21, idx)
                    self.Ft_prime_ss_22[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_22, idx)

        if fix_borders:
            self.fix_borders()

        if gauss_filter:
            self.gauss_filtering()

        self.ut_drag = self.data.omega_shaft * self.r_grid
        self.ut_rel = self.ut - self.ut_drag

        self.ut_drag = self.ut - self.ut_rel
        self.M = self.u_mag / sqrt(1.4 * self.p / self.rho)
        self.compute_stagnation_quantities()
        if bfm == 'radial':
            self.mu = self.compute_mu()
            self.F_t = self.mu * self.u_mag_rel ** 2

    def instantiate_2d_fields(self):
        """
        instantiate the 2D fields that will be averaged starting from the CFD data
        """
        self.rho = np.zeros((self.nstream, self.nspan))
        self.ur = np.zeros((self.nstream, self.nspan))
        self.ut = np.zeros((self.nstream, self.nspan))
        self.uz = np.zeros((self.nstream, self.nspan))
        self.p = np.zeros((self.nstream, self.nspan))
        self.T = np.zeros((self.nstream, self.nspan))
        self.s = np.zeros((self.nstream, self.nspan))
        self.u_mag = np.zeros((self.nstream, self.nspan))
        self.u_mag_rel = np.zeros((self.nstream, self.nspan))
        self.drho_dr = np.zeros((self.nstream, self.nspan))
        self.drho_dtheta = np.zeros((self.nstream, self.nspan))
        self.drho_dz = np.zeros((self.nstream, self.nspan))
        self.dur_dr = np.zeros((self.nstream, self.nspan))
        self.dur_dtheta = np.zeros((self.nstream, self.nspan))
        self.dur_dz = np.zeros((self.nstream, self.nspan))
        self.dut_dr = np.zeros((self.nstream, self.nspan))
        self.dut_dtheta = np.zeros((self.nstream, self.nspan))
        self.dut_dz = np.zeros((self.nstream, self.nspan))
        self.duz_dr = np.zeros((self.nstream, self.nspan))
        self.duz_dtheta = np.zeros((self.nstream, self.nspan))
        self.duz_dz = np.zeros((self.nstream, self.nspan))
        self.dp_dr = np.zeros((self.nstream, self.nspan))
        self.dp_dtheta = np.zeros((self.nstream, self.nspan))
        self.dp_dz = np.zeros((self.nstream, self.nspan))
        self.ds_dr = np.zeros((self.nstream, self.nspan))
        self.ds_dtheta = np.zeros((self.nstream, self.nspan))
        self.ds_dz = np.zeros((self.nstream, self.nspan))
        self.theta_min = np.zeros((self.nstream, self.nspan))
        self.theta_max = np.zeros((self.nstream, self.nspan))

    def instantiate_2d_bfm_fields(self):
        """
        instantiate the 2D fields necessary for the body force model, depending on the specific model used
        """
        if self.bfm == 'radial':
            self.k = np.zeros((self.nstream, self.nspan))
            self.F_ntheta = np.zeros((self.nstream, self.nspan))
            self.F_nr = np.zeros((self.nstream, self.nspan))
            self.F_nz = np.zeros((self.nstream, self.nspan))
            self.a1 = np.zeros((self.nstream, self.nspan))
            self.a2 = np.zeros((self.nstream, self.nspan))
            self.a3 = np.zeros((self.nstream, self.nspan))
            self.Fn_prime_ss_00 = np.zeros((self.nstream, self.nspan))
            self.Fn_prime_ss_01 = np.zeros((self.nstream, self.nspan))
            self.Fn_prime_ss_02 = np.zeros((self.nstream, self.nspan))
            self.Fn_prime_ss_10 = np.zeros((self.nstream, self.nspan))
            self.Fn_prime_ss_11 = np.zeros((self.nstream, self.nspan))
            self.Fn_prime_ss_12 = np.zeros((self.nstream, self.nspan))
            self.Fn_prime_ss_20 = np.zeros((self.nstream, self.nspan))
            self.Fn_prime_ss_21 = np.zeros((self.nstream, self.nspan))
            self.Fn_prime_ss_22 = np.zeros((self.nstream, self.nspan))
            self.Ft_prime_ss_00 = np.zeros((self.nstream, self.nspan))
            self.Ft_prime_ss_01 = np.zeros((self.nstream, self.nspan))
            self.Ft_prime_ss_02 = np.zeros((self.nstream, self.nspan))
            self.Ft_prime_ss_10 = np.zeros((self.nstream, self.nspan))
            self.Ft_prime_ss_11 = np.zeros((self.nstream, self.nspan))
            self.Ft_prime_ss_12 = np.zeros((self.nstream, self.nspan))
            self.Ft_prime_ss_20 = np.zeros((self.nstream, self.nspan))
            self.Ft_prime_ss_21 = np.zeros((self.nstream, self.nspan))
            self.Ft_prime_ss_22 = np.zeros((self.nstream, self.nspan))

    def find_rectangle(self, istream, ispan, A=0):
        """
        Args:
            istream: stream position
            ispan: span position
            A: number of attempts. if >1 it increases the research zone
        Returns:
            quadrilateral path
        """
        # bounding vertices, enlarged wit the number of attempts already performed
        z1 = self.z_grid_secondary[istream, ispan]  # bottom left corner
        z2 = self.z_grid_secondary[istream + 1, ispan]  # bottom right corner
        z3 = self.z_grid_secondary[istream + 1, ispan + 1]  # top right corner
        z4 = self.z_grid_secondary[istream, ispan + 1]  # top left corner
        r1 = self.r_grid_secondary[istream, ispan]
        r2 = self.r_grid_secondary[istream + 1, ispan]
        r3 = self.r_grid_secondary[istream + 1, ispan + 1]
        r4 = self.r_grid_secondary[istream, ispan + 1]

        if A == 0:
            # original research domain
            z_vertices = [z1, z2, z3, z4]
            r_vertices = [r1, r2, r3, r4]

        else:
            # enlarged research domain
            print('research domain enlarged, point (%2d,%2d), attempt %2d' % (istream, ispan, A))
            z_vertices = [z1 - A * (z2 - z1),
                          z2 + A * (z2 - z1),
                          z3 + A * (z3 - z4),
                          z4 - A * (z3 - z4)]

            r_vertices = [r1 - A * (r2 - r1),
                          r2 + A * (r2 - r1),
                          r3 + A * (r3 - r4),
                          r4 - A * (r3 - r4)]

        quadrilateral_path = mplpath.Path(np.column_stack((z_vertices, r_vertices)))
        return quadrilateral_path

    def find_points_inside_circle(self, istream, ispan, A):
        """

        Args:
            istream: index in the stremwise position on the meridional grid
            ispan: index in the spanwise position on the meridional grid
            A: number of attempt in the research

        Returns:
            idx: indices of the elements in the 3D dataset having position inside a circle around the meridional grid point
        """

        distance = sqrt((self.data.z - self.z_grid[istream, ispan]) ** 2 +
                        (self.data.r - self.r_grid[istream, ispan]) ** 2)

        # this block of if's take care of the nodes on the boundary
        if istream == self.nstream - 1:
            istream -= 1
        if ispan == self.nspan - 1:
            ispan -= 1

        l1 = sqrt((self.z_grid[istream + 1, ispan] - self.z_grid[istream, ispan]) ** 2 +
                  (self.r_grid[istream + 1, ispan] - self.r_grid[istream, ispan]) ** 2)

        l2 = sqrt((self.z_grid[istream, ispan + 1] - self.z_grid[istream, ispan]) ** 2 +
                  (self.r_grid[istream, ispan + 1] - self.r_grid[istream, ispan]) ** 2)

        r_lim = 0.25 * (l1 + l2)
        # r_lim = min(l1, l2)

        if A >= 1:
            # enlarged domain of research
            print('research domain enlarged, point (%2d,%2d), attempt %2d' % (istream, ispan, A))
            # r_lim += r_lim * A
            r_lim += r_lim * A

        idx = np.where(distance <= r_lim)
        return idx

    def mass_average(self, field, idx):
        """
        mass weighted average of a fluid dynamics field[idx]
        """
        avg = np.sum(field[idx] * self.data.rho[idx]) / np.sum(self.data.rho[idx])
        return avg

    def gauss_filtering(self):
        """
        apply the gauss filter to the field, overwriting the previous results
        """
        self.rho = self.apply_gaussian_filter(self.rho)
        self.ur = self.apply_gaussian_filter(self.ur)
        self.ut = self.apply_gaussian_filter(self.ut)
        self.uz = self.apply_gaussian_filter(self.uz)
        self.p = self.apply_gaussian_filter(self.p)
        self.s = self.apply_gaussian_filter(self.s)
        self.T = self.apply_gaussian_filter(self.T)
        self.u_mag = self.apply_gaussian_filter(self.u_mag)
        self.u_mag_rel = self.apply_gaussian_filter(self.u_mag_rel)
        self.drho_dr = self.apply_gaussian_filter(self.drho_dr)
        self.drho_dtheta = self.apply_gaussian_filter(self.drho_dtheta)
        self.drho_dz = self.apply_gaussian_filter(self.drho_dz)
        self.dur_dr = self.apply_gaussian_filter(self.dur_dr)
        self.dur_dtheta = self.apply_gaussian_filter(self.dur_dtheta)
        self.dur_dz = self.apply_gaussian_filter(self.dur_dz)
        self.dut_dr = self.apply_gaussian_filter(self.dut_dr)
        self.dut_dtheta = self.apply_gaussian_filter(self.dut_dtheta)
        self.dut_dz = self.apply_gaussian_filter(self.dut_dz)
        self.duz_dr = self.apply_gaussian_filter(self.duz_dr)
        self.duz_dtheta = self.apply_gaussian_filter(self.duz_dr)
        self.duz_dz = self.apply_gaussian_filter(self.duz_dz)
        self.dp_dr = self.apply_gaussian_filter(self.dp_dr)
        self.dp_dtheta = self.apply_gaussian_filter(self.dp_dr)
        self.dp_dz = self.apply_gaussian_filter(self.dp_dz)
        self.ds_dr = self.apply_gaussian_filter(self.ds_dr)
        self.ds_dtheta = self.apply_gaussian_filter(self.ds_dtheta)
        self.ds_dz = self.apply_gaussian_filter(self.ds_dz)
        if self.bfm == 'radial':
            self.k = self.apply_gaussian_filter(self.k)
            self.F_ntheta = self.apply_gaussian_filter(self.F_ntheta)
            self.F_nr = self.apply_gaussian_filter(self.F_nr)
            self.F_nz = self.apply_gaussian_filter(self.F_nz)
            self.a1 = self.apply_gaussian_filter(self.a1)
            self.a2 = self.apply_gaussian_filter(self.a2)
            self.a3 = self.apply_gaussian_filter(self.a3)
            self.Fn_prime_ss_00 = self.apply_gaussian_filter(self.Fn_prime_ss_00)
            self.Fn_prime_ss_01 = self.apply_gaussian_filter(self.Fn_prime_ss_01)
            self.Fn_prime_ss_02 = self.apply_gaussian_filter(self.Fn_prime_ss_02)
            self.Fn_prime_ss_10 = self.apply_gaussian_filter(self.Fn_prime_ss_10)
            self.Fn_prime_ss_11 = self.apply_gaussian_filter(self.Fn_prime_ss_11)
            self.Fn_prime_ss_12 = self.apply_gaussian_filter(self.Fn_prime_ss_12)
            self.Fn_prime_ss_20 = self.apply_gaussian_filter(self.Fn_prime_ss_20)
            self.Fn_prime_ss_21 = self.apply_gaussian_filter(self.Fn_prime_ss_21)
            self.Fn_prime_ss_22 = self.apply_gaussian_filter(self.Fn_prime_ss_22)
            self.Ft_prime_ss_00 = self.apply_gaussian_filter(self.Fn_prime_ss_00)
            self.Ft_prime_ss_01 = self.apply_gaussian_filter(self.Ft_prime_ss_01)
            self.Ft_prime_ss_02 = self.apply_gaussian_filter(self.Ft_prime_ss_02)
            self.Ft_prime_ss_10 = self.apply_gaussian_filter(self.Ft_prime_ss_10)
            self.Ft_prime_ss_11 = self.apply_gaussian_filter(self.Ft_prime_ss_11)
            self.Ft_prime_ss_12 = self.apply_gaussian_filter(self.Ft_prime_ss_12)
            self.Ft_prime_ss_20 = self.apply_gaussian_filter(self.Ft_prime_ss_20)
            self.Ft_prime_ss_21 = self.apply_gaussian_filter(self.Ft_prime_ss_21)
            self.Ft_prime_ss_22 = self.apply_gaussian_filter(self.Ft_prime_ss_22)

    @staticmethod
    def apply_gaussian_filter(field, sigma=3):
        """
        Gaussian filtering of a 2D field, with a specified deviation (sigma). 2 was a good value
        """
        smoothed_array = np.copy(field)
        smoothed_array = gaussian_filter(smoothed_array, sigma=sigma)
        return smoothed_array

    def fix_borders(self):
        """
        Gaussian filtering of a 2D field, with a specified deviation (sigma)
        """
        self.copy_borders(self.rho)
        self.copy_borders(self.ur)
        self.copy_borders(self.ut)
        self.copy_borders(self.uz)
        self.copy_borders(self.p)
        self.copy_borders(self.T)
        self.copy_borders(self.s)
        self.copy_borders(self.u_mag)
        self.copy_borders(self.u_mag_rel)
        self.copy_borders(self.drho_dr)
        self.copy_borders(self.drho_dtheta)
        self.copy_borders(self.drho_dz)
        self.copy_borders(self.dur_dr)
        self.copy_borders(self.dur_dtheta)
        self.copy_borders(self.dur_dz)
        self.copy_borders(self.dut_dr)
        self.copy_borders(self.dut_dtheta)
        self.copy_borders(self.dut_dz)
        self.copy_borders(self.duz_dr)
        self.copy_borders(self.duz_dtheta)
        self.copy_borders(self.duz_dz)
        self.copy_borders(self.dp_dr)
        self.copy_borders(self.dp_dtheta)
        self.copy_borders(self.dp_dz)
        self.copy_borders(self.ds_dr)
        self.copy_borders(self.ds_dtheta)
        self.copy_borders(self.ds_dz)
        if self.bfm == 'radial':
            self.copy_borders(self.k)
            self.copy_borders(self.F_ntheta)
            self.copy_borders(self.F_nr)
            self.copy_borders(self.F_nz)
            self.copy_borders(self.a1)
            self.copy_borders(self.a2)
            self.copy_borders(self.a3)


    def compute_rbf_fields(self):
        """
        compute the rbf interpolation of the primary fields
        """
        self.rho = self.rbf_interpolation(self.rho)
        self.ur = self.rbf_interpolation(self.ur)
        self.ut = self.rbf_interpolation(self.ut)
        self.uz = self.rbf_interpolation(self.uz)
        self.p = self.rbf_interpolation(self.p)
        self.T = self.rbf_interpolation(self.T)
        self.s = self.rbf_interpolation(self.s)


    def compute_rbf_gradients(self):
        """
        compute the gradients of the relevant fields, using RBF interpolation in 2D and then finite differences
        Returns:
        """
        self.drho_dr, self.drho_dtheta, self.drho_dz = self.rbf_finite_difference(self.rho)
        self.dur_dr, self.dur_dtheta, self.dur_dz = self.rbf_finite_difference(self.ur)
        self.dut_dr, self.dut_dtheta, self.dut_dz = self.rbf_finite_difference(self.ut)
        self.duz_dr, self.duz_dtheta, self.duz_dz = self.rbf_finite_difference(self.uz)
        self.dp_dr, self.dp_dtheta, self.dp_dz = self.rbf_finite_difference(self.p)
        self.dT_dr, self.dT_dtheta, self.dT_dz = self.rbf_finite_difference(self.T)
        self.ds_dr, self.ds_dtheta, self.ds_dz = self.rbf_finite_difference(self.s)



    def rbf_interpolation(self, field):
        """
        Args:
            field: 2D field of which we want to compute the gradients. The theta-gradient is artificially set to zero
        Returns: the three components of the field
        """

        z_points_flat = self.z_grid.flatten()
        r_points_flat = self.r_grid.flatten()
        field_flat = field.flatten()

        # Create the RBFInterpolator object with the 'multiquadric' radial basis function
        # You can also try other RBF functions like 'gaussian', 'linear', etc.
        rbf = Rbf(z_points_flat, r_points_flat, field_flat, function='multiquadric')
        field_interp = rbf(self.z_grid, self.r_grid)
        return field_interp



    def rbf_finite_difference(self, field):
        """
        Args:
            field: 2D field of which we want to compute the gradients. The theta-gradient is artificially set to zero
        Returns: the three components of the field
        """

        z_points_flat = self.z_grid.flatten()
        r_points_flat = self.r_grid.flatten()
        field_flat = field.flatten()

        # Create the RBFInterpolator object with the 'multiquadric' radial basis function
        # You can also try other RBF functions like 'gaussian', 'linear', etc.
        rbf = Rbf(z_points_flat, r_points_flat, field_flat, function='multiquadric')
        dz = ((np.max(self.z_grid) - np.min(self.z_grid)) / 50)
        dr = ((np.max(self.r_grid) - np.min(self.r_grid)) / 50)

        # Perform the RBF interpolation of the left points
        field_interp_right = rbf(self.z_grid + dz, self.r_grid)
        field_interp_left = rbf(self.z_grid - dz, self.r_grid)
        field_interp_up = rbf(self.z_grid, self.r_grid + dr)
        field_interp_down = rbf(self.z_grid, self.r_grid - dr)
        dfield_dz = ((field_interp_right - field_interp_left) / (2 * dz))
        dfield_dr = ((field_interp_up - field_interp_down) / (2 * dr))
        dfield_dtheta = np.zeros_like(dfield_dr)

        return dfield_dr, dfield_dtheta, dfield_dz

    @staticmethod
    def copy_borders(field):
        field[0, :] = field[1, :]
        field[-1, :] = field[-2, :]
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]

    def quiver_plot(self, save_filename=None, field=None):
        fig, ax = plt.subplots(figsize=self.blade.blade_picture_size)
        if field == 'p':
            cs = ax.contourf(self.z_grid, self.r_grid, self.p, N_levels, cmap=color_map)
            ax.quiver(self.z_grid, self.r_grid, self.uz, self.ur)
            ax.set_title(r'$p$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[Pa]}$')
        elif field is None:
            ax.quiver(self.z_grid, self.r_grid, self.uz, self.ur)
            ax.set_title(r'$u_z, \ u_r$')
        else:
            raise ValueError('unknown field type, please specify')
        ax.set_xlabel(r'$z \ \mathrm{[m]}$')
        ax.set_ylabel(r'$r \ \mathrm{[m]}$')

        if save_filename is not None:
            fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def plot_stream_line(self, field, n, save_filename=None):
        fig, ax = plt.subplots(figsize=fig_size)
        if field == 'rho':
            ax.plot(self.stream_line_length[:, n], self.rho[:, n], '--s')
            ax.set_ylabel(r'$\rho \ \mathrm{[kg/m^3]}$')
        elif field == 'ur':
            ax.plot(self.stream_line_length[:, n], self.ur[:, n], '--s')
            ax.set_ylabel(r'$u_r \ \mathrm{[m/s]}$')
        elif field == 'ut':
            ax.plot(self.stream_line_length[:, n], self.ut[:, n], '--s')
            ax.set_ylabel(r'$u_t \ \mathrm{[m/s]}$')
        elif field == 'uz':
            ax.plot(self.stream_line_length[:, n], self.uz[:, n], '--s')
            ax.set_ylabel(r'$u_z \ \mathrm{[m/s]}$')
        elif field == 'p':
            ax.plot(self.stream_line_length[:, n], self.p[:, n], '--s')
            ax.set_ylabel(r'$p \ \mathrm{[Pa]}$')

        ax.set_xlabel(r'$s \ \mathrm{[m]}$')
        if save_filename is not None:
            fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def compute_streamline_length(self):
        """
        compute the length along each streamline. Dimensional, same dimensions of cordinates
        """
        self.stream_line_length = np.zeros((self.nstream, self.nspan))
        for ispan in range(0, self.nspan):
            z = self.z_grid[:, ispan]
            r = self.r_grid[:, ispan]
            tmp_len = 0
            for istream in range(1, self.nstream):
                tmp_len += sqrt((z[istream] - z[istream - 1]) ** 2 + (r[istream] - r[istream - 1]) ** 2)
                self.stream_line_length[istream, ispan] = tmp_len

    def compute_spanwise_length(self):
        """
        compute the length along each span direction. Dimensional, same dimensions of cordinates
        """
        self.span_wise_length = np.zeros((self.nstream, self.nspan))
        for istream in range(0, self.nstream):
            z = self.z_grid[istream, :]
            r = self.r_grid[istream, :]
            tmp_len = 0
            for ispan in range(1, self.nspan):
                tmp_len += sqrt((z[ispan] - z[ispan - 1]) ** 2 + (r[ispan] - r[ispan - 1]) ** 2)
                self.span_wise_length[istream, ispan] = tmp_len

    def plot_spanline(self, field, n, save_filename=None):
        fig, ax = plt.subplots(figsize=fig_size)
        if field == 'rho':
            ax.plot(self.span_wise_length[n, :], self.rho[n, :], '--s')
            ax.set_ylabel(r'$\rho \ \mathrm{[kg/m^3]}$')
        elif field == 'ur':
            ax.plot(self.span_wise_length[n, :], self.ur[n, :], '--s')
            ax.set_ylabel(r'$u_r \ \mathrm{[m/s]}$')
        elif field == 'ut':
            ax.plot(self.span_wise_length[n, :], self.ut[n, :], '--s')
            ax.set_ylabel(r'$u_t \ \mathrm{[m/s]}$')
        elif field == 'uz':
            ax.plot(self.span_wise_length[n, :], self.uz[n, :], '--s')
            ax.set_ylabel(r'$u_z \ \mathrm{[m/s]}$')
        elif field == 'p':
            ax.plot(self.span_wise_length[n, :], self.p[n, :], '--s')
            ax.set_ylabel(r'$p \ \mathrm{[Pa]}$')

        ax.set_xlabel(r'$s \ \mathrm{[m]}$')
        if save_filename is not None:
            fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def contour_plot(self, field, save_filename=None, unit_factor=1):
        """
        decide if plotting dimensional or non-dimensional quantities, depending on the CFD dataset if it has been
        non-dimensionalised or not
        """

        if self.data.normalize:
            self.contour_plot_non_dimensional(field, save_filename)
        else:
            self.contour_plot_dimensional(field, save_filename, unit_factor)

    def contour_plot_dimensional(self, field, save_filename=None, unit_factor=1):
        """
        dimensional version of the contour plots
        Args:
            field: field to plot
            save_filename: name to save
            unit_factor: it could be needed in a second time to conver different unit systems
        """
        fig, ax = plt.subplots(figsize=self.blade.blade_picture_size)

        if field == 'rho':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.rho, N_levels, cmap=color_map)
            ax.set_title(r'$\rho$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[kg/m^3]}$')
        elif field == 'ur':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.ur, N_levels, cmap=color_map)
            ax.set_title(r'$u_r$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[m/s]}$')
        elif field == 'ut':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.ut, N_levels, cmap=color_map)
            ax.set_title(r'$u_{\theta}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[m/s]}$')
        elif field == 'uz':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.uz, N_levels, cmap=color_map)
            ax.set_title(r'$u_z$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[m/s]}$')
        elif field == 'p':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.p, N_levels, cmap=color_map)
            ax.set_title(r'$p$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[Pa]}$')
        elif field == 's':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.s, N_levels, cmap=color_map)
            ax.set_title(r'$s$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[J/kgK]}$')
        elif field == 'T':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.T, N_levels, cmap=color_map)
            ax.set_title(r'$T$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[K]}$')
        elif field == 'M':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.M, N_levels, cmap=color_map)
            ax.set_title(r'$M$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'drho_dr':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.drho_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \rho / \partial r$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[kg/m^4]}$')
        elif field == 'drho_dtheta':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.drho_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \rho / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[kg/m^4]}$')
        elif field == 'drho_dz':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.drho_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \rho / \partial z$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[kg/m^4]}$')
        elif field == 'dur_dr':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.dur_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial u_r / \partial r$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[1/s]}$')
        elif field == 'dur_dtheta':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.dur_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial u_r / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[1/s]}$')
        elif field == 'dur_dz':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.dur_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial u_r / \partial z$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[1/s]}$')
        elif field == 'dut_dr':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.dut_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial u_{\theta} / \partial r$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[1/s]}$')
        elif field == 'dut_dtheta':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.dut_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial u_{\theta} / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[1/s]}$')
        elif field == 'dut_dz':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.dut_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial u_{\theta} / \partial z$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[1/s]}$')
        elif field == 'duz_dr':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.duz_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial u_{z} / \partial r$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[1/s]}$')
        elif field == 'duz_dtheta':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.duz_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial u_{z} / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[1/s]}$')
        elif field == 'duz_dz':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.duz_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial u_{z} / \partial z$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[1/s]}$')
        elif field == 'dp_dr':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.dp_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial p / \partial r$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[Pa/m]}$')
        elif field == 'dp_dtheta':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.dp_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial p / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[Pa/m]}$')
        elif field == 'dp_dz':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.dp_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial p / \partial z$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[Pa/m]}$')
        elif field == 'ds_dr':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.ds_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial s / \partial r$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[J/kgKm]}$')
        elif field == 'ds_dtheta':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.ds_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial s / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[J/kgKm]}$')
        elif field == 'ds_dz':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.ds_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial s / \partial z$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[J/kgKm]}$')
        elif field == 'dT_dr':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.dT_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial T / \partial r$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[K/m]}$')
        elif field == 'dT_dtheta':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.dT_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial T / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[K/m]}$')
        elif field == 'dT_dz':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.dT_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial T / \partial z$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[K/m]}$')
        elif field == 'ut_rel':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.ut_rel, N_levels, cmap=color_map)
            ax.set_title(r'$w_{\theta}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[m/s]}$')
        elif field == 'ut_drag':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.ut_drag, N_levels, cmap=color_map)
            ax.set_title(r'$\Omega r$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[m/s]}$')
        elif field == 'k':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.k, N_levels, cmap=color_map)
                ax.set_title(r'$k$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[1/m]}$')
        elif field == 'F_ntheta':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.F_ntheta, N_levels, cmap=color_map)
                ax.set_title(r'$F_{n \theta}$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[m/s^2]}$')
        elif field == 'F_nr':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.F_nr, N_levels, cmap=color_map)
                ax.set_title(r'$F_{n r}$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[m/s^2]}$')
        elif field == 'F_nz':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.F_nz, N_levels, cmap=color_map)
                ax.set_title(r'$F_{n z}$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[m/s^2]}$')
        elif field == 'a1':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.a1, N_levels, cmap=color_map)
                ax.set_title(r'$a_1$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[1/s]}$')
        elif field == 'a2':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.a2, N_levels, cmap=color_map)
                ax.set_title(r'$a_2$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[1/s]}$')
        elif field == 'a3':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.a3, N_levels, cmap=color_map)
                ax.set_title(r'$a_3$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[1/s]}$')
        elif field == 'streamline length':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.stream_line_length,
                                 levels=N_levels, cmap=color_map)
                ax.set_title(r'streamline length')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[m]}$')
        elif field == 'mu':
            if self.bfm == 'radial':
                contour_levels = np.linspace(0, np.max(self.mu), N_levels)
                cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.mu,
                                 levels=contour_levels, cmap=color_map)
                ax.set_title(r'$mu$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[1/m]}$')
        elif field == 'F_t':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.F_t, N_levels, cmap=color_map)
                ax.set_title(r'$F_t$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[m/s^2]}$')
        elif field == 'p_tot':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.p_tot, N_levels, cmap=color_map)
            ax.set_title(r'$p_{t}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[Pa]}$')
        elif field == 'p_tot_bar':
            cs = ax.contourf(self.z_grid * unit_factor, self.r_grid * unit_factor, self.p_tot_bar, N_levels, cmap=color_map)
            ax.set_title(r'$\bar{p}_{t}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[Pa]}$')
        else:
            raise Exception('Choose a valid contour plot data!')
        # cb = fig.colorbar(cs)
        ax.set_xlabel(r'$z \ \mathrm{[mm]}$')
        ax.set_ylabel(r'$r \ \mathrm{[mm]}$')
        if save_filename is not None:
            fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def contour_plot_non_dimensional(self, field, save_filename=None):
        """
        non-dimensional version of the contour plots
        Args:
            field: field to plot
            save_filename: name to save
        """
        fig, ax = plt.subplots(figsize=self.blade.blade_picture_size)

        if field == 'rho':
            cs = ax.contourf(self.z_grid, self.r_grid, self.rho, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{\rho}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ur':
            cs = ax.contourf(self.z_grid, self.r_grid, self.ur, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{u}_r$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ut':
            cs = ax.contourf(self.z_grid, self.r_grid, self.ut, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{u}_{\theta}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'uz':
            cs = ax.contourf(self.z_grid, self.r_grid, self.uz, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{u}_z$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'p':
            cs = ax.contourf(self.z_grid, self.r_grid, self.p, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{p}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 's':
            cs = ax.contourf(self.z_grid, self.r_grid, self.s, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{s}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'T':
            cs = ax.contourf(self.z_grid, self.r_grid, self.T, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{T}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'M':
            cs = ax.contourf(self.z_grid, self.r_grid, self.M, N_levels, cmap=color_map)
            ax.set_title(r'$M$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'drho_dr':
            cs = ax.contourf(self.z_grid, self.r_grid, self.drho_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{\rho} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'drho_dtheta':
            cs = ax.contourf(self.z_grid, self.r_grid, self.drho_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{\rho} / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'drho_dz':
            cs = ax.contourf(self.z_grid, self.r_grid, self.drho_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{\rho} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[]}$')
        elif field == 'dur_dr':
            cs = ax.contourf(self.z_grid, self.r_grid, self.dur_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_r / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dur_dtheta':
            cs = ax.contourf(self.z_grid, self.r_grid, self.dur_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_r / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dur_dz':
            cs = ax.contourf(self.z_grid, self.r_grid, self.dur_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_r / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dut_dr':
            cs = ax.contourf(self.z_grid, self.r_grid, self.dut_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_{\theta} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dut_dtheta':
            cs = ax.contourf(self.z_grid, self.r_grid, self.dut_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_{\theta} / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dut_dz':
            cs = ax.contourf(self.z_grid, self.r_grid, self.dut_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_{\theta} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'duz_dr':
            cs = ax.contourf(self.z_grid, self.r_grid, self.duz_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_{z} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'duz_dtheta':
            cs = ax.contourf(self.z_grid, self.r_grid, self.duz_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_{z} / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'duz_dz':
            cs = ax.contourf(self.z_grid, self.r_grid, self.duz_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_{z} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dp_dr':
            cs = ax.contourf(self.z_grid, self.r_grid, self.dp_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{p} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dp_dtheta':
            cs = ax.contourf(self.z_grid, self.r_grid, self.dp_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{p} / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dp_dz':
            cs = ax.contourf(self.z_grid, self.r_grid, self.dp_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{p} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ds_dr':
            cs = ax.contourf(self.z_grid, self.r_grid, self.ds_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{s} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ds_dtheta':
            cs = ax.contourf(self.z_grid, self.r_grid, self.ds_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{s} / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ds_dz':
            cs = ax.contourf(self.z_grid, self.r_grid, self.ds_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{s} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dT_dr':
            cs = ax.contourf(self.z_grid, self.r_grid, self.dT_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{T} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dT_dtheta':
            cs = ax.contourf(self.z_grid, self.r_grid, self.dT_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{T} / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dT_dz':
            cs = ax.contourf(self.z_grid, self.r_grid, self.dT_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{T} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ut_rel':
            cs = ax.contourf(self.z_grid, self.r_grid, self.ut_rel, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{w}_{\theta}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ut_drag':
            cs = ax.contourf(self.z_grid, self.r_grid, self.ut_drag, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{v}_{\theta}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'k':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid, self.r_grid, self.k, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{k}$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'F_ntheta':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid, self.r_grid, self.F_ntheta, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{F}_{n \theta}$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'F_nr':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid, self.r_grid, self.F_nr, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{F}_{n r}$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'F_nz':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid, self.r_grid, self.F_nz, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{F}_{n z}$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'a1':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid, self.r_grid, self.a1, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{a}_1$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'a2':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid, self.r_grid, self.a2, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{a}_2$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'a3':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid, self.r_grid, self.a3, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{a}_3$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'streamline length':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid, self.r_grid, self.stream_line_length,
                                 levels=N_levels, cmap=color_map)
                ax.set_title(r'streamline length')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'mu':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid, self.r_grid, self.mu,
                                 levels=N_levels, cmap=color_map)
                ax.set_title(r'$\hat{\mu}$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'F_t':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid, self.r_grid, self.F_t, N_levels, cmap=color_map)
                ax.set_title(r'$\hat{F}_t$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'F_t quiver':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid, self.r_grid, self.F_t, N_levels, cmap=color_map)
                ax.quiver(self.z_grid, self.r_grid, -self.uz, -self.ur)
                ax.set_title(r'$\hat{F}_t$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'F_n':
            if self.bfm == 'radial':
                cs = ax.contourf(self.z_grid, self.r_grid, np.sqrt(self.F_nr ** 2 + self.F_ntheta ** 2 +
                                                                   self.F_nz ** 2), N_levels, cmap=color_map)
                ax.set_title(r'$\hat{F}_n$')
                cb = fig.colorbar(cs)
                cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'p_tot':
            cs = ax.contourf(self.z_grid, self.r_grid, self.p_tot, N_levels, cmap=color_map)
            ax.set_title(r'$\hat{p}_{t}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'p_tot_bar':
            cs = ax.contourf(self.z_grid, self.r_grid, self.p_tot_bar, N_levels, cmap=color_map)
            ax.set_title(r'$\bar{\hat{p}}_{t}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        else:
            raise Exception('Choose a valid contour plot data!')
        # cb = fig.colorbar(cs)
        ax.set_xlabel(r'$\hat{z} \ \mathrm{[-]}$')
        ax.set_ylabel(r'$\hat{r} \ \mathrm{[-]}$')
        if save_filename is not None:
            fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def compute_stagnation_quantities(self, gmma=1.4):
        """
        compute the 2D fields of the stagnation quantities
        """
        self.gmma = gmma  # typical value for air
        self.p_tot = self.p * (1 + (self.gmma - 1) / 2 * self.M ** 2) ** (self.gmma / (self.gmma - 1))
        self.T_tot = self.T * (1 + (self.gmma - 1) / 2 * self.M ** 2)

        # rotary total pressure
        self.p_tot_bar = self.p_tot - self.rho * self.r_grid * self.data.omega_shaft * self.ut

    def compute_mu(self):
        """
        compute the parts of the radial BFM that can be computed directly on the meridional grid rather than the 3D dataset
        """

        # compute the mu constant of the model. for each point in the 2D domain. The last streamwise position inherits the data
        # from the previous point
        self.mu = np.zeros_like(self.rho)
        for istream in range(1, self.nstream - 1):
            for ispan in range(1, self.nspan - 1):

                # find the point b, starting from point a and its velocity
                # u = np.array([self.ur[istream, ispan],
                #               self.uz[istream, ispan]])

                # direction vector along the stream direction (west to east)
                t_vec = np.array([self.r_grid[istream + 1, ispan] - self.r_grid[istream, ispan],
                                  self.z_grid[istream + 1, ispan] - self.z_grid[istream, ispan]])
                t_vec /= np.linalg.norm(t_vec)

                # direction vector along the span direction (south to north)
                n_vec = np.array([self.r_grid[istream, ispan + 1] - self.r_grid[istream, ispan],
                                  self.z_grid[istream, ispan + 1] - self.z_grid[istream, ispan]])
                n_vec /= np.linalg.norm(n_vec)

                # # scores of the 4 points surrounding the point
                # score_east = np.dot(u, t_vec)
                # score_west = np.dot(u, -t_vec)
                # score_north = np.dot(u, n_vec)
                # score_south = np.dot(u, -n_vec)
                # directions = ['east', 'west', 'north', 'south']
                # score = [score_east, score_west, score_north, score_south]
                # winner = directions[np.argmax(score)]
                # if winner == 'east':
                #     d_stream = 1
                #     d_span = 0
                # elif winner == 'west':
                #     d_stream = -1
                #     d_span = 0
                # elif winner == 'north':
                #     d_stream = 0
                #     d_span = 1
                # elif winner == 'south':
                #     d_stream = 0
                #     d_span = -1
                # else:
                #     raise ValueError('Error in selection of the streamwise point, no winner')
                #
                # ptot_bar_a = self.p_tot_bar[istream, ispan]
                # ptot_bar_b = self.p_tot_bar[istream + d_stream, ispan + d_span]

                # global fashion to avoid recirculation problems. This makes an overall average effect, using the information
                # from the beginning to the end of streamline. Details explained in Kottapalli thesis pag. 81
                ptot_bar_a = self.p_tot_bar[0, ispan]  # inlet rotary total pressure
                ptot_bar_b = self.p_tot_bar[istream, ispan]  # outlet rotary total pressure

                # z_a = self.z_grid[istream, ispan] / self.x_ref
                # r_a = self.r_grid[istream, ispan] / self.x_ref
                # z_b = self.z_grid[istream + d_stream, ispan + d_span] / self.x_ref
                # r_b = self.r_grid[istream + d_stream, ispan + d_span] / self.x_ref
                # rho_a = self.rho[istream, ispan]
                # V_a = self.u_mag_rel[istream, ispan]

                rho_avg = np.mean(self.rho[:, ispan])
                V_avg = np.mean(self.u_mag_rel[:, ispan])

                # definition of mu, from equation 16 of Sun et Al 2016
                self.mu[istream, ispan] = (ptot_bar_a - ptot_bar_b) / (
                        rho_avg * self.stream_line_length[istream, ispan] * V_avg ** 2)

                # artificially pose zero mu if the code found negative elements, which are not physical
                if self.mu[istream, ispan] < 0:
                    self.mu[istream, ispan] = 0

        self.copy_borders(self.mu)

        return self.mu


    # def build_useful_object(self):
    #     data_container = DataContainer(self.z_grid, self.r_grid, )



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


    def compute_bfm_axial(self, mode='averaged'):
        """
        compute the BFM fields, following Fang et. al. 2023
        """

        self.compute_Floss(mode=mode)

        plt.figure(figsize=self.blade.blade_picture_size)
        plt.contourf(self.z_grid, self.r_grid, self.u_meridional, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.title('u meridional')

        plt.figure(figsize=self.blade.blade_picture_size)
        plt.contourf(self.z_grid, self.r_grid, self.ds_dl, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.title('ds_dl')

        plt.figure(figsize=self.blade.blade_picture_size)
        plt.contourf(self.z_grid, self.r_grid, self.Floss, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.title('Floss')

        self.compute_Ftheta()

        plt.figure(figsize=self.blade.blade_picture_size)
        plt.contourf(self.z_grid, self.r_grid, self.drut_dl, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.title('drut_dl')

        plt.figure(figsize=self.blade.blade_picture_size)
        plt.contourf(self.z_grid, self.r_grid, self.Ftheta, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.title('Ftheta')

        self.alpha = self.Floss / self.u_mag_rel**2
        # self.beta = self.F

        plt.figure(figsize=self.blade.blade_picture_size)
        plt.contourf(self.z_grid, self.r_grid, self.alpha, cmap='jet', levels=N_levels_2)
        plt.colorbar()
        plt.title(r'$\alpha$')

    def compute_Floss(self, mode):

        # meridional flow velocity
        self.u_meridional = np.sqrt(self.ur ** 2 + self.uz ** 2)
        self.compute_ds_dl(mode=mode)
        if mode == 'global':
            self.Floss = self.T * self.u_meridional * self.ds_dl / self.u_mag_rel
        if mode == 'averaged':
            # we need to average also the other fields to make sure the global impact will be the same
            self.T_avg = np.zeros_like(self.z_grid)
            self.u_meridional_avg = np.zeros_like(self.z_grid)
            self.u_mag_rel_avg = np.zeros_like(self.z_grid)

            for ispan in range(0, self.nspan):
                self.T_avg[:, ispan] = np.ones(self.nstream)*np.mean(self.T[:, ispan])
                self.u_meridional_avg[:, ispan] = np.ones(self.nstream)*np.mean(self.u_meridional[:, ispan])
                self.u_mag_rel_avg[:, ispan] = np.ones(self.nstream)*np.mean(self.u_mag_rel[:, ispan])
            self.Floss = self.T_avg * self.u_meridional_avg * self.ds_dl / self.u_mag_rel_avg

    def compute_ds_dl(self, mode):
        """
        compute the derivative of the entropy along the meridional streamlines. In principle the increase should always be
        positive
        """
        self.ds_dl = np.zeros_like(self.s)

        if mode == 'averaged':
            for ispan in range(self.nspan):
                self.ds_dl[:, ispan] = (self.s[-1, ispan] - self.s[0, ispan]) / self.stream_line_length[-1, ispan]

        elif mode == 'global':
            for ispan in range(self.nspan):
                i = slice(1, self.nstream - 1)
                ip = slice(2, self.nstream)
                im = slice(0, self.nstream - 2)

                self.ds_dl[i, ispan] = (self.s[ip, ispan] - self.s[im, ispan]) / \
                                       (self.stream_line_length[ip, ispan] - self.stream_line_length[im, ispan])

            self.ds_dl[0, :] = (self.s[1, :] - self.s[0, :]) / (self.stream_line_length[1, :] - self.stream_line_length[0, :])
            self.ds_dl[-1, :] = (self.s[-1, :] - self.s[-2, :]) / (
                        self.stream_line_length[-1, :] - self.stream_line_length[-2, :])
            for istream in range(self.nstream):
                for ispan in range(self.nspan):
                    if self.ds_dl[istream, ispan] < 0:
                        self.ds_dl[istream, ispan] = 0



    def compute_Ftheta(self):

        self.drut_dl = np.zeros_like(self.z_grid)
        for ispan in range(self.nspan):
            i = slice(1, self.nstream - 1)
            ip = slice(2, self.nstream)
            im = slice(0, self.nstream - 2)

            self.drut_dl[i, ispan] = (self.r_grid[ip, ispan]*self.ut[ip, ispan] - self.r_grid[im, ispan]*self.ut[im, ispan]) / \
                                   (self.stream_line_length[ip, ispan] - self.stream_line_length[im, ispan])

        self.drut_dl[0, :] = (self.r_grid[1, :]*self.ut[1, :] - self.r_grid[0, :]*self.ut[0, :]) / (
                self.stream_line_length[1, :] - self.stream_line_length[0, :])

        self.drut_dl[-1, :] = (self.r_grid[-1, :] * self.ut[-1, :] - self.r_grid[-2, :] * self.ut[-2, :]) / (
                    self.stream_line_length[-1, :] - self.stream_line_length[-2, :])

        self.Ftheta = self.u_meridional / self.r_grid * self.drut_dl




# class DataContainer:
#     """
#     class that stores only the important information, necessary from the sun Model, to avoid storing
#     a lot of data for nothing
#     """
#     def __init__(self, type_name):
#         """
#         type_name: to store the type of block
#         """
#         self.type_name = type_name
#
#     def add_data(self, z_grid, r_grid, rho, ur, ut, uz, p, drho_dr, drho_dz, dur_dr, dur_dz,
#                  dut_dr, dut_dz, duz_dr, duz_dz, dp_dr, dp_dz, nstream, nspan):
#         self.z_grid = z_grid
#         self.r_grid = r_grid
#         self.rho = rho
#         self.ur = ur
#         self.ut = ut
#         self.uz = uz
#         self.p = p
#         self.drho_dr = drho_dr
#         self.drho_dz = drho_dz
#         self.dur_dr = dur_dr
#         self.dur_dz = dur_dz
#         self.dut_dr = dut_dr
#         self.dut_dz = dut_dz
#         self.duz_dr = duz_dr
#         self.duz_dz = duz_dz
#         self.dp_dr = dp_dr
#         self.dp_dz = dp_dz
#         self.nstream = nstream
#         self.nspan = nspan


