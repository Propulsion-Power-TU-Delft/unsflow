#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:07:05 2023
@author: F. Neri, TU Delft
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt
import matplotlib.path as mplpath
from scipy.ndimage import gaussian_filter
from scipy.interpolate import Rbf
import pickle
import sympy as sp
from .styles import *
from .polynomial_ls_regression import *
from .functions import compute_picture_size
from Sun.src.general_functions import print_banner_begin, print_banner_end
from Sun.src.styles import total_chars, total_chars_mid
from scipy.interpolate import griddata


class MeridionalProcess:
    """
    Class that contains a multiblock grid object, and the CFD data results. It performs the circumferential averaging.
    Important note: if the cfd data has been normalized, all the quantities are already non-dimensional.
    """

    def __init__(self, data, block=None, blade=None, verbose=False, GAMMA=1.4):
        """
        Build the MeridionalProcess Object, which contains data and methods for the CFD post-process on the meridional plane.
        :param data: CfdData object contaning the 3D CFD dataset.
        :param block: Block Object contaning the grid, needed for circumferential averaging.
        :param blade: Blade Object.
        :param verbose: to print some info.
        :param GAMMA: cp/cv ratio. It should be modified in a 2D array for non-ideal thermodynamics applications.
        """
        self.data = data
        self.block = block
        self.nstream = np.shape(block.z_grid_cg)[0]  # use the grid baricenters
        self.nspan = np.shape(block.z_grid_cg)[1]
        self.nAxialNodes = self.nstream  # how many grid element centers
        self.nRadialNodes = self.nspan
        if blade is not None:
            self.blade = blade
        self.verbose = verbose
        self.z_grid = block.z_grid_points  # primary grid points
        self.r_grid = block.r_grid_points
        self.z_cg = block.z_grid_cg  # elements centers points
        self.r_cg = block.r_grid_cg
        self.picture_size_blank, self.picture_size_contour = compute_picture_size(self.z_cg, self.r_cg)
        self.normalize = data.normalize
        self.rho_ref = data.rho_ref
        self.u_ref = data.u_ref
        self.T_ref = data.T_ref
        self.s_ref = data.s_ref
        self.x_ref = data.x_ref
        self.omega_ref = data.omega_ref
        self.t_ref = data.t_ref
        self.omega_shaft = data.omega_shaft*self.omega_ref
        self.p_ref = data.p_ref
        self.GAMMA = GAMMA

        print_banner_begin('MERIDIONAL DATA PROCESSING')
        print(f"{'Shaft Omega [rad/s]:':<{total_chars_mid}}{self.omega_shaft:>{total_chars_mid}.3f}")
        print(f"{'Reference Omega [rad/s]:':<{total_chars_mid}}{self.omega_ref:>{total_chars_mid}.3f}")
        print(f"{'Reference Density [kg/m3]:':<{total_chars_mid}}{self.rho_ref:>{total_chars_mid}.3f}")
        print(f"{'Reference Length [m]:':<{total_chars_mid}}{self.x_ref:>{total_chars_mid}.3f}")
        print(f"{'Reference Velocity [m/s]:':<{total_chars_mid}}{self.u_ref:>{total_chars_mid}.3f}")
        print(f"{'Reference Pressure [Pa]:':<{total_chars_mid}}{self.p_ref:>{total_chars_mid}.3f}")
        print(f"{'Reference Time [s]:':<{total_chars_mid}}{self.t_ref:>{total_chars_mid}.6f}")
        print(f"{'Reference Temperature [K]:':<{total_chars_mid}}{self.T_ref:>{total_chars_mid}.3f}")
        print(f"{'Reference Entropy [J/kgK]:':<{total_chars_mid}}{self.s_ref:>{total_chars_mid}.3f}")
        print(f"{'Dataset Normalized:':<{total_chars_mid}}{self.normalize:>{total_chars_mid}}")
        print_banner_end()

    def compute_camber_angles(self):
        """
        Starting from the angles in the blade object (if defined), store the normal camber angle.
        """
        self.camber_normal_r = np.zeros((self.nstream, self.nspan))
        self.camber_normal_theta = np.zeros((self.nstream, self.nspan))
        self.camber_normal_z = np.zeros((self.nstream, self.nspan))
        for istream in range(self.nstream):
            for ispan in range(self.nspan):
                self.camber_normal_r[istream, ispan] = self.blade.normal_vectors_cyl[istream, ispan][0]
                self.camber_normal_theta[istream, ispan] = self.blade.normal_vectors_cyl[istream, ispan][1]
                self.camber_normal_z[istream, ispan] = self.blade.normal_vectors_cyl[istream, ispan][2]


    def get_data_from_meridional_dataset(self):
        """
        Read 2D dataset related to the meridional post-processed results in anysis, and obtain the 2D field by regression of
        it
        """
        self.instantiate_2d_fields()
        self.W = basis_function_matrix(self.data.z, self.data.r)
        self.W_dz, self.W_dr = basis_function_matrix_derivatives(self.W, self.data.z, self.data.r)

        self.rho, self.drho_dr, self.drho_dtheta, self.drho_dz = self.polynomial_regression_solution(self.data.rho)
        self.ur, self.dur_dr, self.dur_dtheta, self.dur_dz = self.polynomial_regression_solution(self.data.ur)
        self.ut, self.dut_dr, self.dut_dtheta, self.dut_dz = self.polynomial_regression_solution(self.data.ut)
        self.uz, self.duz_dr, self.duz_dtheta, self.duz_dz = self.polynomial_regression_solution(self.data.uz)
        self.p, self.dp_dr, self.dp_dtheta, self.dp_dz = self.polynomial_regression_solution(self.data.p)
        self.T, self.dT_dr, self.dT_dtheta, self.dT_dz = self.polynomial_regression_solution(self.data.T)
        self.s, self.ds_dr, self.ds_dtheta, self.ds_dz = self.polynomial_regression_solution(self.data.s)



    def circumferential_average(self, mode, fix_borders=False, bfm=None, gauss_filter=False, threshold=50):
        """
        Perform circumferential averages of the CFD dataset on the Block grid.
        :param mode: type of algorithm selected.
                rectangular: take all the points in the rectangle identified by the secondary grid.
                circular: take all the points inside a circle
                cell centered: associate to every grid element cg, the average of what lies in its domain
        :param fix_borders: if True, the values on the borders are copied from the values of the inner nodes. Deprecated
        :param bfm: if True instantiate BFM fields. (Deprecated)
        :param gauss_filter: if True enables gauss filtering of the 2D fields, to smooth it down
        :param threshold: minimum amount of points found in one element projection, in order to accept the average
        """
        print_banner_begin("CIRCUMFERENTIAL AVG. METHOD")
        print(f"{'Averaging Method:':<{total_chars_mid}}{mode:>{total_chars_mid}}")
        print(f"{'Threshold Number Set At:':<{total_chars_mid}}{threshold:>{total_chars_mid}}")
        print(f"{'Borders Artificially Fixed:':<{total_chars_mid}}{fix_borders:>{total_chars_mid}}")
        print(f"{'Fields Artificially Filtered:':<{total_chars_mid}}{gauss_filter:>{total_chars_mid}}")
        print_banner_end()

        self.instantiate_2d_fields()
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
                    while n_elem < threshold:  # minumum amount of points to make an average
                        quadrilateral_path = self.find_rectangle(istream, ispan, i)
                        idx = np.where(quadrilateral_path.contains_points(np.column_stack((self.data.z, self.data.r))))
                        n_elem = len(idx[0])  # update n_elem
                        i += 1  # update cycle number

                elif mode == 'circular':
                    # use a circle to identifying the scattered points in the meridional plane
                    n_elem = 0  # number of elements found in the rectangle. initialization
                    i = 0  # cycle counter
                    while n_elem < threshold:
                        idx = self.find_points_inside_circle(istream, ispan, i)
                        n_elem = len(idx[0])  # update n_elem
                        i += 1  # update cycle number

                elif mode == 'cell centered':
                    n_elem = 0  # number of elements found in the rectangle. initialization
                    i = 0  # cycle counter
                    while n_elem < threshold:  # limit for accepting the average
                        quadrilateral_path = self.find_rectangle(istream, ispan, i)
                        idx = np.where(quadrilateral_path.contains_points(np.column_stack((self.data.z, self.data.r))))
                        n_elem = len(idx[0])  # update n_elem
                        i += 1  # update cycle number

                else:
                    raise ValueError('Unknown type of circumferential averaging procedure!')

                # main quantities
                self.rho[istream, ispan] = self.mass_average(self.data.rho, idx, istream, ispan)
                self.ur[istream, ispan] = self.mass_average(self.data.ur, idx, istream, ispan)
                self.ut[istream, ispan] = self.mass_average(self.data.ut, idx, istream, ispan)
                self.uz[istream, ispan] = self.mass_average(self.data.uz, idx, istream, ispan)
                self.p[istream, ispan] = self.mass_average(self.data.p, idx, istream, ispan)
                self.T[istream, ispan] = self.mass_average(self.data.T, idx, istream, ispan)
                self.s[istream, ispan] = self.mass_average(self.data.s, idx, istream, ispan)
                # self.u_mag[istream, ispan] = self.mass_average(self.data.u_mag, idx, istream, ispan)
                # self.u_mag_rel[istream, ispan] = self.mass_average(self.data.u_mag_rel, idx, istream, ispan)

                # gradients
                # self.drho_dr[istream, ispan] = self.mass_average(self.data.drho_dr, idx)
                # self.drho_dtheta[istream, ispan] = self.mass_average(self.data.drho_dtheta, idx)
                # self.drho_dz[istream, ispan] = self.mass_average(self.data.drho_dz, idx)
                # self.dur_dr[istream, ispan] = self.mass_average(self.data.dur_dr, idx)
                # self.dur_dtheta[istream, ispan] = self.mass_average(self.data.dur_dtheta, idx)
                # self.dur_dz[istream, ispan] = self.mass_average(self.data.dur_dz, idx)
                # self.dut_dr[istream, ispan] = self.mass_average(self.data.dut_dr, idx)
                # self.dut_dtheta[istream, ispan] = self.mass_average(self.data.dut_dtheta, idx)
                # self.dut_dz[istream, ispan] = self.mass_average(self.data.dut_dz, idx)
                # self.duz_dr[istream, ispan] = self.mass_average(self.data.duz_dr, idx)
                # self.duz_dtheta[istream, ispan] = self.mass_average(self.data.duz_dtheta, idx)
                # self.duz_dz[istream, ispan] = self.mass_average(self.data.duz_dz, idx)
                # self.dp_dr[istream, ispan] = self.mass_average(self.data.dp_dr, idx)
                # self.dp_dtheta[istream, ispan] = self.mass_average(self.data.dp_dtheta, idx)
                # self.dp_dz[istream, ispan] = self.mass_average(self.data.dp_dz, idx)
                # self.ds_dr[istream, ispan] = self.mass_average(self.data.ds_dr, idx)
                # self.ds_dtheta[istream, ispan] = self.mass_average(self.data.ds_dtheta, idx)
                # self.ds_dz[istream, ispan] = self.mass_average(self.data.ds_dz, idx)

                # body force model quantities
                if bfm == 'radial':
                    self.k[istream, ispan] = self.mass_average(self.data.k, idx, istream, ispan)
                    self.F_ntheta[istream, ispan] = self.mass_average(self.data.F_ntheta, idx, istream, ispan)
                    self.F_nr[istream, ispan] = self.mass_average(self.data.F_nr, idx, istream, ispan)
                    self.F_nz[istream, ispan] = self.mass_average(self.data.F_nz, idx, istream, ispan)
                    self.a1[istream, ispan] = self.mass_average(self.data.a1, idx, istream, ispan)
                    self.a2[istream, ispan] = self.mass_average(self.data.a2, idx, istream, ispan)
                    self.a3[istream, ispan] = self.mass_average(self.data.a3, idx, istream, ispan)
                    self.Fn_prime_ss_00[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_00, idx, istream, ispan)
                    self.Fn_prime_ss_01[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_01, idx, istream, ispan)
                    self.Fn_prime_ss_02[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_02, idx, istream, ispan)
                    self.Fn_prime_ss_10[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_10, idx, istream, ispan)
                    self.Fn_prime_ss_11[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_11, idx, istream, ispan)
                    self.Fn_prime_ss_12[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_12, idx, istream, ispan)
                    self.Fn_prime_ss_20[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_20, idx, istream, ispan)
                    self.Fn_prime_ss_21[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_21, idx, istream, ispan)
                    self.Fn_prime_ss_22[istream, ispan] = self.mass_average(self.data.Fn_prime_ss_22, idx, istream, ispan)
                    self.Ft_prime_ss_00[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_00, idx, istream, ispan)
                    self.Ft_prime_ss_01[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_01, idx, istream, ispan)
                    self.Ft_prime_ss_02[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_02, idx, istream, ispan)
                    self.Ft_prime_ss_10[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_10, idx, istream, ispan)
                    self.Ft_prime_ss_11[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_11, idx, istream, ispan)
                    self.Ft_prime_ss_12[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_12, idx, istream, ispan)
                    self.Ft_prime_ss_20[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_20, idx, istream, ispan)
                    self.Ft_prime_ss_21[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_21, idx, istream, ispan)
                    self.Ft_prime_ss_22[istream, ispan] = self.mass_average(self.data.Ft_prime_ss_22, idx, istream, ispan)

        if fix_borders:
            print("WARNING: borders have been artifically fixed")
            self.fix_borders()

        if gauss_filter:
            print("WARNING: the fields have been artificially smoothed")
            self.gauss_filtering()

        self.u_mag = np.sqrt(self.ur**2 + self.ut**2 + self.uz**2)
        self.ut_drag = self.data.omega_shaft * self.r_cg
        self.ut_rel = self.ut - self.ut_drag
        self.u_mag_rel = np.sqrt(self.ur ** 2 + self.ut_rel ** 2 + self.uz ** 2)
        self.u_meridional = np.sqrt(self.ur ** 2 + self.uz ** 2)
        self.M = self.u_mag / sqrt(self.GAMMA * self.p / self.rho)
        self.M_rel = self.u_mag_rel / sqrt(self.GAMMA * self.p / self.rho)
        self.compute_stagnation_quantities()
        if bfm == 'radial':
            print("WARNING: deprecated method, check the code")
            self.mu = self.compute_mu()
            self.F_t = self.mu * self.u_mag_rel ** 2

    def compute_derived_quantities(self):
        """
        From the primary averaged fields, compute derived fields.
        """
        self.ut_drag = self.data.omega_shaft * self.r_cg
        self.ut_rel = self.ut - self.ut_drag
        self.u_mag = np.sqrt(self.ur ** 2 + self.ut ** 2 + self.uz ** 2)
        self.u_mag_rel = np.sqrt(self.ur ** 2 + self.ut_rel ** 2 + self.uz ** 2)
        self.u_meridional = np.sqrt(self.ur ** 2 + self.uz ** 2)
        self.M = self.u_mag / sqrt(self.GAMMA * self.p / self.rho)
        self.M_rel = self.u_mag_rel / sqrt(self.GAMMA * self.p / self.rho)
        self.compute_stagnation_quantities()

    def instantiate_2d_fields(self):
        """
        Instantiate the 2D fields that will be averaged from the CFD data
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

    # def instantiate_2d_bfm_fields(self):
    #     """
    #     instantiate the 2D fields necessary for the body force model, depending on the specific model used
    #     """
    #     if self.bfm == 'radial':
    #         self.k = np.zeros((self.nstream, self.nspan))
    #         self.F_ntheta = np.zeros((self.nstream, self.nspan))
    #         self.F_nr = np.zeros((self.nstream, self.nspan))
    #         self.F_nz = np.zeros((self.nstream, self.nspan))
    #         self.a1 = np.zeros((self.nstream, self.nspan))
    #         self.a2 = np.zeros((self.nstream, self.nspan))
    #         self.a3 = np.zeros((self.nstream, self.nspan))
    #         self.Fn_prime_ss_00 = np.zeros((self.nstream, self.nspan))
    #         self.Fn_prime_ss_01 = np.zeros((self.nstream, self.nspan))
    #         self.Fn_prime_ss_02 = np.zeros((self.nstream, self.nspan))
    #         self.Fn_prime_ss_10 = np.zeros((self.nstream, self.nspan))
    #         self.Fn_prime_ss_11 = np.zeros((self.nstream, self.nspan))
    #         self.Fn_prime_ss_12 = np.zeros((self.nstream, self.nspan))
    #         self.Fn_prime_ss_20 = np.zeros((self.nstream, self.nspan))
    #         self.Fn_prime_ss_21 = np.zeros((self.nstream, self.nspan))
    #         self.Fn_prime_ss_22 = np.zeros((self.nstream, self.nspan))
    #         self.Ft_prime_ss_00 = np.zeros((self.nstream, self.nspan))
    #         self.Ft_prime_ss_01 = np.zeros((self.nstream, self.nspan))
    #         self.Ft_prime_ss_02 = np.zeros((self.nstream, self.nspan))
    #         self.Ft_prime_ss_10 = np.zeros((self.nstream, self.nspan))
    #         self.Ft_prime_ss_11 = np.zeros((self.nstream, self.nspan))
    #         self.Ft_prime_ss_12 = np.zeros((self.nstream, self.nspan))
    #         self.Ft_prime_ss_20 = np.zeros((self.nstream, self.nspan))
    #         self.Ft_prime_ss_21 = np.zeros((self.nstream, self.nspan))
    #         self.Ft_prime_ss_22 = np.zeros((self.nstream, self.nspan))

    def find_rectangle(self, istream, ispan, A=0):
        """
        Given the grid location, find the rectangle surrounding the element.
        :param istream: streamwise position.
        :param ispan: spanwise position.
        :param A: number of attempts. if >1 it increases the research zone.
        """
        # baricenter of the element
        z_cg = self.z_cg[istream, ispan]
        r_cg = self.r_cg[istream, ispan]

        # bounding vertices, enlarged wit the number of attempts already performed
        z1 = self.z_grid[istream, ispan]  # bottom left corner
        z2 = self.z_grid[istream + 1, ispan]  # bottom right corner
        z3 = self.z_grid[istream + 1, ispan + 1]  # top right corner
        z4 = self.z_grid[istream, ispan + 1]  # top left corner
        r1 = self.r_grid[istream, ispan]
        r2 = self.r_grid[istream + 1, ispan]
        r3 = self.r_grid[istream + 1, ispan + 1]
        r4 = self.r_grid[istream, ispan + 1]

        # vertices of the bounding box
        scaling_factor = 0.2  # factor needed to expand the original figure when no points are found
        z_vertices = [z_cg + (z1 - z_cg) * (1 + A * scaling_factor),
                      z_cg + (z2 - z_cg) * (1 + A * scaling_factor),
                      z_cg + (z3 - z_cg) * (1 + A * scaling_factor),
                      z_cg + (z4 - z_cg) * (1 + A * scaling_factor)]
        r_vertices = [r_cg + (r1 - r_cg) * (1 + A * scaling_factor),
                      r_cg + (r2 - r_cg) * (1 + A * scaling_factor),
                      r_cg + (r3 - r_cg) * (1 + A * scaling_factor),
                      r_cg + (r4 - r_cg) * (1 + A * scaling_factor)]

        if A != 0:
            # print warning regarding the enlarged research domain
            print('Research domain enlarged, point [%2d, %2d], Attempt %2d' % (istream, ispan, A))
        quadrilateral_path = mplpath.Path(np.column_stack((z_vertices, r_vertices)))
        return quadrilateral_path

    def find_points_inside_circle(self, istream, ispan, A):
        """
        Given the grid location, find the indexes of the points lying inside a circle around the element.
        :param istream: streamwise position.
        :param ispan: spanwise position.
        :param A: number of attempts. if >1 it increases the research zone.
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

    def mass_average(self, field, idx, istream, ispan, RBF=False):
        """
        Mass weighted average of a generic 2D field.
        :param field: field to be averaged
        :param idx: indexes of the elements to be considered
        :param istream: streamwise position
        :param ispan: spanwise position
        :param RBF: if True, perform radial basis function weigthed average (not validated yet).
        """
        field = field[idx]
        rho = self.data.rho[idx]
        volume = self.data.volume[idx]
        r = self.data.r[idx]  # cordinate of points to average
        z = self.data.z[idx]
        r_c = self.r_cg[istream, ispan]  # grid point (center) that will contain the averaged value
        z_c = self.z_cg[istream, ispan]
        d = np.sqrt((r-r_c)**2 + (z-z_c)**2)  # distances of points from the center
        dmax = np.max(d)  # maximum distance of all points considered
        if RBF:
            weight = (1+d/dmax)**5 * (1-d/dmax)**5  # C4 RBF by Mendez
        else:
            weight = np.zeros_like(rho)+1
        avg = np.sum(field * rho * volume * weight) / np.sum(rho * volume * weight)
        return avg

    def gauss_filtering(self):
        """
        Apply the gauss filter to the specified fields, overwriting the original quantities
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
        # if self.bfm == 'radial':
        #     self.k = self.apply_gaussian_filter(self.k)
        #     self.F_ntheta = self.apply_gaussian_filter(self.F_ntheta)
        #     self.F_nr = self.apply_gaussian_filter(self.F_nr)
        #     self.F_nz = self.apply_gaussian_filter(self.F_nz)
        #     self.a1 = self.apply_gaussian_filter(self.a1)
        #     self.a2 = self.apply_gaussian_filter(self.a2)
        #     self.a3 = self.apply_gaussian_filter(self.a3)
        #     self.Fn_prime_ss_00 = self.apply_gaussian_filter(self.Fn_prime_ss_00)
        #     self.Fn_prime_ss_01 = self.apply_gaussian_filter(self.Fn_prime_ss_01)
        #     self.Fn_prime_ss_02 = self.apply_gaussian_filter(self.Fn_prime_ss_02)
        #     self.Fn_prime_ss_10 = self.apply_gaussian_filter(self.Fn_prime_ss_10)
        #     self.Fn_prime_ss_11 = self.apply_gaussian_filter(self.Fn_prime_ss_11)
        #     self.Fn_prime_ss_12 = self.apply_gaussian_filter(self.Fn_prime_ss_12)
        #     self.Fn_prime_ss_20 = self.apply_gaussian_filter(self.Fn_prime_ss_20)
        #     self.Fn_prime_ss_21 = self.apply_gaussian_filter(self.Fn_prime_ss_21)
        #     self.Fn_prime_ss_22 = self.apply_gaussian_filter(self.Fn_prime_ss_22)
        #     self.Ft_prime_ss_00 = self.apply_gaussian_filter(self.Fn_prime_ss_00)
        #     self.Ft_prime_ss_01 = self.apply_gaussian_filter(self.Ft_prime_ss_01)
        #     self.Ft_prime_ss_02 = self.apply_gaussian_filter(self.Ft_prime_ss_02)
        #     self.Ft_prime_ss_10 = self.apply_gaussian_filter(self.Ft_prime_ss_10)
        #     self.Ft_prime_ss_11 = self.apply_gaussian_filter(self.Ft_prime_ss_11)
        #     self.Ft_prime_ss_12 = self.apply_gaussian_filter(self.Ft_prime_ss_12)
        #     self.Ft_prime_ss_20 = self.apply_gaussian_filter(self.Ft_prime_ss_20)
        #     self.Ft_prime_ss_21 = self.apply_gaussian_filter(self.Ft_prime_ss_21)
        #     self.Ft_prime_ss_22 = self.apply_gaussian_filter(self.Ft_prime_ss_22)

    @staticmethod
    def apply_gaussian_filter(field, sigma=2):
        """
        Gaussian filtering of a 2D field.
        :param field: field to average.
        :param sigma: standard deviation of the filtering. 2 looks good
        """
        smoothed_array = np.copy(field)
        smoothed_array = gaussian_filter(smoothed_array, sigma=sigma)
        return smoothed_array

    def fix_borders(self):
        """
        Copy the information from the inner points to the borders. Not useful anymore
        """
        print("WARNING: artificial fixing of the border values. Deprecated.")
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
        # if self.bfm == 'radial':
        #     self.copy_borders(self.k)
        #     self.copy_borders(self.F_ntheta)
        #     self.copy_borders(self.F_nr)
        #     self.copy_borders(self.F_nz)
        #     self.copy_borders(self.a1)
        #     self.copy_borders(self.a2)
        #     self.copy_borders(self.a3)

    def compute_rbf_fields(self):
        """
        Compute the rbf interpolation of the primary fields
        """
        self.rho = self.rbf_interpolation(self.rho)
        self.ur = self.rbf_interpolation(self.ur)
        self.ut = self.rbf_interpolation(self.ut)
        self.uz = self.rbf_interpolation(self.uz)
        self.p = self.rbf_interpolation(self.p)
        self.T = self.rbf_interpolation(self.T)
        self.s = self.rbf_interpolation(self.s)

    def compute_regressed_fields(self, order=4, data_type='flat'):
        """
        Compute the fourth order polynomial regressed fields, as described in the original papers
        :param order: order of the regression. 4 is the values used in the literature.
        """
        print("Regression of the Flow Fields, order: %i" %(order))
        if order!=4:
            raise ValueError("Choose the regression order equal to 4!")
        self.W = basis_function_matrix(self.z_cg, self.r_cg, order=order)
        self.W_dz, self.W_dr = basis_function_matrix_derivatives(self.W, self.z_cg, self.r_cg)

        self.rho, self.drho_dr, self.drho_dtheta, self.drho_dz = self.polynomial_regression_solution(self.rho)
        self.ur, self.dur_dr, self.dur_dtheta, self.dur_dz = self.polynomial_regression_solution(self.ur)
        self.ut, self.dut_dr, self.dut_dtheta, self.dut_dz = self.polynomial_regression_solution(self.ut)
        self.uz, self.duz_dr, self.duz_dtheta, self.duz_dz = self.polynomial_regression_solution(self.uz)
        self.p, self.dp_dr, self.dp_dtheta, self.dp_dz = self.polynomial_regression_solution(self.p)
        self.T, self.dT_dr, self.dT_dtheta, self.dT_dz = self.polynomial_regression_solution(self.T)
        self.s, self.ds_dr, self.ds_dtheta, self.ds_dz = self.polynomial_regression_solution(self.s)


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
        regr_field_dtheta = np.zeros_like(field)  # theta derivatives always zero
        return regr_field, regr_field_dr, regr_field_dtheta, regr_field_dz


    def compute_rbf_gradients(self):
        """
        Compute the gradients of the relevant fields, using RBF interpolation in 2D and then finite differences
        """
        print("WARNING: deprecated method.")
        self.drho_dr, self.drho_dtheta, self.drho_dz = self.rbf_finite_difference(self.rho)
        self.dur_dr, self.dur_dtheta, self.dur_dz = self.rbf_finite_difference(self.ur)
        self.dut_dr, self.dut_dtheta, self.dut_dz = self.rbf_finite_difference(self.ut)
        self.duz_dr, self.duz_dtheta, self.duz_dz = self.rbf_finite_difference(self.uz)
        self.dp_dr, self.dp_dtheta, self.dp_dz = self.rbf_finite_difference(self.p)
        self.dT_dr, self.dT_dtheta, self.dT_dz = self.rbf_finite_difference(self.T)
        self.ds_dr, self.ds_dtheta, self.ds_dz = self.rbf_finite_difference(self.s)

    def rbf_interpolation(self, field):
        """
        2D RBF based interpolation of field.
        :param field: field to interpolate.
        """
        print("WARNING: deprecated method.")
        z_points_flat = self.z_cg.flatten()
        r_points_flat = self.r_cg.flatten()
        field_flat = field.flatten()

        # Create the RBFInterpolator object with the 'multiquadric' radial basis function
        # You can also try other RBF functions like 'gaussian', 'linear', etc.
        rbf = Rbf(z_points_flat, r_points_flat, field_flat, function='gaussian')
        field_interp = rbf(self.z_cg, self.r_cg)
        return field_interp

    def rbf_finite_difference(self, field):
        """
        Computes the gradients of field that is RBF interpolated with finite differences.
        :param field: field used
        """
        z_points_flat = self.z_cg.flatten()
        r_points_flat = self.r_cg.flatten()
        field_flat = field.flatten()

        # Create the RBFInterpolator object with the 'multiquadric' radial basis function
        # You can also try other RBF functions like 'gaussian', 'linear', etc.
        rbf = Rbf(z_points_flat, r_points_flat, field_flat, function='multiquadric')
        dz = ((np.max(self.z_cg) - np.min(self.z_cg)) / self.nstream)/50
        dr = ((np.max(self.r_cg) - np.min(self.r_cg)) / self.nspan)/50

        # Perform the RBF interpolation of the left points
        field_interp_right = rbf(self.z_cg + dz, self.r_cg)
        field_interp_left = rbf(self.z_cg - dz, self.r_cg)
        field_interp_up = rbf(self.z_cg, self.r_cg + dr)
        field_interp_down = rbf(self.z_cg, self.r_cg - dr)
        dfield_dz = ((field_interp_right - field_interp_left) / (2 * dz))
        dfield_dr = ((field_interp_up - field_interp_down) / (2 * dr))

        # plt.figure()
        # plt.contourf(self.z_cg, self.r_cg, field, levels=50)
        #
        # plt.figure()
        # plt.contourf(self.z_cg, self.r_cg, dfield_dz, levels=50)
        # plt.title('d/dz')
        # plt.colorbar()
        #
        # plt.figure()
        # plt.contourf(self.z_cg, self.r_cg, dfield_dr, levels=50)
        # plt.title('d/dr')
        # plt.colorbar()

        return dfield_dr, dfield_dz


    @staticmethod
    def copy_borders(field):
        """
        Copy the inner border values to the border of a generic 2D array. Deprecated.
        :param field: field to modify.
        """
        print("WARNING: deprecated method.")
        field[0, :] = field[1, :]
        field[-1, :] = field[-2, :]
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]


    def plot_stream_line(self, field, n, save_filename=None):
        """
        Plot the quantity along a streamline.
        :param field: quantitiy to plot
        :param n: streamline to consider.
        :param save_filename: if specified, saves the figure
        """
        sl_max = self.stream_line_length[:, n].max()
        fig, ax = plt.subplots(figsize=fig_size)
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
        elif field == 'M':
            ax.plot(self.stream_line_length[:, n] / sl_max, self.M[:, n], '--s')
            ax.set_ylabel(r'$M \ \mathrm{[-]}$')
        elif field == 'M_rel':
            ax.plot(self.stream_line_length[:, n] / sl_max, self.M_rel[:, n], '--s')
            ax.set_ylabel(r'$M_{rel} \ \mathrm{[-]}$')
        else:
            raise ValueError("Field name unknown!")

        ax.grid(alpha=0.3)
        ax.set_xlabel(r'$l \ \mathrm{[-]}$')
        if save_filename is not None:
            fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')
            plt.close()

    def compute_streamline_length(self):
        """
        Compute the length along each streamline. If the data was normalized, the length is already non-dimensional.
        """
        self.stream_line_length = np.zeros((self.nstream, self.nspan))
        for ispan in range(0, self.nspan):
            z = self.z_cg[:, ispan]
            r = self.r_cg[:, ispan]
            tmp_len = 0
            for istream in range(1, self.nstream):
                tmp_len += sqrt((z[istream] - z[istream - 1]) ** 2 + (r[istream] - r[istream - 1]) ** 2)
                self.stream_line_length[istream, ispan] = tmp_len

    def compute_spanwise_length(self):
        """
        Compute the length along each span direction. If the data was normalized, the length is already non-dimensional.
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
        """
        Plot the quantity along a spanline.
        :param field: quantitiy to plot
        :param n: streamline to consider.
        :param save_filename: if specified, saves the figure
        """

        fig, ax = plt.subplots(figsize=self.picture_size_blank)
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
            plt.close()


    def contour_plot(self, field, save_filename=None, unit_factor=1, quiver=False):
        """
        Contour plot of a 2D field.
        :param field: field to plot
        :param save_filename: if specified, saves the figure
        :param unit_factor: factor to be used for dimensional plots
        :param quiver: if True, superposes the quiver plots of the meridional velocity
        """

        if self.data.normalize:
            self.contour_plot_non_dimensional(field, save_filename, quiver)
        else:
            self.contour_plot_dimensional(field, save_filename, unit_factor, quiver)

    def contour_all_plots(self):
        """
        call all the contour plots
        """
        self.contour_plot(field='rho')
        self.contour_plot(field='ur')
        self.contour_plot(field='ut')
        self.contour_plot(field='ut_rel')
        self.contour_plot(field='ut_drag')
        self.contour_plot(field='uz')
        self.contour_plot(field='p')
        self.contour_plot(field='s')
        self.contour_plot(field='T')
        self.contour_plot(field='drho_dr')
        self.contour_plot(field='drho_dz')
        self.contour_plot(field='dur_dr')
        self.contour_plot(field='dur_dz')
        self.contour_plot(field='dut_dr')
        self.contour_plot(field='dut_dz')
        self.contour_plot(field='duz_dr')
        self.contour_plot(field='duz_dz')
        self.contour_plot(field='dp_dr')
        self.contour_plot(field='dp_dz')
        self.contour_plot(field='dT_dr')
        self.contour_plot(field='dT_dz')
        self.contour_plot(field='ds_dr')
        self.contour_plot(field='ds_dz')
        self.contour_plot(field='M')
        self.contour_plot(field='p_tot')
        self.contour_plot(field='p_tot_bar')
        self.contour_plot(field='T_tot')

    def contour_plot_dimensional(self, field, save_filename=None, unit_factor=1, quiver=False):
        """
        Contour plot of a 2D field.
        :param field: field to plot
        :param save_filename: if specified, saves the figure
        :param unit_factor: factor to be used for dimensional plots
        :param quiver: if True, superposes the quiver plots of the meridional velocity
        """
        fig, ax = plt.subplots(figsize=self.picture_size_contour)

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
        if quiver:
            ax.quiver(self.z_cg, self.r_cg, self.uz, self.ur)
        ax.set_xlabel(r'$z \ \mathrm{[mm]}$')
        ax.set_ylabel(r'$r \ \mathrm{[mm]}$')
        if save_filename is not None:
            fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')
            plt.close()

    def contour_plot_non_dimensional(self, field, save_filename=None, quiver=False):
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
        elif field == 'drho_dr':
            cs = ax.contourf(self.z_cg, self.r_cg, self.drho_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{\rho} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'drho_dtheta':
            cs = ax.contourf(self.z_cg, self.r_cg, self.drho_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{\rho} / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'drho_dz':
            cs = ax.contourf(self.z_cg, self.r_cg, self.drho_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{\rho} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dur_dr':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dur_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_r / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dur_dtheta':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dur_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_r / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dur_dz':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dur_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_r / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dut_dr':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dut_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_{\theta} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dut_dtheta':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dut_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_{\theta} / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dut_dz':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dut_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_{\theta} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'duz_dr':
            cs = ax.contourf(self.z_cg, self.r_cg, self.duz_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_{z} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'duz_dtheta':
            cs = ax.contourf(self.z_cg, self.r_cg, self.duz_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_{z} / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'duz_dz':
            cs = ax.contourf(self.z_cg, self.r_cg, self.duz_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{u}_{z} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dp_dr':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dp_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{p} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dp_dtheta':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dp_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{p} / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dp_dz':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dp_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{p} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ds_dr':
            cs = ax.contourf(self.z_cg, self.r_cg, self.ds_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{s} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ds_dtheta':
            cs = ax.contourf(self.z_cg, self.r_cg, self.ds_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{s} / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'ds_dz':
            cs = ax.contourf(self.z_cg, self.r_cg, self.ds_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{s} / \partial \hat{z}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dT_dr':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dT_dr, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{T} / \partial \hat{r}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dT_dtheta':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dT_dtheta, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{T} / \partial \theta$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'dT_dz':
            cs = ax.contourf(self.z_cg, self.r_cg, self.dT_dz, N_levels, cmap=color_map)
            ax.set_title(r'$\partial \hat{T} / \partial \hat{z}$')
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
        if quiver:
            ax.quiver(self.z_cg, self.r_cg, self.uz, self.ur)
        if save_filename is not None:
            fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')
            plt.close()

    def compute_stagnation_quantities(self):
        """
        Compute the 2D fields of the stagnation quantities
        """
        self.p_tot = self.p * (1 + (self.GAMMA - 1) / 2 * self.M ** 2) ** (self.GAMMA / (self.GAMMA - 1))
        self.T_tot = self.T * (1 + (self.GAMMA - 1) / 2 * self.M ** 2)

        # rotary total pressure, as defined by Sun et al. (centrifugal compressor analysis 2016)
        self.p_tot_bar = self.p_tot - self.rho * self.r_cg * self.data.omega_shaft * self.ut

    def compute_mu(self):
        """
        Compute the parts of the radial BFM that can be computed directly on the meridional grid rather than the 3D dataset
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
        Store the object content in a pickle.
        :param file_name: name to store. if None, default one is selected
        :param folder: location to store. if None, default one is selected
        """
        if folder is None:
            folder = folder_meta_data_default
        if file_name is None:
            file_name = 'meridional_process_%d_%d.pickle' % (self.nstream, self.nspan)

        with open(folder + file_name + '.pickle', "wb") as file:
            pickle.dump(self, file)

    def compute_bfm_axial(self, mode='global', save_fig=False):
        """
        Compute the BFM fields, following the description in Fang et al. 2023.
        :param mode: if global is the default one, without any artifical fixing.
        :param save_fig: if specified, saves the figure
        """
        self.compute_Floss(mode=mode)
        self.compute_Ftheta()
        self.compute_Fturn()
        self.alpha = self.Floss / (self.u_mag_rel ** 2)
        self.beta = self.Fturn / (self.u_meridional * self.ut_rel)

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.u_meridional, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$\hat{u}_{m}$')
        if save_fig:
            plt.savefig('pictures/u_meridional_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.ds_dl, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$\frac{\partial s}{\partial m}$')
        if save_fig:
            plt.savefig('pictures/ds_dl_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.Floss, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$F_{l}$')
        if save_fig:
            plt.savefig('pictures/F_loss_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.Floss_r, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$F_{l,r}$')
        if save_fig:
            plt.savefig('pictures/Fl_r_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.Floss_t, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$F_{l,\theta}$')
        if save_fig:
            plt.savefig('pictures/Fl_t_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.Floss_z, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$F_{l,z}$')
        if save_fig:
            plt.savefig('pictures/Fl_z_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.drut_dl, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$\frac{\partial (r u_{\theta})}{\partial m}$')
        if save_fig:
            plt.savefig('pictures/drut_dl_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.Ftheta, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$F_{\theta}$')
        if save_fig:
            plt.savefig('pictures/F_theta_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.Fturn_r, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$F_{t, r}$')
        if save_fig:
            plt.savefig('pictures/Fturn_r_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.Fturn_t, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$F_{t, \theta}$')
        if save_fig:
            plt.savefig('pictures/Fturn_t_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.Fturn_z, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$F_{t, z}$')
        if save_fig:
            plt.savefig('pictures/Fturn_z_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.Fturn, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$F_{t}$')
        if save_fig:
            plt.savefig('pictures/F_turn_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.alpha, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$\alpha$')
        if save_fig:
            plt.savefig('pictures/alpha_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.beta, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$\beta$')
        if save_fig:
            plt.savefig('pictures/beta_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

    def contour_entropy_generation(self, save_fig=None):
        """
        Show the contour of the entropy generation, defined as the difference between the local entropy and the
        one that was at leading edge
        """
        z = self.z_cg
        r = self.r_cg
        s = self.s.copy()
        for istream in range(self.nstream):
            s[istream, :] = s[istream, :] - s[0, :]

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(z, r, s, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$s_{GEN}$')
        if save_fig:
            plt.savefig('pictures/entropy_generation_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()

    def contour_local_entropy_generation(self, save_fig=None):
        """
        Show the contour of the entropy generation, defined as the difference between the local entropy and the
        one that was at leading edge
        """
        z = self.z_cg
        r = self.r_cg
        s = self.s.copy()
        for istream in range(1, self.nstream):
            s[istream, :] = s[istream, :] - s[istream-1, :]
        s[0, :] = s[1, :]

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(z, r, s, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$s_{GEN,L}$')
        if save_fig:
            plt.savefig('pictures/local_entropy_generation_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            plt.close()


    def compute_body_fource_S(self, domain):
        """
        if the domain is bladed compute the body force steady state matrices. Otherwise, instantiate zeros
        for the related terms.
        :param domain: type of domain (stator, rotor, unbladed).
        """
        self.domain = domain
        if domain == 'rotor' or domain == 'stator':
            print("%s Domain" %(domain))
            tr = self.Floss_r / self.Floss
            ttheta = self.Floss_t / self.Floss
            tz = self.Floss_z / self.Floss

            nr = self.Fturn_r/self.Fturn
            ntheta = self.Fturn_t / self.Fturn
            nz = self.Fturn_z / self.Fturn

            self.S00 = np.zeros_like(self.ur)
            self.S01 = np.zeros_like(self.ur)
            self.S02 = np.zeros_like(self.ur)
            self.S03 = np.zeros_like(self.ur)
            self.S04 = np.zeros_like(self.ur)

            self.S10 = np.zeros_like(self.ur)
            self.S11 = tr * 2 * self.alpha * self.ur + nr * self.ur * self.beta * self.ut_rel / self.u_meridional
            self.S12 = tr * 2 * self.alpha * self.ut_rel + nr * self.beta * self.u_meridional
            self.S13 = tr * 2 * self.alpha * self.uz + nr * self.uz * self.beta * self.ut_rel / self.u_meridional
            self.S14 = np.zeros_like(self.ur)

            self.S20 = np.zeros_like(self.ur)
            self.S21 = ttheta * 2 * self.alpha * self.ur + ntheta * self.ur * self.beta * self.ut_rel / self.u_meridional
            self.S22 = ttheta * 2 * self.alpha * self.ut_rel + ntheta * self.beta * self.u_meridional
            self.S23 = ttheta * 2 * self.alpha * self.uz + ntheta * self.uz * self.beta * self.ut_rel / self.u_meridional
            self.S24 = np.zeros_like(self.ur)

            self.S30 = np.zeros_like(self.ur)
            self.S31 = tz * 2 * self.alpha * self.ur + nz * self.ur * self.beta * self.ut_rel / self.u_meridional
            self.S32 = tz * 2 * self.alpha * self.ut_rel + nz * self.beta * self.u_meridional
            self.S33 = tz * 2 * self.alpha * self.uz + nz * self.uz * self.beta * self.ut_rel / self.u_meridional
            self.S34 = np.zeros_like(self.ur)

            self.S40 = np.zeros_like(self.ur)
            self.S41 = np.zeros_like(self.ur)
            self.S42 = np.zeros_like(self.ur)
            self.S43 = np.zeros_like(self.ur)
            self.S44 = np.zeros_like(self.ur)

            # compute quantities needed for the Sun Model Algorithm Variant
            if domain == 'rotor':
                self.Omega = np.zeros_like(self.z_cg)+self.omega_shaft
            elif domain == 'stator':
                self.Omega = np.zeros_like(self.z_cg)
            else:
                raise ValueError("Unknown domain type")
            tau_throughflow = (np.max(self.stream_line_length) - np.min(self.stream_line_length))*self.x_ref / \
                              (np.max(self.u_meridional)*self.u_ref)
            self.tau = np.zeros_like(self.z_cg) + tau_throughflow

        elif domain=='unbladed':
            print("Unbladed Domain...")
            self.S00 = np.zeros_like(self.ur)
            self.S01 = np.zeros_like(self.ur)
            self.S02 = np.zeros_like(self.ur)
            self.S03 = np.zeros_like(self.ur)
            self.S04 = np.zeros_like(self.ur)
            self.S10 = np.zeros_like(self.ur)
            self.S11 = np.zeros_like(self.ur)
            self.S12 = np.zeros_like(self.ur)
            self.S13 = np.zeros_like(self.ur)
            self.S14 = np.zeros_like(self.ur)
            self.S20 = np.zeros_like(self.ur)
            self.S21 = np.zeros_like(self.ur)
            self.S22 = np.zeros_like(self.ur)
            self.S23 = np.zeros_like(self.ur)
            self.S24 = np.zeros_like(self.ur)
            self.S30 = np.zeros_like(self.ur)
            self.S31 = np.zeros_like(self.ur)
            self.S32 = np.zeros_like(self.ur)
            self.S33 = np.zeros_like(self.ur)
            self.S34 = np.zeros_like(self.ur)
            self.S40 = np.zeros_like(self.ur)
            self.S41 = np.zeros_like(self.ur)
            self.S42 = np.zeros_like(self.ur)
            self.S43 = np.zeros_like(self.ur)
            self.S44 = np.zeros_like(self.ur)

            self.Omega = np.zeros_like(self.z_cg)
            self.tau = np.zeros_like(self.z_cg)
        else:
            raise ValueError("Unknown domain type for body force calculation. Available choices: rotor, stator, unbladed!")




    def compute_Floss(self, mode):
        """
        Compute the Loss component of the body force
        """

        # meridional flow velocity
        self.u_meridional = np.sqrt(self.ur ** 2 + self.uz ** 2)
        self.compute_ds_dl(mode=mode)
        if mode == 'global':
            # compute the modulus, and then the components, which are opposed to the relative flow velocity
            self.Floss = self.T * self.u_meridional * self.ds_dl / self.u_mag_rel

            # compute the components, which are opposite to the relative velocity
            self.Floss_r = -self.Floss * self.ur / self.u_mag_rel
            self.Floss_t = -self.Floss * self.ut_rel / self.u_mag_rel
            self.Floss_z = -self.Floss * self.uz / self.u_mag_rel

            self.Floss_check = self.Floss_r**2 + self.Floss_t**2 + self.Floss_z**2 - self.Floss**2


    def compute_ds_dl(self, mode):
        """
        Compute the derivative of the entropy along the meridional direction. In principle the increase should always be
        positive, but in reality it can also decrease (as explained by Kottapalli, due to the meridional projection).
        :param mode: if global performs the physical calculation. Averaged artificially fixes some problems.
        """
        self.ds_dl = np.zeros_like(self.s)

        if mode == 'averaged':
            for ispan in range(self.nspan):
                self.ds_dl[:, ispan] = (self.s[-1, ispan] - self.s[0, ispan]) / self.stream_line_length[-1, ispan]

        elif mode == 'global':
            # calculate ds_dl projecting the gradient of s over the direction in which the particle is going
            for istream in range(self.nAxialNodes):
                for ispan in range(self.nRadialNodes):
                    dir_vector = np.array((self.uz[istream, ispan],
                                           self.ur[istream, ispan]))
                    dir_vector /= np.linalg.norm(dir_vector)
                    self.ds_dl[istream, ispan] = self.ds_dz[istream, ispan] * dir_vector[0] + \
                                                 self.ds_dr[istream, ispan] * dir_vector[1]

    def compute_Ftheta(self):
        """
        Compute the modulus of the global theta component of the body force
        """
        dr_dl = self.ur / self.u_meridional
        dut_dl = np.zeros_like(dr_dl)

        # find the derivative projecting the gradients along the meridional velocity direction
        for istream in range(self.nAxialNodes):
            for ispan in range(self.nRadialNodes):
                dir_vector = np.array((self.uz[istream, ispan],
                                       self.ur[istream, ispan]))
                dir_vector /= np.linalg.norm(dir_vector)
                dut_dl[istream, ispan] = self.dut_dz[istream, ispan] * dir_vector[0] + \
                                         self.dut_dr[istream, ispan] * dir_vector[1]
        self.drut_dl = dr_dl * self.ut + self.r_cg * dut_dl
        self.Ftheta = self.u_meridional / self.r_cg * self.drut_dl

    def compute_Fturn(self):
        """
        Starting from the Ftheta and camber normal vectors, compute the magnitude of the turning force
        """
        self.Fturn_t = self.Ftheta - self.Floss_t
        self.Fturn = self.Fturn_t / self.camber_normal_theta
        self.Fturn_r = self.Fturn * self.camber_normal_r
        self.Fturn_z = self.Fturn * self.camber_normal_z
        self.Fturn_check = self.Fturn_r**2 + self.Fturn_t**2 + self.Fturn_z**2 - self.Fturn**2

    def compute_averaged_fluxes(self):
        """
        On the meridional plane, compute the averaged fluxed for each streamwise position.
        """
        self.dA = np.zeros_like(self.z_cg)
        self.dA_nz = np.zeros_like(self.z_cg)
        self.dA_nr = np.zeros_like(self.z_cg)
        for istream in range(self.nstream):
            for ispan in range(self.nspan):
                dz = self.z_grid[istream, ispan + 1] - self.z_grid[istream, ispan]
                dr = self.r_grid[istream, ispan + 1] - self.r_grid[istream, ispan]

                # Area of the flux per unit length in circumferential direction
                self.dA[istream, ispan] = np.sqrt(dz ** 2 + dr ** 2)

                # normal of the flux area (-90 deg rotation of the edge, normalized)
                self.dA_nz[istream, ispan] = dr/self.dA[istream, ispan]
                self.dA_nr[istream, ispan] = -dz/self.dA[istream, ispan]

        self.rho_flux = self.compute_flux(self.rho)
        self.ur_flux = self.compute_flux(self.ur)
        self.ut_flux = self.compute_flux(self.ut)
        self.uz_flux = self.compute_flux(self.uz)
        self.p_flux = self.compute_flux(self.p)
        self.M_flux = self.compute_flux(self.M)
        self.M_rel_flux = self.compute_flux(self.M_rel)
        self.s_flux = self.compute_flux(self.s)
        self.T_flux = self.compute_flux(self.T)
        self.p_tot_flux = self.compute_flux(self.p_tot)
        self.T_tot_flux = self.compute_flux(self.T_tot)

    def compute_flux(self, field):
        """
        Compute the averaged transported quantity along the span lines, and store it in the stream index.
        :param field: transport variable.
        """
        fluxes = np.zeros(self.nstream)
        for istream in range(self.nstream):
            # projection of meridional velocity on the normal boundary of each cell
            normal_velocity = self.uz[istream, :] * self.dA_nz[istream, :] + self.ur[istream, :] * self.dA_nr[istream, :]

            # flux calculation as sum{rho*u*A*field}/sum{rho*u*A}
            fluxes[istream] = np.sum(self.rho[istream, :] * self.dA[istream, :] * field[istream, :] * normal_velocity) / \
                              np.sum(self.rho[istream, :] * self.dA[istream, :] * normal_velocity)
        return fluxes


    def plot_averaged_fluxes(self, field, save_filename=None):
        """
        Plot the averaged fluxes.
        Contour plot of a 2D field.
        :param field: field to plot
        :param save_filename: if specified, saves the figure
        """
        sl_max = self.stream_line_length[:, 0].max()
        fig, ax = plt.subplots(figsize=fig_size)
        if field == 'rho':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.rho_flux, '--s')
            ax.set_ylabel(r'$\rho \ \mathrm{[-]}$')
        elif field == 'ur':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.ur_flux*self.u_ref, '--s')
            ax.set_ylabel(r'$u_r \ \mathrm{[m/s]}$')
        elif field == 'ut':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.ut_flux, '--s')
            ax.set_ylabel(r'$u_t \ \mathrm{[-]}$')
        elif field == 'uz':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.uz_flux, '--s')
            ax.set_ylabel(r'$u_z \ \mathrm{[-]}$')
        elif field == 'M':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.M_flux, '--s')
            ax.set_ylabel(r'$M \ \mathrm{[-]}$')
        elif field == 'M_rel':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.M_rel_flux, '--s')
            ax.set_ylabel(r'$M_{rel} \ \mathrm{[-]}$')
        elif field == 'p':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.p_flux, '--s')
            ax.set_ylabel(r'$p \ \mathrm{[-]}$')
        elif field == 'T':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.T_flux, '--s')
            ax.set_ylabel(r'$T \ \mathrm{[-]}$')
        elif field == 's':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.s_flux, '--s')
            ax.set_ylabel(r'$s \ \mathrm{[-]}$')
        elif field == 'p_tot':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.p_tot_flux, '--s')
            ax.set_ylabel(r'$p_{t} \ \mathrm{[-]}$')
        elif field == 'T_tot':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.T_tot_flux, '--s')
            ax.set_ylabel(r'$T_{t} \ \mathrm{[-]}$')
        else:
            raise ValueError("Field name unknown!")

        ax.grid(alpha=0.3)
        ax.set_xlabel(r'$l \ \mathrm{[-]}$')
        if save_filename is not None:
            fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')
            plt.close()

    def interpolate_on_working_grid(self, method):
        self.instantiate_2d_fields()
        self.rho = self.interpolate_function(self.data.rho, self.data.z, self.data.r, method=method)
        self.ur = self.interpolate_function(self.data.ur, self.data.z, self.data.r, method=method)
        self.ut = self.interpolate_function(self.data.ut, self.data.z, self.data.r, method=method)
        self.uz = self.interpolate_function(self.data.uz, self.data.z, self.data.r, method=method)
        self.p = self.interpolate_function(self.data.p, self.data.z, self.data.r, method=method)
        self.T = self.interpolate_function(self.data.T, self.data.z, self.data.r, method=method)
        self.s = self.interpolate_function(self.data.s, self.data.z, self.data.r, method=method)

        try:
            self.drho_dr = self.interpolate_function(self.data.drho_dr, self.data.z, self.data.r, method=method)
            self.drho_dz = self.interpolate_function(self.data.drho_dz, self.data.z, self.data.r, method=method)
            self.dur_dr = self.interpolate_function(self.data.dur_dr, self.data.z, self.data.r, method=method)
            self.dur_dz = self.interpolate_function(self.data.dur_dz, self.data.z, self.data.r, method=method)
            self.dut_dr = self.interpolate_function(self.data.dut_dr, self.data.z, self.data.r, method=method)
            self.dut_dz = self.interpolate_function(self.data.dut_dz, self.data.z, self.data.r, method=method)
            self.duz_dr = self.interpolate_function(self.data.duz_dr, self.data.z, self.data.r, method=method)
            self.duz_dz = self.interpolate_function(self.data.duz_dz, self.data.z, self.data.r, method=method)
            self.dp_dr = self.interpolate_function(self.data.dp_dr, self.data.z, self.data.r, method=method)
            self.dp_dz = self.interpolate_function(self.data.dp_dz, self.data.z, self.data.r, method=method)
            self.ds_dr = self.interpolate_function(self.data.ds_dr, self.data.z, self.data.r, method=method)
            self.ds_dz = self.interpolate_function(self.data.ds_dz, self.data.z, self.data.r, method=method)
        except:
            pass

    def compute_field_gradients(self, method='rbf'):
        """
        compute the gradient of the flow field using a certain interpolation method, in order to evaluate functions at
        z+deltaz and r+deltar.
        """
        if method == 'rbf':
            # self.drho_dr, self.drho_dz = self.rbf_finite_difference(self.rho)
            # self.dur_dr, self.dur_dz = self.rbf_finite_difference(self.ur)
            # self.dut_dr, self.dut_dz = self.rbf_finite_difference(self.ut)
            # self.duz_dr, self.duz_dz = self.rbf_finite_difference(self.uz)
            # self.dp_dr, self.dp_dz = self.rbf_finite_difference(self.p)
            self.dT_dr, self.dT_dz = self.rbf_finite_difference(self.T)
            # self.ds_dr, self.ds_dz = self.rbf_finite_difference(self.s)
        elif method == 'linear':
            # self.drho_dr, self.drho_dz = self.linear_interpolation_gradient(self.rho)
            # self.dur_dr, self.dur_dz = self.linear_interpolation_gradient(self.ur)
            # self.dut_dr, self.dut_dz = self.linear_interpolation_gradient(self.ut)
            # self.duz_dr, self.duz_dz = self.linear_interpolation_gradient(self.uz)
            # self.dp_dr, self.dp_dz = self.linear_interpolation_gradient(self.p)
            self.dT_dr, self.dT_dz = self.linear_interpolation_gradient(self.T)
            # self.ds_dr, self.ds_dz = self.linear_interpolation_gradient(self.s)
        else:
            raise ValueError("Method not recognized.")

    def linear_interpolation_gradient(self, f):
        """
        Linear interpolation method in order to compute the gradient of f
        """
        Z = self.z_cg
        R = self.r_cg
        Zplus = np.zeros_like(self.z_cg)
        Rplus = np.zeros_like(self.r_cg)
        Zminus = np.zeros_like(self.z_cg)
        Rminus = np.zeros_like(self.r_cg)
        for ii in range(0, self.nstream):
            for jj in range(0, self.nspan):
                if ii==self.nstream-1 or jj==self.nspan-1:
                    Zplus[ii, jj] = Z[ii, jj] + np.abs((Z[ii, jj] - Z[ii-1, jj]) / 2)
                    Zminus[ii, jj] = Z[ii, jj] - np.abs((Z[ii, jj] - Z[ii-1, jj]) / 2)
                    Rplus[ii, jj] = R[ii, jj] + np.abs((R[ii, jj] - R[ii, jj-1]) / 2)
                    Rminus[ii, jj] = R[ii, jj] - np.abs((R[ii, jj] - R[ii, jj-1]) / 2)
                else:
                    Zplus[ii, jj] = Z[ii, jj]+ np.abs((Z[ii+1, jj] - Z[ii, jj])/2)
                    Rplus[ii, jj] = R[ii, jj] + np.abs((R[ii, jj+1] - R[ii, jj]) / 2)
                    Zminus[ii, jj] = Z[ii, jj] - np.abs((Z[ii+1, jj] - Z[ii, jj]) / 2)
                    Rminus[ii, jj] = R[ii, jj] - np.abs((R[ii, jj+1] - R[ii, jj]) / 2)

        values = f.flatten()

        points = np.column_stack((Z.flatten(), R.flatten()))
        f_zplus = griddata(points, values, (Zplus, R), method='cubic')
        f_zminus = griddata(points, values, (Zminus, R), method='cubic')
        f_rplus = griddata(points, values, (Z, Rplus), method='cubic')
        f_rminus = griddata(points, values, (Z, Rminus), method='cubic')
        df_dz = (f_zplus-f_zminus)/(Zplus-Zminus)
        df_dz = np.reshape(df_dz, self.z_cg.shape)
        df_dr = (f_rplus - f_rminus) / (Rplus - Rminus)
        df_dr = np.reshape(df_dr, self.z_cg.shape)

        return f_zplus, f_zminus


    def interpolate_function(self, f, z, r, method):
        """

        """
        Xnew = self.z_cg  # original grid
        Ynew = self.r_cg  # original grid
        if method=='linear':
            f_new = griddata((z, r), f, (Xnew, Ynew), method='linear')
        elif method=='cubic':
            f_new = griddata((z, r), f, (Xnew, Ynew), method='cubic')
        elif method=='nearest':
            f_new = griddata((z, r), f, (Xnew, Ynew), method='nearest')
        elif method=='rbf':
            rbf_interpolator = Rbf(z, r, f, function='multiquadric')
            f_new = rbf_interpolator(Xnew, Ynew)
        else:
            raise ValueError("Method unknown")
        return f_new
