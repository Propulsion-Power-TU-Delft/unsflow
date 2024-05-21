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
from Utils.styles import *
from .polynomial_ls_regression import *
from .functions import compute_picture_size
from Sun.src.general_functions import print_banner_begin, print_banner_end
from scipy.interpolate import griddata
from Grid.src.weighted_least_squares import *
import matplotlib.lines as mlines
from scipy.interpolate import LinearNDInterpolator
from scipy import integrate
from numpy.polynomial.chebyshev import chebvander2d
import warnings


class MeridionalProcess:
    """
    Class that contains a multiblock grid object, and the CFD data results. It performs the circumferential averaging.
    Important note: if the cfd data has been normalized, all the quantities are already non-dimensional.
    """

    def __init__(self, config, data, block, blade=None):
        """
        Build the MeridionalProcess Object, which contains data and methods for the CFD post-process on the meridional plane.
        :param data: CfdData object contaning the 3D CFD dataset.
        :param block: Block Object contaning the grid, needed for circumferential averaging.
        :param blade: Blade Object.
        """
        self.config = config
        self.data = data
        self.block = block
        self.nstream = np.shape(block.z_grid_cg)[0]  # use the grid baricenters
        self.nspan = np.shape(block.z_grid_cg)[1]
        self.nAxialNodes = self.nstream  # how many grid element centers
        self.nRadialNodes = self.nspan
        if blade is not None:
            self.blade = blade
        self.z_grid = block.z_grid_points  # primary grid points
        self.r_grid = block.r_grid_points
        self.z_cg = block.z_grid_cg  # elements centers points
        self.r_cg = block.r_grid_cg
        self.picture_size_blank, self.picture_size_contour = compute_picture_size(self.z_cg, self.r_cg)

        print_banner_begin('MERIDIONAL DATA PROCESSING')
        print(f"{'Shaft Omega [rad/s]:':<{total_chars_mid}}{self.config.get_omega_shaft():>{total_chars_mid}.3f}")
        print(f"{'Reference Omega [rad/s]:':<{total_chars_mid}}{self.config.get_reference_omega():>{total_chars_mid}.3f}")
        print(f"{'Reference Density [kg/m3]:':<{total_chars_mid}}{self.config.get_reference_density():>{total_chars_mid}.3f}")
        print(f"{'Reference Length [m]:':<{total_chars_mid}}{self.config.get_reference_length():>{total_chars_mid}.3f}")
        print(f"{'Reference Velocity [m/s]:':<{total_chars_mid}}{self.config.get_reference_velocity():>{total_chars_mid}.3f}")
        print(f"{'Reference Pressure [Pa]:':<{total_chars_mid}}{self.config.get_reference_pressure():>{total_chars_mid}.3f}")
        print(f"{'Reference Time [s]:':<{total_chars_mid}}{self.config.get_reference_time():>{total_chars_mid}.6f}")
        print(f"{'Reference Temperature [K]:':<{total_chars_mid}}{self.config.get_reference_temperature():>{total_chars_mid}.3f}")
        print(f"{'Reference Entropy [J/kgK]:':<{total_chars_mid}}{self.config.get_reference_entropy():>{total_chars_mid}.3f}")
        print(f"{'Dataset Normalized:':<{total_chars_mid}}{self.config.get_normalize_data():>{total_chars_mid}}")
        print_banner_end()

    def compute_camber_angles(self):
        """
        Starting from the angles in the blade object (if defined), store the normal camber vector, and modify if something
        is wrong about it.
        """
        self.camber_normal_r = np.zeros((self.nstream, self.nspan))
        self.camber_normal_theta = np.zeros((self.nstream, self.nspan))
        self.camber_normal_z = np.zeros((self.nstream, self.nspan))

        warnings.warn('ATTENTION: radial and theta component of the camber normal vectors have been artificially made'
                      'positive to avoid problems. Check if it is correct for your case.')
        for istream in range(self.nstream):
            for ispan in range(self.nspan):
                self.camber_normal_r[istream, ispan] = np.abs(self.blade.normal_vectors_cyl[istream, ispan][0])
                self.camber_normal_theta[istream, ispan] = np.abs(self.blade.normal_vectors_cyl[istream, ispan][1])
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

    def circumferential_average(self, mode='cell centered', fix_borders=False, bfm=None, gauss_filter=False, threshold=25):
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
                # self.s[istream, ispan] = self.mass_average(self.data.s, idx, istream, ispan)
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

        if fix_borders:
            print("WARNING: borders have been artifically fixed")
            self.fix_borders()

        if gauss_filter:
            print("WARNING: the fields have been artificially smoothed")
            self.gauss_filtering()

        self.u_mag = np.sqrt(self.ur ** 2 + self.ut ** 2 + self.uz ** 2)
        self.ut_drag = self.config.get_omega_shaft()/self.config.get_reference_omega() * self.r_cg
        self.ut_rel = self.ut - self.ut_drag
        self.u_mag_rel = np.sqrt(self.ur ** 2 + self.ut_rel ** 2 + self.uz ** 2)
        self.u_meridional = np.sqrt(self.ur ** 2 + self.uz ** 2)
        self.M = self.u_mag / sqrt(self.config.get_fluid_gamma() * self.p / self.rho)
        self.M_rel = self.u_mag_rel / sqrt(self.config.get_fluid_gamma() * self.p / self.rho)
        self.compute_stagnation_quantities()
        if bfm == 'radial':
            print("WARNING: deprecated method, check the code")
            self.mu = self.compute_mu()
            self.F_t = self.mu * self.u_mag_rel ** 2

    def circumferential_average_interpolation(self):
        """
        Perform circumferential averages of the CFD dataset on the Block grid, based on 3d interpolation.
        """
        print_banner_begin("CIRCUMFERENTIAL AVG. METHOD")
        print(f"{'Averaging Method:':<{total_chars_mid}}{'3D Interpolation':>{total_chars_mid}}")
        print_banner_end()

        self.instantiate_2d_fields()
        print('performing circumferential averages...')

        # loop over all the elements in the meridional grid
        for istream in range(0, self.nstream):
            for ispan in range(0, self.nspan):
                # get the indexes of the elements used for the interpolation
                distance = ((self.data.z - self.z_cg[istream, ispan]) ** 2 + (self.data.r - self.r_cg[istream, ispan]) ** 2)

                if istream == self.nstream - 1 or ispan == self.nspan - 1:
                    ref_distance = np.sqrt((self.z_cg[istream, ispan] - self.z_cg[istream - 1, ispan]) ** 2 + \
                                           (self.r_cg[istream, ispan] - self.r_cg[istream, ispan - 1]) ** 2)
                else:
                    ref_distance = np.sqrt((self.z_cg[istream + 1, ispan] - self.z_cg[istream, ispan]) ** 2 + \
                                           (self.r_cg[istream, ispan + 1] - self.r_cg[istream, ispan + 1]) ** 2)
                idx = np.where(distance < ref_distance)
                points_considered = len(self.data.z[idx])
                assert (points_considered > 1000)

                # plt.figure()
                # plt.plot(self.z_cg[:, 0], self.r_cg[:, 0], 'black')
                # plt.plot(self.z_cg[:, -1], self.r_cg[:, -1], 'black')
                # plt.plot(self.z_cg[0, :], self.r_cg[0, :], 'black')
                # plt.plot(self.z_cg[-1, :], self.r_cg[-1, :], 'black')
                # plt.scatter(self.z_cg[istream, ispan], self.r_cg[istream, ispan])
                # theta = np.linspace(0, 2 * np.pi, 50)
                # x = self.z_cg[istream, ispan] + ref_distance * np.cos(theta)
                # y = self.r_cg[istream, ispan] + ref_distance * np.sin(theta)
                # plt.plot(x, y, 'red')

                # main quantities
                self.rho[istream, ispan] = self.three_dimensional_weighted_interpolation(self.data.rho, idx, istream, ispan)
                # self.ur[istream, ispan] = self.mass_average(self.data.ur, idx, istream, ispan)
                # self.ut[istream, ispan] = self.mass_average(self.data.ut, idx, istream, ispan)
                # self.uz[istream, ispan] = self.mass_average(self.data.uz, idx, istream, ispan)
                # self.p[istream, ispan] = self.mass_average(self.data.p, idx, istream, ispan)
                # self.T[istream, ispan] = self.mass_average(self.data.T, idx, istream, ispan)
                # self.s[istream, ispan] = self.mass_average(self.data.s, idx, istream, ispan)
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

        # self.u_mag = np.sqrt(self.ur ** 2 + self.ut ** 2 + self.uz ** 2)
        # self.ut_drag = self.config.get_omega_shaft() * self.r_cg
        # self.ut_rel = self.ut - self.ut_drag
        # self.u_mag_rel = np.sqrt(self.ur ** 2 + self.ut_rel ** 2 + self.uz ** 2)
        # self.u_meridional = np.sqrt(self.ur ** 2 + self.uz ** 2)
        # self.M = self.u_mag / sqrt(self.GAMMA * self.p / self.rho)
        # self.M_rel = self.u_mag_rel / sqrt(self.GAMMA * self.p / self.rho)
        # self.compute_stagnation_quantities()

        plt.figure()
        plt.contourf(self.z_cg, self.r_cg, self.rho, levels=N_levels)
        plt.colorbar()

    def compute_derived_quantities(self):
        """
        From the primary averaged fields, compute derived fields.
        """
        self.ut_drag = self.data.omega_shaft * self.r_cg
        self.ut_rel = self.ut - self.ut_drag

        check = np.abs(self.ut_drag + self.ut_rel - self.ut)
        if (check>1e-8).any():
            raise ValueError('Wrong decomposition')

        self.u_mag = np.sqrt(self.ur ** 2 + self.ut ** 2 + self.uz ** 2)
        self.u_mag_rel = np.sqrt(self.ur ** 2 + self.ut_rel ** 2 + self.uz ** 2)
        self.u_meridional = np.sqrt(self.ur ** 2 + self.uz ** 2)
        self.M = self.u_mag / sqrt(self.config.get_fluid_gamma() * self.p / self.rho)
        self.M_rel = self.u_mag_rel / sqrt(self.config.get_fluid_gamma() * self.p / self.rho)
        self.compute_stagnation_quantities()

    def compute_green_gauss_gradients(self):
        """
        Based on the dataset, compute the gradients on the physical grid nodes, based on the divergence theorem.
        Formula taken from Ferziger-Peritch book, eq. 9.51
        """

        self.drho_dz = np.zeros_like(self.z_cg)
        self.drho_dr = np.zeros_like(self.z_cg)
        for i in range(self.nstream):
            for j in range(self.nspan):
                area_element = self.block.area_elements[i,j]
                line_elements = self.block.area_elements[i,j].line_elements
                flux = 0
                for k,line in enumerate(line_elements):
                    z_mid = line.z_cg
                    r_mid = line.r_cg
                    f_mid = griddata((self.data.z, self.data.r), self.data.rho,
                                     (z_mid, r_mid), 'nearest')
                    flux += f_mid*line.l_orth
                flux /= area_element.area
                self.drho_dz[i,j] = flux[0]
                self.drho_dr[i, j] = flux[1]





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

        self.drho_dr = np.zeros((self.nstream, self.nspan))
        self.drho_dz = np.zeros((self.nstream, self.nspan))
        self.dur_dr = np.zeros((self.nstream, self.nspan))
        self.dur_dz = np.zeros((self.nstream, self.nspan))
        self.dut_dr = np.zeros((self.nstream, self.nspan))
        self.dut_dz = np.zeros((self.nstream, self.nspan))
        self.duz_dr = np.zeros((self.nstream, self.nspan))
        self.duz_dz = np.zeros((self.nstream, self.nspan))
        self.dp_dr = np.zeros((self.nstream, self.nspan))
        self.dp_dz = np.zeros((self.nstream, self.nspan))
        self.ds_dr = np.zeros((self.nstream, self.nspan))
        self.ds_dz = np.zeros((self.nstream, self.nspan))

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

        # bounding vertices, enlarged with the number of attempts already performed
        z1 = self.block.z_grid_dual[istream, ispan]  # bottom left corner
        z2 = self.block.z_grid_dual[istream + 1, ispan]  # bottom right corner
        z3 = self.block.z_grid_dual[istream + 1, ispan + 1]  # top right corner
        z4 = self.block.z_grid_dual[istream, ispan + 1]  # top left corner
        r1 = self.block.r_grid_dual[istream, ispan]
        r2 = self.block.r_grid_dual[istream + 1, ispan]
        r3 = self.block.r_grid_dual[istream + 1, ispan + 1]
        r4 = self.block.r_grid_dual[istream, ispan + 1]

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
        d = np.sqrt((r - r_c) ** 2 + (z - z_c) ** 2)  # distances of points from the center
        dmax = np.max(d)  # maximum distance of all points considered
        if RBF:
            weight = (1 + d / dmax) ** 5 * (1 - d / dmax) ** 5  # C4 RBF by Mendez
        else:
            weight = np.zeros_like(rho) + 1
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
        self.drho_dz = self.apply_gaussian_filter(self.drho_dz)
        self.dur_dr = self.apply_gaussian_filter(self.dur_dr)
        self.dur_dz = self.apply_gaussian_filter(self.dur_dz)
        self.dut_dr = self.apply_gaussian_filter(self.dut_dr)
        self.dut_dz = self.apply_gaussian_filter(self.dut_dz)
        self.duz_dr = self.apply_gaussian_filter(self.duz_dr)
        self.duz_dz = self.apply_gaussian_filter(self.duz_dz)
        self.dp_dr = self.apply_gaussian_filter(self.dp_dr)
        self.dp_dz = self.apply_gaussian_filter(self.dp_dz)
        self.ds_dr = self.apply_gaussian_filter(self.ds_dr)
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

    def compute_regressed_fields_chebyshev(self,):
        """
        Compute the fourth order polynomial regressed fields, as described in the original papers
        :param order: order of the regression. 4 is the values used in the literature.
        """
        print("Chebyshev Regression of the Flow Fields, orders: [4, 4]")
        self.W = chebvander2d(self.z_cg.flatten(), self.r_cg.flatten(), [8, 8])
        self.W_dz, self.W_dr = compute_derivative_matrices_chebyshev([8, 8], self.z_cg.flatten(), self.r_cg.flatten())

        def compute_solutions(field):
            coeffs, _, _, _ = np.linalg.lstsq(self.W, field.flatten(), rcond=None)
            f = np.dot(self.W, coeffs).reshape(self.z_cg.shape)
            df_dz = np.dot(self.W_dz, coeffs).reshape(self.z_cg.shape)
            df_dr = np.dot(self.W_dr, coeffs).reshape(self.z_cg.shape)
            return f, df_dr, df_dz

        self.rho, self.drho_dr, self.drho_dz = compute_solutions(self.rho)
        self.ur, self.dur_dr, self.dur_dz = compute_solutions(self.ur)
        self.ut, self.dut_dr, self.dut_dz = compute_solutions(self.ut)
        self.uz, self.duz_dr, self.duz_dz = compute_solutions(self.uz)
        self.p, self.dp_dr, self.dp_dz = compute_solutions(self.p)
        self.T, self.dT_dr, self.dT_dz = compute_solutions(self.T)
        self.s, self.ds_dr, self.ds_dz = compute_solutions(self.s)

    def polynomial_regression_solution(self, field):
        """
        Given a 2D field, and the weight vector coefficients, compute the values of the regressed field and derivatives.
        :param field: 2D array storing the values of the field to be regressed.
        """
        Nz = np.shape(self.z_cg)[0]
        Nr = np.shape(self.r_cg)[1]
        coeff_vector = least_square_regression(self.W, field)
        regr_field = regression_evaluation(self.W, coeff_vector, Nz, Nr)
        regr_field_dz = regression_evaluation(self.W_dz, coeff_vector, Nz, Nr)
        regr_field_dr = regression_evaluation(self.W_dr, coeff_vector, Nz, Nr)
        return regr_field, regr_field_dr, regr_field_dz

    def compute_rbf_gradients(self):
        """
        Compute the gradients of the relevant fields, using RBF interpolation in 2D and then finite differences
        """
        print("WARNING: deprecated method.")
        self.drho_dr, self.drho_dz = self.rbf_finite_difference(self.rho)
        self.dur_dr, self.dur_dz = self.rbf_finite_difference(self.ur)
        self.dut_dr, self.dut_dz = self.rbf_finite_difference(self.ut)
        self.duz_dr, self.duz_dz = self.rbf_finite_difference(self.uz)
        self.dp_dr, self.dp_dz = self.rbf_finite_difference(self.p)
        self.dT_dr, self.dT_dz = self.rbf_finite_difference(self.T)
        self.ds_dr, self.ds_dz = self.rbf_finite_difference(self.s)

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

        # find the step dz,dr based on the minimum streamwise-spanwise steps, divided by a diff_factor
        dz_min = np.min(np.abs(self.z_cg[1, :] - self.z_cg[0, :]))
        for ii in range(1, self.nstream - 1):
            tmp = np.min(np.abs(self.z_cg[ii + 1, :] - self.z_cg[ii, :]))
            if tmp < dz_min:
                dz_min = tmp

        dr_min = np.min(np.abs(self.r_cg[:, 1] - self.r_cg[:, 0]))
        for jj in range(1, self.nspan - 1):
            tmp = np.min(np.abs(self.r_cg[:, jj + 1] - self.r_cg[:, jj]))
            if tmp < dr_min:
                dr_min = tmp

        diff_factor = 5
        dz = dz_min / diff_factor
        dr = dr_min / diff_factor

        z_plus = self.z_cg + dz
        z_minus = self.z_cg - dz
        r_plus = self.r_cg + dr
        r_minus = self.r_cg - dr

        # Perform the RBF interpolation of the left points
        # field_interp = rbf(self.z_cg, self.r_cg)
        field_interp_right = rbf(z_plus, self.r_cg)
        field_interp_left = rbf(z_minus, self.r_cg)
        field_interp_up = rbf(self.z_cg, r_plus)
        field_interp_down = rbf(self.z_cg, r_minus)
        dfield_dz = ((field_interp_right - field_interp_left) / (2 * dz))
        dfield_dr = ((field_interp_up - field_interp_down) / (2 * dr))

        return dfield_dr, dfield_dz

    def tangent_finite_difference_gradient(self, field):
        """
        Computes the gradients of field based on the gradient obtained with tangent vectors.
        :param field: field used
        """
        import math
        dfield_dr, dfield_dz = np.zeros_like(self.z_cg), np.zeros_like(self.z_cg)
        for i in range(self.nstream):
            for j in range(self.nspan):
                if (i == 0 and j == 0):
                    im = 0
                    ip = 1
                    jm = 0
                    jp = 1
                elif (i == 0 and j == self.nspan - 1):
                    im = 0
                    ip = 1
                    jm = -1
                    jp = 0
                elif (i == self.nstream - 1 and j == 0):
                    im = -1
                    ip = 0
                    jm = 0
                    jp = 1
                elif (i == self.nstream - 1 and j == self.nspan - 1):
                    im = -1
                    ip = 0
                    jm = -1
                    jp = 0
                elif i == 0:
                    im = 0
                    ip = 1
                    jm = -1
                    jp = 1
                elif i == self.nstream - 1:
                    im = -1
                    ip = 0
                    jm = -1
                    jp = 1
                elif j == 0:
                    im = -1
                    ip = 1
                    jm = 0
                    jp = 1
                elif j == self.nspan - 1:
                    im = -1
                    ip = 1
                    jm = -1
                    jp = 0
                else:
                    im = -1
                    ip = 1
                    jm = -1
                    jp = 1

                # streamwise
                dz_st = self.z_cg[i + ip, j] - self.z_cg[i + im, j]
                dr_st = self.r_cg[i + ip, j] - self.r_cg[i + im, j]
                dl_st = sqrt(dz_st ** 2 + dr_st ** 2)
                alpha_st = np.arctan2(dr_st, dz_st)
                df_dst = (field[i + ip, j] - field[i + im, j]) / dl_st
                st_vers = np.array([[dz_st, dr_st]]) / dl_st

                # spanwise
                dz_sp = self.z_cg[i, j + jp] - self.z_cg[i, j + jm]
                dr_sp = self.r_cg[i, j + jp] - self.r_cg[i, j + jm]
                dl_sp = sqrt(dz_sp ** 2 + dr_sp ** 2)
                alpha_sp = np.arctan2(dr_sp, dz_sp)
                df_dsp = (field[i, j + jp] - field[i, j + jm]) / dl_sp
                sp_vers = np.array([[dz_sp, dr_sp]]) / dl_sp

                df_vect = np.array([[df_dst],
                                    [df_dsp]])

                A = np.concatenate((st_vers, sp_vers), axis=0)
                df_dxdy = np.linalg.inv(A) @ df_vect
                dfield_dz[i, j] = df_dxdy[0]
                dfield_dr[i, j] = df_dxdy[1]

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, field, levels=N_levels, cmap=color_map)
        plt.title('field')
        plt.colorbar()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, dfield_dz, levels=N_levels, cmap=color_map)
        plt.title('d/dz')
        plt.colorbar()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, dfield_dr, levels=N_levels, cmap=color_map)
        plt.title('d/dr')
        plt.colorbar()

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

    def plot_stream_line(self, field, n, save_filename=None, folder_name='pictures'):
        """
        Plot the quantity along a streamline.
        :param field: quantitiy to plot
        :param n: streamline to consider.
        :param save_filename: if specified, saves the figure
        :param folder_name: name of the folder
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
        elif field == 'M':
            ax.plot(self.stream_line_length[:, n] / sl_max, self.M[:, n], '--s')
            ax.set_ylabel(r'$M \ \mathrm{[-]}$')
        elif field == 'M_rel':
            ax.plot(self.stream_line_length[:, n] / sl_max, self.M_rel[:, n], '--s')
            ax.set_ylabel(r'$M_{rel} \ \mathrm{[-]}$')
        elif field == 'F_turn':
            ax.plot(self.stream_line_length[:, n] / sl_max, np.abs(self.Fturn[:, n]), '--s')
            ax.set_ylabel(r'$|F_{t}| \ \mathrm{[-]}$')
            ax.set_title('Span %.1f' % (n / self.nspan * 100))
        else:
            raise ValueError("Field name unknown!")

        ax.grid(alpha=0.3)
        ax.set_xlabel(r'$l \ \mathrm{[-]}$')
        if save_filename is not None:
            fig.savefig(folder_name + '/' + save_filename + '.pdf', bbox_inches='tight')
            plt.close()

    def plot_stream_line_superposed(self, field, n_array, save_filename=None, folder_name='pictures'):
        """
        Plot the quantity along a streamline.
        :param field: quantitiy to plot
        :param n_array: array of the streamlines to consider.
        :param save_filename: if specified, saves the figure
        :param folder_name: name of the folder
        """
        fig, ax = plt.subplots()
        if field == 'F_turn':
            for n in n_array:
                sl_max = self.stream_line_length[:, n].max()
                ax.plot(self.stream_line_length[:, n] / sl_max, np.abs(self.Fturn[:, n]), '--s',
                        label='%.1f %%' % (n / self.nspan * 100))
            ax.set_ylabel(r'$|F_{t}| \ \mathrm{[-]}$')
        elif field == 'F_loss':
            for n in n_array:
                sl_max = self.stream_line_length[:, n].max()
                ax.plot(self.stream_line_length[:, n] / sl_max, np.abs(self.Floss[:, n]), '--s',
                        label='%.1f %%' % (n / self.nspan * 100))
            ax.set_ylabel(r'$F_{l} \ \mathrm{[-]}$')
        else:
            raise ValueError("Field name unknown!")
        plt.legend()
        ax.grid(alpha=0.3)
        ax.set_xlabel(r'$l \ \mathrm{[-]}$')
        if save_filename is not None:
            fig.savefig(folder_name + '/' + save_filename + '_streamlines.pdf', bbox_inches='tight')
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

    def plot_spanline(self, field, n, save_filename=None, xlim=None, folder_name='pictures'):
        """
        Plot the quantity along a spanline.
        :param field: quantitiy to plot
        :param n: streamline to consider.
        :param save_filename: if specified, saves the figure
        :param folder_name: name of the folder
        """

        fig, ax = plt.subplots(figsize=self.picture_size_blank)
        if field == 'rho':
            ax.plot(self.rho[n, :], self.span_wise_length[n, :] / self.span_wise_length[n, :].max(), '--s')
            ax.set_xlabel(r'$\rho \ \mathrm{[kg/m^3]}$')
        elif field == 'ur':
            ax.plot(self.ur[n, :], self.span_wise_length[n, :] / self.span_wise_length[n, :].max(), '--s')
            ax.set_xlabel(r'$u_r \ \mathrm{[m/s]}$')
        elif field == 'ut':
            ax.plot(self.ut[n, :], self.span_wise_length[n, :] / self.span_wise_length[n, :].max(), '--s')
            ax.set_xlabel(r'$u_t \ \mathrm{[m/s]}$')
        elif field == 'uz':
            ax.plot(self.uz[n, :], self.span_wise_length[n, :] / self.span_wise_length[n, :].max(), '--s')
            ax.set_xlabel(r'$u_z \ \mathrm{[m/s]}$')
        elif field == 'p':
            ax.plot(self.p[n, :], self.span_wise_length[n, :] / self.span_wise_length[n, :].max(), '--s')
            ax.set_xlabel(r'$p \ \mathrm{[Pa]}$')
        elif field == 'p_tot':
            ax.plot(self.p_tot[n, :], self.span_wise_length[n, :] / self.span_wise_length[n, :].max(), '--s')
            ax.set_xlabel(r'$p_{t} \ \mathrm{[-]}$')
        elif field == 'p_tot_ratio':
            ax.plot(self.p_tot[n, :] / (101325 / self.config.get_reference_pressure()),
                    self.span_wise_length[n, :] / self.span_wise_length[n, :].max(), '--s')
            ax.set_xlabel(r'$p_{t} \ \mathrm{[-]}$')
        elif field == 'T_tot_ratio':
            ax.plot(self.T_tot[n, :] / (288.15 / self.config.get_reference_temperature()),
                    self.span_wise_length[n, :] / self.span_wise_length[n, :].max(), '--s')
            ax.set_xlabel(r'$T_{t} \ \mathrm{[-]}$')

        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_ylabel(r'$s \ \mathrm{[-]}$')
        if save_filename is not None:
            fig.savefig(folder_name + '/' + save_filename + '.pdf', bbox_inches='tight')
            plt.close()

    def plot_span_line_superposed(self, field, n_array, save_filename=None, folder_name='pictures'):
        """
        Plot the quantity along an array of spanlines.
        :param field: quantitiy to plot
        :param n_array: streamline to consider.
        :param save_filename: if specified, saves the figure
        :param folder_name: folder name
        """

        fig, ax = plt.subplots()
        if field == 'rho':
            for n in n_array:
                ax.plot(self.rho[n, :], '--s', self.span_wise_length[n, :], label='%.1f %%' % (n / self.nstream * 100))
            ax.set_xlabel(r'$\rho \ \mathrm{[kg/m^3]}$')
        if field == 'F_loss':
            for n in n_array:
                sp_max = self.span_wise_length[n, :].max()
                ax.plot(self.Floss[n, :], self.span_wise_length[n, :] / sp_max, '--s', label='%.1f %%' % (n / self.nstream * 100))
            ax.set_xlabel(r'$F_{l} \ \mathrm{[-]}$')

        ax.set_ylabel(r'$s \ \mathrm{[-]}$')
        plt.legend()
        if save_filename is not None:
            fig.savefig(folder_name + '/' + save_filename + '_spanlines.pdf', bbox_inches='tight')
            plt.close()

    def contour_plot(self, field, save_filename=None, unit_factor=1, quiver=False):
        """
        Contour plot of a 2D field.
        :param field: field to plot
        :param save_filename: if specified, saves the figure
        :param unit_factor: factor to be used for dimensional plots
        :param quiver: if True, superposes the quiver plots of the meridional velocity
        """

        if self.config.get_normalize_data():
            self.contour_plot_non_dimensional(field, save_filename, quiver)
        else:
            self.contour_plot_dimensional(field, save_filename, unit_factor, quiver)

    def contour_all_plots(self, save_filename, additional_field=False):
        """
        call all the contour plots
        """
        self.contour_plot(field='rho', save_filename=save_filename + '_rho')
        self.contour_plot(field='ur', save_filename=save_filename + '_ur')
        self.contour_plot(field='ut', save_filename=save_filename + '_ut')
        self.contour_plot(field='ut_rel', save_filename=save_filename + '_ut_rel')
        self.contour_plot(field='ut_drag', save_filename=save_filename + '_ut_drag')
        self.contour_plot(field='uz', save_filename=save_filename + '_uz')
        self.contour_plot(field='p', save_filename=save_filename + '_p')
        self.contour_plot(field='s', save_filename=save_filename + '_s')
        self.contour_plot(field='T', save_filename=save_filename + '_T')
        self.contour_plot(field='p_tot', save_filename=save_filename + '_p_tot')
        self.contour_plot(field='T_tot', save_filename=save_filename + '_T_tot')
        self.contour_plot(field='M', save_filename=save_filename + '_M')

        if additional_field:
            self.contour_plot(field='drho_dr', save_filename=save_filename + '_drho_dr')
            self.contour_plot(field='drho_dz', save_filename=save_filename + '_drho_dz')
            self.contour_plot(field='dur_dr', save_filename=save_filename + '_dur_dr')
            self.contour_plot(field='dur_dz', save_filename=save_filename + '_dur_dz')
            self.contour_plot(field='dut_dr', save_filename=save_filename + '_dut_dr')
            self.contour_plot(field='dut_dz', save_filename=save_filename + '_dut_dz')
            self.contour_plot(field='duz_dr', save_filename=save_filename + '_duz_dr')
            self.contour_plot(field='duz_dz', save_filename=save_filename + '_duz_dz')
            self.contour_plot(field='dp_dr', save_filename=save_filename + '_dp_dr')
            self.contour_plot(field='dp_dz', save_filename=save_filename + '_dp_dz')
            self.contour_plot(field='ds_dr', save_filename=save_filename + '_ds_dr')
            self.contour_plot(field='ds_dz', save_filename=save_filename + '_ds_dz')
            self.contour_plot(field='p_tot_bar', save_filename=save_filename + '_p_tot_bar')


    def contour_plot_dimensional(self, field, save_filename=None, unit_factor=1, quiver=False, folder_name='pictures'):
        """
        Contour plot of a 2D field.
        :param field: field to plot
        :param save_filename: if specified, saves the figure
        :param unit_factor: factor to be used for dimensional plots
        :param quiver: if True, superposes the quiver plots of the meridional velocity
        :param folder_name: folder name
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
            fig.savefig(folder_name + '/' + save_filename + '.pdf', bbox_inches='tight')
            plt.close()

    def contour_plot_non_dimensional(self, field, save_filename=None, quiver=False, folder_name='pictures'):
        """
        Contour plot of a 2D field.
        :param field: field to plot
        :param save_filename: if specified, saves the figure
        :param quiver: if True, superposes the quiver plots of the meridional velocity
        """
        fig, ax = plt.subplots(figsize=self.picture_size_contour)

        # PRIMARY FIELDS
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

        # GRADIENTS
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

        # AUXILLIARY FIELDS
        elif field == 'um':
            cs = ax.contourf(self.z_cg, self.r_cg, self.u_meridional, N_levels, cmap=color_map)
            ax.set_title(r'$u_m$')
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
        elif field == 'streamwise length':
            cs = ax.contourf(self.z_cg, self.r_cg, self.stream_line_length,
                             levels=N_levels, cmap=color_map)
            ax.set_title(r'$s_{stwl}$')
            cb = fig.colorbar(cs)
            cb.set_label(r'$\mathrm{[-]}$')
        elif field == 'spanwise length':
            cs = ax.contourf(self.z_cg, self.r_cg, self.span_wise_length,
                             levels=N_levels, cmap=color_map)
            ax.set_title(r'$s_{spwl}$')
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


        # BODY FORCE FIELDS
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

    def compute_stagnation_quantities(self):
        """
        Compute the 2D fields of the stagnation quantities
        """
        GAMMA = self.config.get_fluid_gamma()
        self.p_tot = self.p * (1 + (GAMMA - 1) / 2 * self.M ** 2) ** (GAMMA / (GAMMA - 1))
        self.T_tot = self.T * (1 + (GAMMA - 1) / 2 * self.M ** 2)

        # rotary total pressure, as defined by Sun et al. (centrifugal compressor analysis 2016)
        self.p_tot_bar = (self.p_tot - self.rho * self.r_cg * self.config.get_omega_shaft()/
                          self.config.get_reference_omega() * self.ut)

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
            folder = ''
        if file_name is None:
            file_name = 'meridional_process_%d_%d.pickle' % (self.nstream, self.nspan)

        with open(folder + file_name + '.pickle', "wb") as file:
            pickle.dump(self, file)

    def compute_bfm_axial(self, mode='averaged', save_fig=False):
        """
        Compute the BFM fields, following the description in Fang et al. 2023.
        :param mode: averaged uses the average inlet-outlet quantities to compute the gradients. The local uses the local values.
        :param save_fig: if specified, saves the figure
        """
        self.compute_Floss(mode=mode)
        self.compute_Ftheta(mode=mode)
        self.compute_Fturn()
        self.alpha = self.Floss / (self.u_mag_rel ** 2)
        self.beta = self.Fturn_t / (self.u_meridional * self.ut_rel)

        # plt.figure(figsize=self.picture_size_contour)
        # plt.contourf(self.z_cg, self.r_cg, self.ds_dl, cmap=color_map, levels=N_levels)
        # plt.colorbar()
        # plt.title(r'$\partial s / \partial l$')
        # if save_fig:
        #     plt.savefig('pictures/ds_dl_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
        #     # plt.close()

        # plt.figure(figsize=self.picture_size_contour)
        # plt.contourf(self.z_cg, self.r_cg, self.drut_dl, cmap=color_map, levels=N_levels)
        # plt.colorbar()
        # plt.title(r'$\partial (r u_{\theta}) / \partial l$')
        # if save_fig:
        #     plt.savefig('pictures/drut_dl_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
        #     # plt.close()

        # plt.figure(figsize=self.picture_size_contour)
        # plt.contourf(self.z_cg, self.r_cg, self.u_meridional, cmap=color_map, levels=N_levels)
        # plt.colorbar()
        # plt.title(r'$\hat{u}_{m}$')
        # if save_fig:
        #     plt.savefig('pictures/u_meridional_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
        #     # plt.close()

        # plt.figure(figsize=self.picture_size_contour)
        # plt.contourf(self.z_cg, self.r_cg, self.ds_dl, cmap=color_map, levels=N_levels)
        # plt.colorbar()
        # plt.title(r'$\frac{\partial s}{\partial m}$')
        # if save_fig:
        #     plt.savefig('pictures/ds_dl_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
        #     # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.Floss, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$F_{l}$')
        if save_fig:
            plt.savefig('pictures/F_loss_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            # plt.close()

        # plt.figure(figsize=self.picture_size_contour)
        # plt.contourf(self.z_cg, self.r_cg, self.Floss_r, cmap=color_map, levels=N_levels)
        # plt.colorbar()
        # plt.title(r'$F_{l,r}$')
        # if save_fig:
        #     plt.savefig('pictures/Fl_r_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
        #     # plt.close()

        # plt.figure(figsize=self.picture_size_contour)
        # plt.contourf(self.z_cg, self.r_cg, self.Floss_t, cmap=color_map, levels=N_levels)
        # plt.colorbar()
        # plt.title(r'$F_{l,\theta}$')
        # if save_fig:
        #     plt.savefig('pictures/Fl_t_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
        #     # plt.close()

        # plt.figure(figsize=self.picture_size_contour)
        # plt.contourf(self.z_cg, self.r_cg, self.Floss_z, cmap=color_map, levels=N_levels)
        # plt.colorbar()
        # plt.title(r'$F_{l,z}$')
        # if save_fig:
        #     plt.savefig('pictures/Fl_z_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
        #     # plt.close()

        # plt.figure(figsize=self.picture_size_contour)
        # plt.contourf(self.z_cg, self.r_cg, self.drut_dl, cmap=color_map, levels=N_levels)
        # plt.colorbar()
        # plt.title(r'$\frac{\partial (r u_{\theta})}{\partial m}$')
        # if save_fig:
        #     plt.savefig('pictures/drut_dl_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
        #     # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.Ftheta, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$F_{\theta}$')
        if save_fig:
            plt.savefig('pictures/F_theta_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            # plt.close()

        # plt.figure(figsize=self.picture_size_contour)
        # plt.contourf(self.z_cg, self.r_cg, self.Fturn_r, cmap=color_map, levels=N_levels)
        # plt.colorbar()
        # plt.title(r'$F_{t, r}$')
        # if save_fig:
        #     plt.savefig('pictures/Fturn_r_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
        #     # plt.close()

        # plt.figure(figsize=self.picture_size_contour)
        # plt.contourf(self.z_cg, self.r_cg, self.Fturn_t, cmap=color_map, levels=N_levels)
        # plt.colorbar()
        # plt.title(r'$F_{t, \theta}$')
        # if save_fig:
        #     plt.savefig('pictures/Fturn_t_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
        #     # plt.close()

        # plt.figure(figsize=self.picture_size_contour)
        # plt.contourf(self.z_cg, self.r_cg, self.Fturn_z, cmap=color_map, levels=N_levels)
        # plt.colorbar()
        # plt.title(r'$F_{t, z}$')
        # if save_fig:
        #     plt.savefig('pictures/Fturn_z_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
        #     # plt.close()

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.Fturn, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$F_{t}$')
        if save_fig:
            plt.savefig('pictures/F_turn_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
            # plt.close()

        # plt.figure(figsize=self.picture_size_contour)
        # plt.contourf(self.z_cg, self.r_cg, self.alpha, cmap=color_map, levels=N_levels)
        # plt.colorbar()
        # plt.title(r'$\alpha$')
        # if save_fig:
        #     plt.savefig('pictures/alpha_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
        #     # plt.close()

        # plt.figure(figsize=self.picture_size_contour)
        # plt.contourf(self.z_cg, self.r_cg, self.beta, cmap=color_map, levels=N_levels)
        # plt.colorbar()
        # plt.title(r'$\beta$')
        # if save_fig:
        #     plt.savefig('pictures/beta_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
        #     # plt.close()

    def compute_bfm_radial(self, save_fig=None):
        """
        Radial variation of the body force model found on the Chinese articles
        """
        F_ntheta = self.ur * (self.dut_dr + self.uz/self.ur*self.dut_dz +self.ut/self.r_cg)
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, F_ntheta, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$F_{n,\theta}$')
        if save_fig:
            plt.savefig('pictures/F_loss_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')

        F_t = np.zeros_like(self.rho)
        for ii in range(self.nstream):
            if ii<self.nstream-1:
                F_t[ii, :] = (self.p_tot_bar[ii+1, :] - self.p_tot_bar[ii, :]) / (self.s[ii+1, :] - self.s[ii, :]) / self.rho[ii, :]
            else:
                F_t[ii, :] = (self.p_tot_bar[ii, :] - self.p_tot_bar[ii-1, :]) / (self.s[ii, :] - self.s[ii-1, :]) / self.rho[ii, :]

        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, F_t, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$F_{t}$')
        if save_fig:
            plt.savefig('pictures/F_loss_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')
        
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, self.p_tot_bar, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title(r'$\bar{p_{t}}$')
        if save_fig:
            plt.savefig('pictures/F_loss_%d_%d.pdf' % (self.nstream, self.nspan), bbox_inches='tight')

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

    def contour_local_entropy_generation(self, save_fig=None):
        """
        Show the contour of the entropy generation, defined as the difference between the local entropy and the
        one that was at leading edge
        """
        z = self.z_cg
        r = self.r_cg
        s = self.s.copy()
        for istream in range(1, self.nstream):
            s[istream, :] = s[istream, :] - s[istream - 1, :]
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
            print("%s Domain" % (domain))

            # cosine directors of the loss force
            if (np.abs(self.u_mag_rel) < 1e-8).any():
                raise ValueError('Attention, division by small number')
            
            # cosine directors of the loss force
            tr = -self.ur / self.u_mag_rel
            ttheta = -self.ut_rel / self.u_mag_rel
            tz = -self.uz / self.u_mag_rel

            #cosine directors of the turning force
            nr = self.camber_normal_r.copy()
            ntheta = self.camber_normal_theta.copy()
            nz = self.camber_normal_z.copy()

            self.plot_bfm_angles(nr, ntheta, nz, tr, ttheta, tz)

            self.S00 = np.zeros_like(self.ur)
            self.S01 = np.zeros_like(self.ur)
            self.S02 = np.zeros_like(self.ur)
            self.S03 = np.zeros_like(self.ur)
            self.S04 = np.zeros_like(self.ur)

            self.S10 = np.zeros_like(self.ur)
            self.S11 = tr * 2 * self.alpha * self.ur + nr / ntheta * self.ur * self.beta * self.ut_rel / self.u_meridional
            self.S12 = tr * 2 * self.alpha * self.ut_rel + nr / ntheta * self.beta * self.u_meridional
            self.S13 = tr * 2 * self.alpha * self.uz + nr / ntheta * self.uz * self.beta * self.ut_rel / self.u_meridional
            self.S14 = np.zeros_like(self.ur)

            self.S20 = np.zeros_like(self.ur)
            self.S21 = ttheta * 2 * self.alpha * self.ur + self.ur * self.beta * self.ut_rel / self.u_meridional
            self.S22 = ttheta * 2 * self.alpha * self.ut_rel + self.beta * self.u_meridional
            self.S23 = ttheta * 2 * self.alpha * self.uz + self.uz * self.beta * self.ut_rel / self.u_meridional
            self.S24 = np.zeros_like(self.ur)

            self.S30 = np.zeros_like(self.ur)
            self.S31 = tz * 2 * self.alpha * self.ur + nz / ntheta * self.ur * self.beta * self.ut_rel / self.u_meridional
            self.S32 = tz * 2 * self.alpha * self.ut_rel + nz / ntheta * self.beta * self.u_meridional
            self.S33 = tz * 2 * self.alpha * self.uz + nz / ntheta * self.uz * self.beta * self.ut_rel / self.u_meridional
            self.S34 = np.zeros_like(self.ur)

            self.S40 = np.zeros_like(self.ur)
            self.S41 = self.S21*self.r_cg*self.config.get_omega_shaft()/self.config.get_reference_omega()
            self.S42 = self.S22*self.r_cg*self.config.get_omega_shaft()/self.config.get_reference_omega()
            self.S43 = self.S23*self.r_cg*self.config.get_omega_shaft()/self.config.get_reference_omega()
            self.S44 = np.zeros_like(self.ur)

        elif domain == 'unbladed':
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
        else:
            raise ValueError("Unknown domain type for body force calculation. Available choices: rotor, stator, unbladed.")

    def compute_Floss(self, mode):
        """
        Compute the Loss component of the body force
        """

        # meridional flow velocity
        self.u_meridional = np.sqrt(self.ur ** 2 + self.uz ** 2)
        self.compute_ds_dl(mode=mode)

        self.Floss = self.T * self.u_meridional * self.ds_dl / self.u_mag_rel

        if self.config.get_clipping_bfm():
            idx = np.where(self.Floss < 0)
            self.Floss[idx] = 0

        # compute the components, which are opposite to the relative velocity
        self.Floss_r = -self.Floss * self.ur / self.u_mag_rel
        self.Floss_t = -self.Floss * self.ut_rel / self.u_mag_rel
        self.Floss_z = -self.Floss * self.uz / self.u_mag_rel

        check = np.sqrt(self.Floss_r ** 2 + self.Floss_t ** 2 + self.Floss_z ** 2) - self.Floss
        # if (np.abs(check) > 1e-6).any():
        #     raise ValueError('The direction vector is not unitary')

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

        elif mode == 'local':
            # calculate ds_dl projecting the gradient of s over the direction in which the particle is going
            for istream in range(self.nAxialNodes):
                for ispan in range(self.nRadialNodes):
                    dir_vector = np.array((self.uz[istream, ispan],
                                           self.ur[istream, ispan]))
                    dir_vector /= np.linalg.norm(dir_vector)
                    self.ds_dl[istream, ispan] = self.ds_dz[istream, ispan] * dir_vector[0] + \
                                                 self.ds_dr[istream, ispan] * dir_vector[1]

    def compute_Ftheta(self, mode='averaged'):
        """
        Compute the modulus of the global theta component of the body force
        """
        if (np.abs(self.u_meridional) < 1e-8).any():
            raise ValueError('Attention, division by small number')
        dr_dl = self.ur / self.u_meridional

        dut_dl = np.zeros_like(dr_dl)
        if mode == 'local':
            # find the derivative projecting the gradients along the meridional velocity direction
            for istream in range(self.nAxialNodes):
                for ispan in range(self.nRadialNodes):
                    dir_vector = np.array((self.uz[istream, ispan],
                                           self.ur[istream, ispan]))
                    dir_vector /= np.linalg.norm(dir_vector)
                    dut_dl[istream, ispan] = self.dut_dz[istream, ispan] * dir_vector[0] + \
                                             self.dut_dr[istream, ispan] * dir_vector[1]
        elif mode == 'averaged':
            for ispan in range(self.nRadialNodes):
                dut_dl[:, ispan] = (self.ut[-1, ispan] - self.ut[0, ispan]) / self.stream_line_length[-1, ispan]
                dr_dl[:, ispan] = (self.r_cg[-1, ispan] - self.r_cg[0, ispan]) / self.stream_line_length[-1, ispan]
        else:
            raise ValueError("Mode not recognized")

        self.drut_dl = dr_dl * self.ut + self.r_cg * dut_dl
        self.Ftheta = self.u_meridional * self.drut_dl / self.r_cg

        if self.config.get_clipping_bfm():
            if self.config.get_omega_shaft() < 0:
                idx = np.where(self.Ftheta > 0)
                self.Ftheta[idx] = 0
            else:
                idx = np.where(self.Ftheta < 0)
                self.Ftheta[idx] = 0

    def compute_Fturn(self):
        """
        Starting from the Ftheta and camber normal vectors, compute the magnitude of the turning force
        """
        self.Fturn_t = self.Ftheta - self.Floss_t
        if self.config.get_clipping_bfm():
            if self.config.get_omega_shaft() < 0:
                idx = np.where(self.Fturn_t > 0)
                self.Fturn_t[idx] = 0
            else:
                idx = np.where(self.Fturn_t < 0)
                self.Fturn_t[idx] = 0

        if (np.abs(self.camber_normal_theta) < 1e-8).any():
            raise ValueError('Attention, division by small number')
        self.Fturn = self.Fturn_t / self.camber_normal_theta
        self.Fturn_r = self.Fturn * self.camber_normal_r
        self.Fturn_z = self.Fturn * self.camber_normal_z

    def plot_bfm_angles(self, nr, ntheta, nz, tr, ttheta, tz):
        """
        Plot the angles that define the direction of the body force
        :param nr: radial component of the turning force
        :param ntheta: theta component of the turning force
        :param nz: axial component of the turning force
        :param tr: radial component of the loss force
        :param ttheta: theta component of the loss force
        :param tz: axial component of the loss force
        """
        fig, ax = plt.subplots(1, 3, figsize=(14, 8))
        contour0 = ax[0].contourf(self.z_cg, self.r_cg, nr, levels=15)
        contour1 = ax[1].contourf(self.z_cg, self.r_cg, ntheta, levels=15)
        contour2 = ax[2].contourf(self.z_cg, self.r_cg, nz, levels=15)
        cbar0 = plt.colorbar(contour0)
        cbar1 = plt.colorbar(contour1)
        cbar2 = plt.colorbar(contour2)
        ax[0].set_title(r'$n_{r}$')
        ax[1].set_title(r'$n_{\theta}$')
        ax[2].set_title(r'$n_{z}$')
        fig.suptitle('Turning component cosine directors')
        for i in range(0, 3):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.savefig('pictures/turning_bfm_directions.pdf', bbox_inches='tight')

        fig, ax = plt.subplots(1, 3, figsize=(14, 8))
        contour0 = ax[0].contourf(self.z_cg, self.r_cg, tr, levels=15)
        contour1 = ax[1].contourf(self.z_cg, self.r_cg, ttheta, levels=15)
        contour2 = ax[2].contourf(self.z_cg, self.r_cg, tz, levels=15)
        cbar0 = plt.colorbar(contour0)
        cbar1 = plt.colorbar(contour1)
        cbar2 = plt.colorbar(contour2)
        ax[0].set_title(r'$l_{r}$')
        ax[1].set_title(r'$l_{\theta}$')
        ax[2].set_title(r'$l_{z}$')
        fig.suptitle('Loss component cosine directors')
        for i in range(0, 3):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.savefig('pictures/loss_bfm_directions.pdf', bbox_inches='tight')

        # fig, ax = plt.subplots(1, 2, figsize=(10, 8))
        # contour0 = ax[0].contourf(self.z_cg, self.r_cg, tr ** 2 + ttheta ** 2 + tz ** 2, levels=15)
        # contour1 = ax[1].contourf(self.z_cg, self.r_cg, nr ** 2 + ntheta ** 2 + nz ** 2, levels=15)
        # cbar0 = plt.colorbar(contour0)
        # cbar1 = plt.colorbar(contour1)
        # ax[0].set_title(r'$|l|$')
        # ax[1].set_title(r'$|n|$')
        # for i in range(0, 2):
        #     ax[i].set_xticks([])
        #     ax[i].set_yticks([])

    def compute_averaged_fluxes(self):
        """
        On the meridional plane, compute the averaged fluxed for each streamwise position.
        """
        self.dA = np.zeros_like(self.z_cg)
        self.dA_nz = np.zeros_like(self.z_cg)
        self.dA_nr = np.zeros_like(self.z_cg)
        # for istream in range(self.nstream):
        #     for ispan in range(self.nspan):
        #
        #         dz = self.z_grid[istream, ispan + 1] - self.z_grid[istream, ispan]
        #         dr = self.r_grid[istream, ispan + 1] - self.r_grid[istream, ispan]
        #
        #         # Area of the flux per unit length in circumferential direction
        #         self.dA[istream, ispan] = np.sqrt(dz ** 2 + dr ** 2)
        #
        #         # normal of the flux area (-90 deg rotation of the edge, normalized)
        #         self.dA_nz[istream, ispan] = dr / self.dA[istream, ispan]
        #         self.dA_nr[istream, ispan] = -dz / self.dA[istream, ispan]
        for ispan in range(self.nspan):
            if ispan == 0:
                isplus = 1
                isminus = 0
            elif ispan == self.nspan - 1:
                isplus = 0
                isminus = -1
            else:
                isplus = 1
                isminus = -1
            dz = (self.z_grid[:, ispan + isplus] - self.z_grid[:, ispan + isminus]) / 2
            dr = (self.r_grid[:, ispan + isplus] - self.r_grid[:, ispan + isminus]) / 2

            self.dA[:, ispan] = np.sqrt(dz ** 2 + dr ** 2)
            # normal of the flux area (-90 deg rotation of the edge, normalized)
            self.dA_nz[:, ispan] = dr / self.dA[:, ispan]
            self.dA_nr[:, ispan] = -dz / self.dA[:, ispan]

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

    def plot_averaged_fluxes(self, field, save_filename=None, folder_name='pictures'):
        """
        Plot the averaged fluxes.
        Contour plot of a 2D field.
        :param field: field to plot
        :param save_filename: if specified, saves the figure
        :param folder_name: folder name
        """
        sl_max = self.stream_line_length[:, 0].max()
        fig, ax = plt.subplots()
        if field == 'rho':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.rho_flux*self.config.get_reference_density(), '--s')
            ax.set_ylabel(r'$\rho \ \mathrm{[kg/m^3]}$')
        elif field == 'ur':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.ur_flux * self.config.get_reference_velocity(), '--s')
            ax.set_ylabel(r'$u_r \ \mathrm{[m/s]}$')
        elif field == 'ut':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.ut_flux * self.config.get_reference_velocity(), '--s')
            ax.set_ylabel(r'$u_t \ \mathrm{[m/s]}$')
        elif field == 'uz':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.uz_flux * self.config.get_reference_velocity(), '--s')
            ax.set_ylabel(r'$u_z \ \mathrm{[m/s]}$')
        elif field == 'M':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.M_flux, '--s')
            ax.set_ylabel(r'$M \ \mathrm{[-]}$')
        elif field == 'M_rel':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.M_rel_flux, '--s')
            ax.set_ylabel(r'$M_{rel} \ \mathrm{[-]}$')
        elif field == 'p':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.p_flux*self.config.get_reference_pressure(), '--s')
            ax.set_ylabel(r'$p \ \mathrm{[Pa]}$')
        elif field == 'T':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.T_flux*self.config.get_reference_temperature(), '--s')
            ax.set_ylabel(r'$T \ \mathrm{[K]}$')
        elif field == 's':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.s_flux, '--s')
            ax.set_ylabel(r'$s \ \mathrm{[-]}$')
        elif field == 'p_tot':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.p_tot_flux*self.config.get_reference_pressure(), '--s')
            ax.set_ylabel(r'$p_{t} \ \mathrm{[Pa]}$')
        elif field == 'T_tot':
            ax.plot(self.stream_line_length[:, 0] / sl_max, self.T_tot_flux*self.config.get_reference_temperature(), '--s')
            ax.set_ylabel(r'$T_{t} \ \mathrm{[K]}$')
        else:
            raise ValueError("Field name unknown!")

        ax.grid(alpha=0.3)
        ax.set_xlabel(r'$l \ \mathrm{[-]}$')
        if save_filename is not None:
            fig.savefig(folder_name + '/' + save_filename + '.pdf', bbox_inches='tight')

    def interpolate_on_working_grid(self):
        """
        Interpolate the 2d dataset on the working grid.
        """
        self.instantiate_2d_fields()
        method = self.config.get_cfd_interpolation_method()

        self.rho = self.interpolate_function(self.data.rho, self.data.z, self.data.r, method=method, return_type='field')
        self.ur = self.interpolate_function(self.data.ur, self.data.z, self.data.r, method=method, return_type='field')
        self.ut = self.interpolate_function(self.data.ut, self.data.z, self.data.r, method=method, return_type='field')
        self.uz = self.interpolate_function(self.data.uz, self.data.z, self.data.r, method=method, return_type='field')
        self.p = self.interpolate_function(self.data.p, self.data.z, self.data.r, method=method, return_type='field')
        self.T = self.interpolate_function(self.data.T, self.data.z, self.data.r, method=method, return_type='field')
        self.s = self.interpolate_function(self.data.s, self.data.z, self.data.r, method=method, return_type='field')

        try:
            self.drho_dr = self.interpolate_function(self.data.drho_dr, self.data.z, self.data.r, method=method,
                                                     return_type='field')
            self.drho_dz = self.interpolate_function(self.data.drho_dz, self.data.z, self.data.r, method=method,
                                                     return_type='field')
            self.dur_dr = self.interpolate_function(self.data.dur_dr, self.data.z, self.data.r, method=method,
                                                    return_type='field')
            self.dur_dz = self.interpolate_function(self.data.dur_dz, self.data.z, self.data.r, method=method,
                                                    return_type='field')
            self.dut_dr = self.interpolate_function(self.data.dut_dr, self.data.z, self.data.r, method=method,
                                                    return_type='field')
            self.dut_dz = self.interpolate_function(self.data.dut_dz, self.data.z, self.data.r, method=method,
                                                    return_type='field')
            self.duz_dr = self.interpolate_function(self.data.duz_dr, self.data.z, self.data.r, method=method,
                                                    return_type='field')
            self.duz_dz = self.interpolate_function(self.data.duz_dz, self.data.z, self.data.r, method=method,
                                                    return_type='field')
            self.dp_dr = self.interpolate_function(self.data.dp_dr, self.data.z, self.data.r, method=method, return_type='field')
            self.dp_dz = self.interpolate_function(self.data.dp_dz, self.data.z, self.data.r, method=method, return_type='field')
            self.ds_dr = self.interpolate_function(self.data.ds_dr, self.data.z, self.data.r, method=method, return_type='field')
            self.ds_dz = self.interpolate_function(self.data.ds_dz, self.data.z, self.data.r, method=method, return_type='field')
        except:
            pass

    def weight_least_square_regression(self):
        """
        weighted least square regression method applied to all the fields
        """
        self.instantiate_2d_fields()

        print("WLS regression of Density...")
        self.rho, self.drho_dz, self.drho_dr = self.approximate_with_weighted_least_square(self.data.rho, self.data.z,
                                                                                           self.data.r)

        print("WLS regression of Radial Velocity...")
        self.ur, self.dur_dz, self.dur_dr = self.approximate_with_weighted_least_square(self.data.ur, self.data.z, self.data.r)

        print("WLS regression of Tangential Velocity...")
        self.ut, self.dut_dz, self.dut_dr = self.approximate_with_weighted_least_square(self.data.ut, self.data.z, self.data.r)

        print("WLS regression of Axial Velocity...")
        self.uz, self.duz_dz, self.duz_dr = self.approximate_with_weighted_least_square(self.data.uz, self.data.z, self.data.r)

        print("WLS regression of Pressure...")
        self.p, self.dp_dz, self.dp_dr = self.approximate_with_weighted_least_square(self.data.p, self.data.z, self.data.r)

        print("WLS regression of Entropy...")
        self.s, self.ds_dz, self.ds_dr = self.approximate_with_weighted_least_square(self.data.s, self.data.z, self.data.r)

        print("WLS regression of Temperature...")
        self.T, self.dT_dz, self.dT_dr = self.approximate_with_weighted_least_square(self.data.T, self.data.z, self.data.r)

    def approximate_with_weighted_least_square(self, f_points, z_points, r_points):
        F = np.zeros_like(self.z_cg)
        dFdZ = np.zeros_like(self.z_cg)
        dFdR = np.zeros_like(self.z_cg)
        distance_limit = ((np.max(z_points) - np.min(z_points)) + (np.max(r_points) - np.min(r_points))) * 1000

        for ii in range(self.nstream):
            for jj in range(self.nspan):
                print("Regression %i of %i" % (jj + ii * self.nspan, self.nstream * self.nspan))
                distance = np.sqrt((self.z_cg[ii, jj] - z_points) ** 2 + (self.r_cg[ii, jj] - r_points) ** 2)
                idx = np.where(distance < distance_limit)
                F[ii, jj], dFdZ[ii, jj], dFdR[ii, jj] = compute_function_and_gradient_approximation(
                    self.z_cg[ii, jj], self.r_cg[ii, jj],
                    z_points[idx], r_points[idx], f_points[idx])

        return F, dFdZ, dFdR

    def compute_field_gradients(self):
        """
        compute the gradient of the flow field using a certain interpolation method, in order to evaluate functions at
        z+deltaz and r+deltar.
        """
        method = self.config.get_gradient_interpolation_method()

        if method == 'rbf':
            self.drho_dr, self.drho_dz = self.rbf_finite_difference(self.rho)
            self.dur_dr, self.dur_dz = self.rbf_finite_difference(self.ur)
            self.dut_dr, self.dut_dz = self.rbf_finite_difference(self.ut)
            self.duz_dr, self.duz_dz = self.rbf_finite_difference(self.uz)
            self.dp_dr, self.dp_dz = self.rbf_finite_difference(self.p)
            self.dT_dr, self.dT_dz = self.rbf_finite_difference(self.T)
            self.ds_dr, self.ds_dz = self.rbf_finite_difference(self.s)
        elif method == 'linear' or method == 'cubic':
            self.drho_dr, self.drho_dz = self.interpolation_gradient(self.data.rho, self.data.r, self.data.z, method=method)
            self.dur_dr, self.dur_dz = self.interpolation_gradient(self.data.ur, self.data.r, self.data.z, method=method)
            self.dut_dr, self.dut_dz = self.interpolation_gradient(self.data.ut, self.data.r, self.data.z, method=method)
            self.duz_dr, self.duz_dz = self.interpolation_gradient(self.data.uz, self.data.r, self.data.z, method=method)
            self.dp_dr, self.dp_dz = self.interpolation_gradient(self.data.p, self.data.r, self.data.z, method=method)
            self.dT_dr, self.dT_dz = self.interpolation_gradient(self.data.T, self.data.r, self.data.z, method=method)
            self.ds_dr, self.ds_dz = self.interpolation_gradient(self.data.s, self.data.r, self.data.z, method=method)
        elif method == 'tangent gradient':
            self.drho_dr, self.drho_dz = self.tangent_finite_difference_gradient(self.rho)
            self.dur_dr, self.dur_dz = self.tangent_finite_difference_gradient(self.ur)
            self.dut_dr, self.dut_dz = self.tangent_finite_difference_gradient(self.ut)
            self.duz_dr, self.duz_dz = self.tangent_finite_difference_gradient(self.uz)
            self.dp_dr, self.dp_dz = self.tangent_finite_difference_gradient(self.p)
            self.dT_dr, self.dT_dz = self.tangent_finite_difference_gradient(self.T)
            self.ds_dr, self.ds_dz = self.tangent_finite_difference_gradient(self.s)
        else:
            raise ValueError("Method not recognized.")

    def interpolation_gradient(self, f, r, z, method):
        """
        Linear interpolation method in order to compute the gradient of f(z,r)
        """
        visual_check = False
        dz_min = np.min(np.abs(self.z_cg[1, :] - self.z_cg[0, :]))
        for ii in range(1, self.nstream - 1):
            tmp = np.min(np.abs(self.z_cg[ii + 1, :] - self.z_cg[ii, :]))
            if tmp < dz_min:
                dz_min = tmp

        dr_min = np.min(np.abs(self.r_cg[:, 1] - self.r_cg[:, 0]))
        for jj in range(1, self.nspan - 1):
            tmp = np.min(np.abs(self.r_cg[:, jj + 1] - self.r_cg[:, jj]))
            if tmp < dr_min:
                dr_min = tmp

        diff_factor = 0.5
        dz = dz_min / diff_factor
        dr = dr_min / diff_factor

        Zplus = self.z_cg + dz
        Zminus = self.z_cg - dz
        Rplus = self.r_cg + dr
        Rminus = self.r_cg - dr

        f_interp = griddata((z, r), f, (self.z_cg, self.r_cg), method=method)
        f_zplus = griddata((z, r), f, (Zplus, self.r_cg), method=method)
        f_zminus = griddata((z, r), f, (Zminus, self.r_cg), method=method)
        f_rplus = griddata((z, r), f, (self.z_cg, Rplus), method=method)
        f_rminus = griddata((z, r), f, (self.z_cg, Rminus), method=method)
        df_dz = (f_zplus - f_zminus) / 2 / dz
        df_dr = (f_rplus - f_rminus) / 2 / dr

        if visual_check:
            plt.figure()
            plt.contourf(self.z_cg, self.r_cg, f_interp, cmap=color_map, levels=N_levels)
            plt.title('f')
            plt.colorbar()

            plt.figure()
            plt.contourf(self.z_cg, self.r_cg, df_dz, cmap=color_map, levels=N_levels)
            plt.title('dfdz')
            plt.colorbar()

            plt.figure()
            plt.contourf(self.z_cg, self.r_cg, df_dr, cmap=color_map, levels=N_levels)
            plt.title('dfdr')
            plt.colorbar()

        return df_dr, df_dz

    def interpolation_gradient_backup(self, f, r, z, method):
        """
        Linear interpolation method in order to compute the gradient of f(z,r)
        """
        visual_check = True


        Zplus = np.zeros_like(self.z_cg)
        # Zminus = np.zeros_like(self.z_cg)
        Rplus = np.zeros_like(self.z_cg)
        # Rminus = np.zeros_like(self.z_cg)

        Zplus[1:-2, 1:-2] = self.z_cg[2:-1, 1:-2]
        Zplus[0, :] = self.z_cg[0, :]
        Zplus[-1, :] = self.z_cg[-1, :]

        Rplus[1:-2, 1:-2] = self.r_cg[1:-2, 2:-1]
        Rplus[0, :] = self.r_cg[0, :]
        Rplus[-1, :] = self.r_cg[-1, :]

        f_interp = griddata((z, r), f, (self.z_cg, self.r_cg), method=method)
        f_zplus = griddata((z, r), f, (Zplus, self.r_cg), method=method)
        # f_zminus = griddata((z, r), f, (Zminus, self.r_cg), method=method)
        f_rplus = griddata((z, r), f, (self.z_cg, Rplus), method=method)
        # f_rminus = griddata((z, r), f, (self.z_cg, Rminus), method=method)
        df_dz = (f_zplus - f_interp) / (Zplus-self.z_cg)
        df_dz[0, :] = df_dz[1, :]
        df_dz[-1, :] = df_dz[-2, :]
        df_dr = (f_rplus - f_interp) / (Rplus-self.r_cg)
        df_dr[:, 0] = df_dr[:, 1]
        df_dr[:, -1] = df_dr[:, -2]

        if visual_check:
            plt.figure()
            plt.contourf(self.z_cg, self.r_cg, f_interp, cmap=color_map, levels=N_levels)
            plt.title('f')
            plt.colorbar()

            plt.figure()
            plt.contourf(self.z_cg, self.r_cg, f_zplus, cmap=color_map, levels=N_levels)
            plt.title('f_zplus')
            plt.colorbar()

            plt.figure()
            plt.contourf(self.z_cg, self.r_cg, f_rplus, cmap=color_map, levels=N_levels)
            plt.title('f_rplus')
            plt.colorbar()

            plt.figure()
            plt.contourf(self.z_cg, self.r_cg, df_dz, cmap=color_map, levels=N_levels)
            plt.title('dfdz')
            plt.colorbar()

            plt.figure()
            plt.contourf(self.z_cg, self.r_cg, df_dr, cmap=color_map, levels=N_levels)
            plt.title('dfdr')
            plt.colorbar()

        return df_dr, df_dz

    def interpolate_function(self, f, z, r, method, return_type='all'):
        """
        Interpolate the unstructured dataset f(z,r) on the working grid of the analysis.
        :param f: function values
        :param z: x or z cordinate of dataset at which is evaluated
        :param r: y or r cordinate of dataset at which is evaluated
        :param method: order of the interpolation
        :param return_type: specify if you want to return only the field or also the gradients computed on it
        """
        Xnew = self.z_cg  # original grid
        Ynew = self.r_cg  # original grid
        dx = np.abs(np.max(z) - np.min(z)) / 20
        dy = np.abs(np.max(r) - np.min(r)) / 20

        if method != 'rbf':
            f_new = griddata((z, r), f, (Xnew, Ynew), method=method)

            # take care of possible nan valus, due to extrapolation compared to dataset of points
            contains_nan = np.isnan(f_new).any()
            if contains_nan:
                nan_rows, nan_cols = np.where(np.isnan(f_new))
                for i in nan_rows:
                    for j in nan_cols:
                        f_new[i, j] = griddata((z, r), f, (Xnew[i, j], Ynew[i, j]), 'nearest')

            if return_type == 'all':
                f_new_dx = griddata((z, r), f, (Xnew + dx, Ynew), method=method)
                f_new_dy = griddata((z, r), f, (Xnew, Ynew + dy), method=method)

        else:
            rbf = Rbf(z, r, f, function='linear')
            f_new = rbf(Xnew, Ynew)
            if return_type == 'all':
                f_new_dx = rbf(Xnew + dx, Ynew)
                f_new_dy = rbf(Xnew, Ynew + dy)

        try:
            df_dx_new = (f_new_dx - f_new) / dx
            df_dy_new = (f_new_dy - f_new) / dy
            return f_new, df_dx_new, df_dy_new
        except:
            return f_new

    def compute_meridional_area(self):
        """
        compute the meridional area of the block
        """
        pass

    def check_bfm_local(self):
        """
        Check that the fields obtained in the bfm reflect the actual governing equations.
        """
        F_theta_check = self.u_meridional / self.r_cg * self.drut_dl - self.Ftheta
        F_loss_check = self.T * self.u_meridional / self.u_mag_rel * self.ds_dl - self.Floss

        plt.figure()
        plt.contourf(self.z_cg, self.r_cg, F_theta_check, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title('F theta check')

        plt.figure()
        plt.contourf(self.z_cg, self.r_cg, F_loss_check, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.title('F loss check')

    def compute_mass_flow_rate(self):
        """
        For each element compute the mass flow rate, defined as the quantity of mass per time
        """
        self.mass_flow = np.zeros_like(self.z_cg)
        for ii in range(self.nstream):
            for jj in range(self.nspan):

                # mid-point rule. Center of the flux area
                rp = self.block.area_elements[ii, jj].line_elements[1].r_cg
                zp = self.block.area_elements[ii, jj].line_elements[1].z_cg

                if ii < self.nstream - 1:
                    d_tot = np.sqrt(
                        (self.z_cg[ii + 1, jj] - self.z_cg[ii, jj]) ** 2 + (self.r_cg[ii + 1, jj] - self.r_cg[ii, jj]) ** 2)
                    d_partial = np.sqrt((self.z_cg[ii, jj] - zp) ** 2 + (self.r_cg[ii, jj] - rp) ** 2)
                    lmbda = 1 - d_partial / d_tot
                    rho = lmbda * self.rho[ii, jj] + (1 - lmbda) * self.rho[ii + 1, jj]
                    u = np.array([lmbda * self.uz[ii, jj] + (1 - lmbda) * self.uz[ii + 1, jj],
                                  lmbda * self.ur[ii, jj] + (1 - lmbda) * self.ur[ii + 1, jj]])
                    blockage = lmbda * self.blade.blockage[ii, jj] + (1 - lmbda) * self.blade.blockage[ii + 1, jj]
                else:
                    rho = self.rho[ii, jj]
                    u = np.array([self.uz[ii, jj],
                                  self.ur[ii, jj]])
                    blockage = self.blade.blockage[ii, jj]

                dl_vec = self.block.area_elements[ii, jj].line_elements[1].l_orth
                self.mass_flow[ii, jj] = (dl_vec @ u) * rho * 2 * np.pi * rp * blockage

    def compute_mass_flow_in_out(self):
        """
        For each element compute the mass flow rate in and out of the domain
        """
        conversion_factor = self.config.get_reference_density() * self.config.get_reference_velocity() * self.config.get_reference_length() ** 2
        self.mass_flow_in = np.sum(self.mass_flow[0, :]) * conversion_factor
        self.mass_flow_out = np.sum(self.mass_flow[-2, :]) * conversion_factor

        print('Inlet mass flow: %.3f [kg/s]' % (self.mass_flow_in))
        print('Outlet mass flow: %.3f [kg/s]' % (self.mass_flow_out))

    def check_mass_flow_streamwise(self):
        """
        For each stream position, compute the mass flow. Then check all of them together
        """
        conversion_factor = self.config.get_reference_density() * self.config.get_reference_velocity() * self.config.get_reference_length() ** 2
        mdot = np.array([np.sum(self.mass_flow[i, :]) for i in range(self.nstream)]) * conversion_factor
        plt.figure()
        plt.plot([i for i in range(self.nstream)], mdot, '-ko')
        plt.xlabel('streamwise position index')
        plt.ylabel(r'$\dot{m} \ \mathrm{[kg/s]}$')
        plt.grid(alpha=0.2)

    def check_bfm_global(self):
        """
        Check if the momentum equation is respected when integrating it over all the blade swept volume. The reference
        equations to be check are the eqs. 3.1 in Benneke MSc thesis.
        convection_term + pressure_term = force_term should be satisfied
        """
        force_term = self.compute_global_force()
        pressure_term = self.compute_pressure_term()
        convection_term = self.compute_convection_term()
        residual = ((convection_term + pressure_term - force_term) / force_term)

        print_banner_begin("BFM RESIDUALS CHECK")
        print(f"{'Radial Momentum Balance:':<{total_chars_mid}}{residual[0]:>{total_chars_mid}.2f}")
        print(f"{'Tangential Momentum Balance:':<{total_chars_mid}}{residual[1]:>{total_chars_mid}.2f}")
        print(f"{'Axial Momentum Balance:':<{total_chars_mid}}{residual[2]:>{total_chars_mid}.2f}")
        print_banner_end()

        # width = 0.25  # the width of the bars
        # multiplier = 0
        # names = ("Radial", "Tangential", "Axial")
        # terms = {'Pressure Term': pressure_term,
        #          'Convection Term': convection_term,
        #          'BFM Term': force_term}
        # x = np.arange(len(names))
        # fig, ax = plt.subplots(layout='constrained')
        #
        # for attribute, measurement in terms.items():
        #     offset = width * multiplier
        #     rects = ax.bar(x + offset, measurement, width, label=attribute)
        #     ax.bar_label(rects, padding=3)
        #     multiplier += 1
        #
        # # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax.set_ylabel('Balance Terms')
        # ax.set_title('Momentum Conservation')
        # ax.set_xticks(x + width, names)
        # ax.legend(loc='upper left', ncols=3)

    def compute_global_force(self):
        """
        Integrate the BFM all over the domain
        """
        global_force = np.zeros(3)  # components (r, theta, z)
        for ii in range(self.nstream):
            for jj in range(self.nspan):
                dv = self.block.area_elements[ii, jj].area * 2 * np.pi * self.r_cg[ii, jj]
                global_force[0] += self.rho[ii, jj] * (self.Fturn_r[ii, jj] + self.Floss_r[ii, jj]) * dv
                global_force[1] += self.rho[ii, jj] * (self.Fturn_t[ii, jj] + self.Floss_t[ii, jj]) * dv
                global_force[2] += self.rho[ii, jj] * (self.Fturn_z[ii, jj] + self.Floss_z[ii, jj]) * dv
        return global_force

    def compute_pressure_term(self):
        """
        Compute the pressure term all over the domain boundaries. Given that the problem is axisymmetric, the radial summation
        and the tangential summation will be zero by simmetry. Therefore, we only need to integrate the terms in the axial
        direction
        """
        pressure_term = np.zeros(3)  # components (r, theta, z)

        # hub integration
        for ii in range(self.nstream):
            p = self.p[ii, 0]
            dl_r = self.block.area_elements[ii, 0].line_elements[0].l_orth[1]
            dl_z = self.block.area_elements[ii, 0].line_elements[0].l_orth[0]
            pressure_term[2] += p * dl_z * 2 * np.pi * self.r_cg[ii, 0]

        # shroud integration
        for ii in range(self.nstream):
            p = self.p[ii, -1]
            dl_r = self.block.area_elements[ii, -1].line_elements[2].l_orth[1]
            dl_z = self.block.area_elements[ii, -1].line_elements[2].l_orth[0]
            pressure_term[2] += p * dl_z * 2 * np.pi * self.r_cg[ii, -1]

        # inlet integration
        for jj in range(self.nspan):
            p = self.p[0, jj]
            dl_r = self.block.area_elements[0, jj].line_elements[3].l_orth[1]
            dl_z = self.block.area_elements[0, jj].line_elements[3].l_orth[0]
            pressure_term[2] += p * dl_z * 2 * np.pi * self.r_cg[0, jj]

        # outlet integration
        for jj in range(self.nspan):
            p = self.p[-1, jj]
            dl_r = self.block.area_elements[-1, jj].line_elements[1].l_orth[1]
            dl_z = self.block.area_elements[-1, jj].line_elements[1].l_orth[0]
            pressure_term[2] += p * dl_z * 2 * np.pi * self.r_cg[-1, jj]

        return pressure_term

    def compute_convection_term(self):
        """
        Compute the convection term all over the domain boundaries
        """
        convection_term = np.zeros(3)  # components (r, theta, z)

        # inlet integration
        for jj in range(self.nspan):
            ur = self.ur[0, jj]
            ut = self.ut[0, jj]
            uz = self.uz[0, jj]
            UU = np.array([[ur ** 2, ur * ut, ur * uz],
                           [ur * ut, ut ** 2, ut * uz],
                           [ur * uz, ut * uz, uz ** 2]])
            dl = np.array([[self.block.area_elements[0, jj].line_elements[3].l_orth[1]],
                           [0],
                           [self.block.area_elements[0, jj].line_elements[3].l_orth[0]]])
            convection_term += 2 * np.pi * self.r_cg[0, jj] * (UU @ dl.flatten())

        # outlet integration
        for jj in range(self.nspan):
            ur = self.ur[-1, jj]
            ut = self.ut[-1, jj]
            uz = self.uz[-1, jj]
            UU = np.array([[ur ** 2, ur * ut, ur * uz],
                           [ur * ut, ut ** 2, ut * uz],
                           [ur * uz, ut * uz, uz ** 2]])
            dl = np.array([[self.block.area_elements[-1, jj].line_elements[1].l_orth[1]],
                           [0],
                           [self.block.area_elements[-1, jj].line_elements[1].l_orth[0]]])
            convection_term += 2 * np.pi * self.r_cg[-1, jj] * (UU @ dl.flatten())

        return convection_term

    def compute_body_force_residuals(self):
        """
        Use the 2d meridional equations to check the body forces obtained from the residual of the simulations data averaged
        """
        Fr = self.ur * self.dur_dr + self.uz * self.dur_dz - self.ut ** 2 / self.r_cg + 1 / self.rho * self.dp_dr
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, Fr, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.contour(self.z_cg, self.r_cg, Fr, levels=[0], colors='white', linestyles='dashed', linewidths=2)
        plt.title(r'$F_{r,res}$')

        Ft = self.ur * self.dut_dr + self.uz * self.dut_dz + self.ur * self.ut / self.r_cg
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, Ft, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.contour(self.z_cg, self.r_cg, Ft, levels=[0], colors='white', linestyles='dashed', linewidths=2)
        plt.title(r'$F_{\theta,res}$')

        Fz = self.ur * self.duz_dr + self.uz * self.duz_dz + 1 / self.rho * self.dp_dz
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, Fz, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.contour(self.z_cg, self.r_cg, Fz, levels=[0], colors='white', linestyles='dashed', linewidths=2)
        plt.title(r'$F_{z,res}$')

        Wf = Fz * self.uz + self.ut * Ft + self.ur * Fr
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_cg, self.r_cg, Wf, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.contour(self.z_cg, self.r_cg, Wf, levels=[0], colors='white', linestyles='dashed', linewidths=2)
        plt.title(r'$W_{F,res}$')

    def three_dimensional_weighted_interpolation(self, field, idx, istream, ispan):
        """
        Create a line going from one periodic boundary to the other, truncated in the middle to make space for the blade.
        Then interpolate the field value on it, and perform weighted integral to evaluate the circumferential average of it.
        """
        Nbins = 10
        theta_min = np.min(self.data.theta[idx])
        theta_max = np.max(self.data.theta[idx])
        theta_ps = self.blade.theta_ps[istream, ispan]
        theta_ss = self.blade.theta_ss[istream, ispan]

        if theta_ps > theta_ss:
            arc1 = np.linspace(theta_min, theta_ss, Nbins)
            arc2 = np.linspace(theta_ps, theta_max, Nbins)
        else:
            arc1 = np.linspace(theta_min, theta_ps, Nbins)
            arc2 = np.linspace(theta_ss, theta_max, Nbins)

        # plt.figure()
        # plt.scatter(theta_min, 0, label=r'$\theta_{min}$', marker='x')
        # plt.scatter(theta_max, 0, label=r'$\theta_{max}$', marker='x')
        # plt.scatter(theta_ps, 0, label=r'$\theta_{ps}$', marker='x')
        # plt.scatter(theta_ss, 0, label=r'$\theta_{ss}$', marker='x')
        # plt.plot(arc1, arc1*0, '-o', label='arc1')
        # plt.plot(arc2, arc2*0, '-o', label='arc2')
        # plt.legend()

        points = np.column_stack((self.data.r[idx], self.data.theta[idx], self.data.z[idx]))
        values = field[idx]

        # interp = LinearNDInterpolator(points, values)
        #
        # # Perform interpolation, something wrong here
        # interpolated_value = interp(self.r_cg[istream, ispan], arc_theta[4], self.z_cg[istream, ispan])
        # print(interpolated_value)
        arc_theta = np.concatenate((arc1, arc2))
        r = np.zeros_like(arc_theta) + self.r_cg[istream, ispan]
        z = np.zeros_like(arc_theta) + self.z_cg[istream, ispan]
        f = np.zeros_like(arc_theta)

        interp_values = griddata((self.data.r[idx], self.data.theta[idx], self.data.z[idx]), values,
                                 (r, arc_theta, z), method='nearest')
        # interp_values.reshape(np.shape(X_eval))
        # Create a 3D scatter plot
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(self.data.x[idx][::50], self.data.y[idx][::50], self.data.z[idx][::50])
        # ax.plot(r*np.cos(arc_theta), r*np.sin(arc_theta), z, 'red')
        # # Set labels
        # ax.set_xlabel('r')
        # ax.set_ylabel('theta')
        # ax.set_zlabel('z')

        # ref_radius = np.sqrt((self.z_cg[istream + 1, ispan] - self.z_cg[istream, ispan]) ** 2 + \
        #                        (self.r_cg[istream, ispan + 1] - self.r_cg[istream, ispan + 1]) ** 2)
        #
        # for i in range(len(r)):
        #     xp = r[i]*np.cos(arc_theta[i])
        #     yp = r[i] * np.sin(arc_theta[i])
        #     zp = z[i]
        #
        #     d = ref_radius
        #     idx_points = np.where(np.sqrt((self.data.x-xp)**2 + (self.data.y-yp)**2 + (self.data.z-zp)**2) < d)
        #     while (len(self.data.x[idx_points])<1):
        #         d *= 1.2
        #         idx_points = np.where(np.sqrt((self.data.x - xp) ** 2 + (self.data.y - yp) ** 2 + (self.data.z - zp) ** 2) < d)
        #
        #     f_values = field[idx_points]
        #     volume = self.data.finite_volume[idx_points]
        #     f[i] = np.sum(f_values*volume)/np.sum(volume)

        # plt.figure()
        # plt.plot(arc_theta, interp_values)

        integ = np.trapz(interp_values, arc_theta) / (theta_max - theta_min)
        return integ

    def set_gradients_to_zero(self):
        """
        Set the gradient fields to zero. Just for development purpose
        """
        self.drho_dz *= 0
        self.drho_dr *= 0
        self.dur_dz *= 0
        self.dur_dr *= 0
        self.dut_dz *= 0
        self.dut_dr *= 0
        self.duz_dz *= 0
        self.duz_dr *= 0
        self.dp_dz *= 0
        self.dp_dr *= 0

    def override_baseflow(self):
        """
        Test method, which overrides the baseflow field used in the Sun Model. Tangential velocity zero, axial and
        radial determined by continuity and streamline directions, pressure from Bernoulli
        """
        mdot = 1 # 1 kg/s of mass flow
        self.rho = np.ones_like(self.rho)
        self.ut = np.ones_like(self.ut)
        self.uz, self.ur = self.compute_streamline_directions()
        self.p = np.ones_like(self.ut)
        self.set_gradients_to_zero()

    def compute_streamline_directions(self):
        """
        Compute the cosine directors for the streamlines
        """
        cos_dirz = np.zeros_like(self.z_grid)
        cos_dirr = np.zeros_like(self.r_grid)
        for ii in range(self.nstream):
            # if ii<self.nstream-1:
            #     dz = self.z_grid[ii + 1, :] - self.z_grid[ii, :]
            #     dr = self.r_grid[ii + 1, :] - self.r_grid[ii, :]
            # else:
            #     dz = self.z_grid[ii, :] - self.z_grid[ii - 1, :]
            #     dr = self.r_grid[ii, :] - self.r_grid[ii - 1, :]
            if ii==0:
                dz = self.z_grid[ii+1, :] - self.z_grid[ii, :]
                dr = self.r_grid[ii+1, :] - self.r_grid[ii, :]
            elif ii==self.nstream-1:
                dz = self.z_grid[ii, :] - self.z_grid[ii-1, :]
                dr = self.r_grid[ii, :] - self.r_grid[ii-1, :]
            else:
                dz = self.z_grid[ii+1, :] - self.z_grid[ii - 1, :]
                dr = self.r_grid[ii+1, :] - self.r_grid[ii - 1, :]
            cos_dirz[ii, :] = dz / sqrt(dz**2+dr**2)
            cos_dirr[ii, :] = dr / sqrt(dz ** 2 + dr ** 2)
        return cos_dirz, cos_dirr




