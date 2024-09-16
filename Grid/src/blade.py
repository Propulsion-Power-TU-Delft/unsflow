#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:41:53 2023
@author: F. Neri, TU Delft
"""
import warnings

import matplotlib.pyplot as plt
from numpy import array
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from .functions import cartesian_to_cylindrical, compute_2d_curvilinear_gradient
from Sun.src.general_functions import print_banner_begin, print_banner_end
from Utils.styles import total_chars, total_chars_mid
from Grid.src.functions import compute_picture_size, clip_negative_values, compute_curvilinear_abscissa, compute_3dSpline_curve, compute_2dSpline_curve, find_intersection
from Grid.src.profile import Profile
from Utils.styles import *
from scipy import interpolate
import math
import os
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import bisplrep, bisplev
from shapely.geometry import LineString
from scipy.spatial import KDTree
from Grid.src.surface import Surface
from scipy.interpolate import bisplev, bisplrep




class Blade:
    """
    class that stores the information regarding the blade topology.
    """

    def __init__(self, config, iblade=0, poly_degree=3):
        """
        reads the info from the blade file .curve, which is created during blade generation, e.g. with BladeGen.
        :param config : configuration object
        :param iblade : bladerow number
        """
        self.config = config
        self.x = []
        self.y = []
        self.z = []
        self.blade = []  # main or splitter type
        self.profile = []  # span level
        self.mark = []  # leading, trailing edge
        self.leading_edge = []
        self.trailing_edge = []
        self.camberSurf = Surface('Camber Surface')
        self.psSurf = Surface('Pressure Surface')
        self.ssSurf = Surface('Suctions Surface')

        self.read_from_curve_file(iblade, poly_degree)
        self.print_blade_info()

    def read_from_curve_file(self, iblade, poly_degree, blade_dataset='ordered', visual_debug=False):
        """
        Reads from a specific format of file, which has been generated during blade generation (e.g. BladeGen).
        :param iblade: number of the blade row
        :param poly_degree: degree of the polynomial for guessing the camber distribution
        :param blade_dataset: specify if the points in the blade cordinates file are ordered or not ordered
        """
        blade_dataset = blade_dataset.lower()
        if blade_dataset not in ['ordered', 'not ordered']:
            raise ValueError('Specify ordering type of the blade cordinates file. Ordered or not ordered')
        blade_type = 'MAIN'
        filepath = self.config.get_blade_curve_filepath()
        if isinstance(filepath, list):
            filepath = filepath[iblade]
        else:
            pass

        with open(filepath) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            words_list = line.split()
            if len(words_list) > 0:

                if words_list[0] == '##':
                    blade_type = words_list[1].upper()

                elif words_list[0] == '#':
                    profile_span = words_list[2]

                elif (len(words_list) == 3 or len(words_list) == 4):
                    self.x.append(words_list[0])
                    self.y.append(words_list[1])
                    self.z.append(words_list[2])
                    self.blade.append(blade_type)
                    self.profile.append(profile_span)

                    if len(words_list) == 3:
                        self.mark.append('')
                    else:
                        self.mark.append(words_list[-1])

        self.convert_to_floats()
        self.convert_to_arrays()

        self.x *= self.config.get_coordinates_rescaling_factor() / self.config.get_reference_length()
        self.y *= self.config.get_coordinates_rescaling_factor() / self.config.get_reference_length()
        self.z *= self.config.get_coordinates_rescaling_factor() / self.config.get_reference_length()
        self.theta = np.arctan2(self.y, self.x)
        self.r = np.sqrt(self.x ** 2 + self.y ** 2)

        # check if the blade has a splitter blade
        if np.unique(self.blade).shape[0] > 1:
            self.splitter = True
        else:
            self.splitter = False

        self.idx_main = np.where(self.blade == 'MAIN')
        self.x_main = self.x[self.idx_main]
        self.y_main = self.y[self.idx_main]
        self.z_main = self.z[self.idx_main]
        self.r_main = self.r[self.idx_main]
        self.theta_main = self.theta[self.idx_main]

        self.number_profiles = np.unique(self.profile).shape[0]
        main_profiles = np.unique(self.profile)
        # for i,prof in enumerate(main_profiles):
        #     if '%' in prof:
        #         main_profiles[i] = i
        #     else:
        #         main_profiles[i] = prof
        main_profiles = [int(prof) for prof in main_profiles]
        main_profiles.sort()

        # create a list of profiles, which store information of the pressure and suction side
        profiles = []
        zss = []
        rss = []
        thetass = []
        zps = []
        rps = []
        thetaps = []
        zcamb = []
        rcamb = []
        thetacamb = []
        tCamb = []
        kappaCamb = []

        if blade_dataset == 'not ordered':
            """
            Try to guess the right ordering of the points, when the points are not given in an ordered fashion. It doesn't
            look really good, especially for radial blades. If possible use the ordered method, providing ordered data points
            """
            for i in range(self.number_profiles):
                idx = np.where(self.profile == main_profiles[i])
                z = self.z_main[idx]
                r = self.r_main[idx]
                theta = self.theta_main[idx]
                var1 = z
                var2 = theta
                le_idx = np.where(var1 == var1.min())
                te_idx = np.where(var1 == var1.max())
                if len(le_idx[0])>1:
                    le_idx = le_idx[0][0]
                if len(te_idx[0])>1:
                    te_idx = te_idx[0][0]

                plt.figure()
                plt.plot(var1, var2, '-k.')
                plt.scatter(var1[le_idx], var2[le_idx], label='LE', s=40, c='red')
                plt.scatter(var1[te_idx], var2[te_idx], label='TE', s=40, c='red')
                var1_pol = np.linspace(var1[le_idx], var1[te_idx], 100)
                coefficients = np.polyfit(var1, var2, poly_degree)
                var2_pol = np.polyval(coefficients, var1_pol)
                plt.plot(var1_pol, var2_pol, '--b', label='camber')

                ps = []
                ss = []
                for ipoint in range(len(z)):
                    if theta[ipoint]>np.polyval(coefficients, z[ipoint]):
                        if self.config.get_shaft_rpm()>=0:
                            ps.append(ipoint)
                        else:
                            ss.append(ipoint)
                    else:
                        if self.config.get_shaft_rpm() >= 0:
                            ss.append(ipoint)
                        else:
                            ps.append(ipoint)
                plt.scatter(z[ps], theta[ps], s=100, label='PS')
                plt.scatter(z[ss], theta[ss], s=100, label='SS')
                plt.legend()

                z_camber = z[ps].copy()
                r_camber = r[ps].copy()
                theta_camber = np.polyval(coefficients, z_camber)

                # n_per_side = math.ceil(len(idx[0]) / 2)
                profiles.append(Profile(self.x_main[idx][ss],
                                        self.y_main[idx][ss],
                                        self.z_main[idx][ss],
                                        self.x_main[idx][ps],
                                        self.y_main[idx][ps],
                                        self.z_main[idx][ps]))
                profiles[i].plot_profile()
                zss.append(z[ss])
                rss.append(r[ss])
                thetass.append(theta[ss])

                zps.append(z[ps])
                rps.append(r[ps])
                thetaps.append(theta[ps])

                zcamb.append(z_camber)
                rcamb.append(r_camber)
                thetacamb.append(theta_camber)

        elif blade_dataset == 'ordered':
            """
            Make use of the ordering of the points. The first half of the points belongs to one surface, the second half to the
            second surface. Pressure or suction side designation depends on the rotational direction.
            """
            for i in range(self.number_profiles):
                idx = np.where((self.profile == str(main_profiles[i])) & (self.blade == 'MAIN'))
                z = self.z_main[idx]
                r = self.r_main[idx]
                theta = self.theta_main[idx]
                x = self.x_main[idx]
                y = self.y_main[idx]

                z1, r1, theta1 = z[0:len(z)//2], r[0:len(z)//2], theta[0:len(z)//2]
                z2, r2, theta2 = z[len(z)//2:], r[len(z)//2:], theta[len(z)//2:]
                x1, y1 = x[0:len(z)//2], y[0:len(z)//2]
                x2, y2 = x[len(z)//2:], y[len(z)//2:]

                # order from inlet to outlet
                if z1[0]<z1[-1]:
                    z2, r2, theta2 = np.flip(z2), np.flip(r2), np.flip(theta2)
                else:
                    z1, r1, theta1 = np.flip(z1), np.flip(r1), np.flip(theta1)

                # decide which one is pressure side and suction side
                dum = len(z1)//2
                if np.mean(theta1[0:dum] - theta2[0:dum]) * self.config.get_omega_shaft() > 0:
                    z_ps, r_ps, theta_ps = z1, r1, theta1
                    z_ss, r_ss, theta_ss = z2, r2, theta2
                else:
                    z_ps, r_ps, theta_ps = z2, r2, theta2
                    z_ss, r_ss, theta_ss = z1, r1, theta1

                s_ps, s_ss = self.compute_streamwise_meridional_projection_length(z_ps, r_ps, theta_ps, z_ss, r_ss, theta_ss)
                zglob, rglob, thetaglob = np.append(z_ps, z_ss), np.append(r_ps, r_ss), np.append(theta_ps, theta_ss)
                sglob = np.append(s_ps, s_ss)
                s_camber = np.linspace(np.min(sglob), np.max(sglob), self.config.get_streamwise_points()[1])
                coeff = np.polyfit(sglob, rglob*thetaglob, deg=9) # degree 9 should be fine for a radial compressor
                rtheta_camber = np.polyval(coeff, s_camber)
                coeff = np.polyfit(sglob, rglob, deg=9)  # degree 9 should be fine for a radial compressor
                r_camber = np.polyval(coeff, s_camber)
                coeff = np.polyfit(sglob, zglob, deg=9)  # degree 9 should be fine for a radial compressor
                z_camber = np.polyval(coeff, s_camber)
                theta_camber = rtheta_camber/r_camber

                if visual_debug:
                    # 3D plot of the camber line
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(x, y, z, c='b', marker='o')
                    ax.scatter(r_camber*np.cos(theta_camber), r_camber*np.sin(theta_camber), z_camber, c='r', marker='o')
                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_zlabel('Z Label')

                t_norm = self.compute_blade_thickness_normal_to_camber(s_camber, rtheta_camber, s_ps, r_ps*theta_ps,
                                                                       s_ss, r_ss*theta_ss)
                t_tang = self.compute_blade_thickness_tangential(s_camber, rtheta_camber, s_ps, r_ps * theta_ps, s_ss,
                                                                       r_ss * theta_ss)
                metal_angle = self.compute_metal_angle_along_camber(s_camber, rtheta_camber)

                if visual_debug:
                    plt.figure()
                    plt.plot(s_ps, r_ps*theta_ps, '-', color='C0', label='PSide')
                    plt.plot(s_ss, r_ss*theta_ss, '-', color='C1', label='SSide')
                    plt.plot(s_camber, rtheta_camber, '-o', color='C2', ms=2, label='Camber')
                    plt.xlabel('s')
                    plt.ylabel('r*theta')
                    plt.legend()
                    plt.gca().set_aspect('equal', adjustable='box')

                zss.append(z_ss)
                rss.append(r_ss)
                thetass.append(theta_ss)

                zps.append(z_ps)
                rps.append(r_ps)
                thetaps.append(theta_ps)

                zcamb.append(z_camber)
                rcamb.append(r_camber)
                thetacamb.append(theta_camber)

                self.camberSurf.add_curve(r_camber*np.cos(theta_camber), r_camber*np.sin(theta_camber), z_camber)
                self.psSurf.add_curve(r_ps*np.cos(theta_ps), r_ps*np.sin(theta_ps), z_ps)
                self.ssSurf.add_curve(r_ss * np.cos(theta_ss), r_ss * np.sin(theta_ss), z_ss)

                tCamb.append(t_tang)
                kappaCamb.append(metal_angle)

        self.camberSurf.loft_through_profiles(extension=0.0)
        if visual_debug: self.camberSurf.plot_surface(surfaces=True)
        self.r_cambSurface, self.theta_cambSurface, self.z_cambSurface = self.camberSurf.get_global_surface(method='cylindrical')

        self.psSurf.loft_through_profiles(extension=0.0)
        if visual_debug: self.psSurf.plot_surface(surfaces=True)
        self.r_psSurface, self.theta_psSurface, self.z_psSurface = self.psSurf.get_global_surface(method='cylindrical')

        self.ssSurf.loft_through_profiles(extension=0.0)
        if visual_debug: self.ssSurf.plot_surface(surfaces=True)
        self.r_ssSurface, self.theta_ssSurface, self.z_ssSurface = self.ssSurf.get_global_surface(method='cylindrical')

        if self.splitter:
            pass
            # raise ValueError('Splitter blade not implemented yet')
            # self.idx_splitter = np.where(self.blade == 'SPLITTER')
            # self.x_splitter = self.x[self.idx_splitter]
            # self.y_splitter = self.y[self.idx_splitter]
            # self.z_splitter = self.z[self.idx_splitter]
            # self.theta_splitter = self.theta[self.idx_splitter]
            # self.r_splitter = self.r[self.idx_splitter]

    def compute_thickness_on_camber_loft(self):
        """
        Using the blade camber surface obtained through lofting
        """
        self.thk_tang_cambSurface = self.r_cambSurface * (self.theta_ssSurface - self.theta_psSurface)
        if np.mean(self.thk_tang_cambSurface) > 0:
            for ii in range(self.thk_tang_cambSurface.shape[0]):
                for jj in range(self.thk_tang_cambSurface.shape[1]):
                    if self.thk_tang_cambSurface[ii, jj] < 0:
                        self.thk_tang_cambSurface[ii, jj] = 0
        else:
            for ii in range(self.thk_tang_cambSurface.shape[0]):
                for jj in range(self.thk_tang_cambSurface.shape[1]):
                    if self.thk_tang_cambSurface[ii, jj] < 0:
                        self.thk_tang_cambSurface[ii, jj] *= -1
                    else:
                        self.thk_tang_cambSurface[ii, jj] = 0

    def compute_thickness_along_camber(self):
        """
        compute thickness for each points on the spline along the camber
        """
        points_per_profile = len(self.zc_points) // self.number_profiles
        def get_profile(arr, ii):
            return arr[ii*points_per_profile:(ii+1)*points_per_profile]

        for iProfile in range(self.number_profiles):
            zc, rc, thetac = get_profile(self.zc_points, iProfile), get_profile(self.rc_points, iProfile), get_profile(self.thetac_points, iProfile)
            zss, rss, thetass = get_profile(self.zss_points, iProfile), get_profile(self.rss_points, iProfile), get_profile(self.thetass_points, iProfile)
            zps, rps, thetaps = get_profile(self.zps_points, iProfile), get_profile(self.rps_points, iProfile), get_profile(self.thetaps_points, iProfile)

            plt.figure()
            plt.plot(zc, rc*thetac, '-o', label='camber', mec='C0', mfc='none')
            plt.plot(zss, rss*thetass, '-^', label='pside', mec='C1', mfc='none')
            plt.plot(zps, rps*thetaps, '--s', label='sside', mec='C2', mfc='none')
            plt.legend()
            print()

            zint, rint, thetaint = np.zeros_like(zc), np.zeros_like(rc), np.zeros_like(thetac)
            dz, dy = np.zeros_like(zc), np.zeros_like(rc)
            dz[1:-1], dy[1:-1] = zc[2:]-zc[0:-2], rc[2:]*thetac[2:]-rc[0:-2]*thetac[0:-2]
            dz[0], dy[0] = zc[1] - zc[0], rc[1]*thetac[1] - rc[0]*thetac[0]
            dz[-1], dy[-1] = zc[-1]-zc[-2], rc[-2]*thetac[-2]-rc[-1]*thetac[-1]
            zdir = dz/np.sqrt(dz**2+dy**2)
            dydir = dy/np.sqrt(dz**2+dy**2)
            t = np.linspace(-zc[-1]-zc[0], zc[-1]-zc[0])
            for iPoint in range(len(dz)):
                zline = zc[iPoint] -dydir[iPoint]*t
                yline = rc[iPoint]*thetac[iPoint] + zdir[iPoint]*t
                plt.plot(zline, yline, 'k', lw=0.1)
                plt.gca().set_aspect('equal', adjustable='box')

    def point_intersection(self, curve1, curve2, tol=1e-18):
        """
        find and return the intersection between 2 curves. static method because it is bound to the class, not to an instance
        of the class. It could also avoid to specify the self, since it is not used.
        :param curve1: first curve
        :param curve2: second curve
        :param tol: tolerance threshold for the algorithm. 1e-2 seems like a good value, since at this point the cordinates
        are already non-dimensional
        """
        tree = KDTree(curve1)
        intersection_points = []

        # while loop to make sure the intersection algorithm finds a point
        while len(intersection_points) == 0:
            distances, indices = tree.query(curve2)
            intersection_points = curve1[indices[distances < tol]]
            tol *= 10
        point = np.mean(intersection_points, axis=0)
        return point

    def print_blade_info(self):
        """
        Print information of the blade object during construction.
        """
        if self.config.get_verbosity():
            print_banner_begin('BLADE')
            print(
                f"{'Rescale Factor [-]:':<{total_chars_mid}}{self.config.get_coordinates_rescaling_factor():>{total_chars_mid}.3f}")
            print(f"{'Reference Length [m]:':<{total_chars_mid}}{self.config.get_reference_length():>{total_chars_mid}.3f}")
            print(f"{'Splitter Blade:':<{total_chars_mid}}{self.splitter:>{total_chars_mid}}")
            print_banner_end()

    def convert_to_floats(self):
        """
        Convert the list of cordinates to a list of float variables
        """
        self.x = [float(a) for a in self.x]
        self.y = [float(a) for a in self.y]
        self.z = [float(a) for a in self.z]

    def convert_to_arrays(self):
        """
        Convert the data lists in numpy arrays.
        """
        self.x = array(self.x, dtype=float)
        self.y = array(self.y, dtype=float)
        self.z = array(self.z, dtype=float)
        self.blade = array(self.blade)
        self.profile = array(self.profile)
        self.mark = array(self.mark)

    def plot_blade_points(self, save_filename=None, folder_name=None):
        """
        Plot the blade points. Distinguish between main or main and splitter blade
        """
        fig = plt.figure(figsize=self.picture_size_blank)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x_main, self.y_main, self.z_main, label='main blade')
        if self.splitter:
            ax.scatter(self.x_splitter, self.y_splitter, self.z_splitter, label='splitter blade')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        fig.legend()
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def twoD_function_evaluation(self, z, r, theta, z_eval, r_eval, method, degree, smooth):
        """
        Routine valid for different surfaces. Evaluate theta as a function of the z and r.
        :param method: decide between regression and interpolation
        :param degree: degree of the regression surface
        :param smooth: smooth parameter of the rbf interpolation
        """
        if method == 'regression':
            poly_features = PolynomialFeatures(degree=degree)  # object for regression
            X = poly_features.fit_transform(np.column_stack((z, r)))
            model = LinearRegression()
            model.fit(X, theta)
            coefficients = model.coef_
            intercept = model.intercept_
            X_eval = poly_features.fit_transform(np.column_stack((z_eval.flatten(), r_eval.flatten())))
            surface_values = np.dot(X_eval, coefficients) + intercept
            theta_eval = surface_values.reshape(z_eval.shape)
        elif method == 'rbf-interpolation':
            rbf = interpolate.Rbf(z, r, theta, function='multiquadric', smooth=smooth)
            theta_eval = rbf(z_eval, r_eval)
        elif method == 'griddata':
            points = np.array((z.flatten(), r.flatten())).T
            values = theta.flatten()
            theta_eval = interpolate.griddata(points, values, (z_eval, r_eval), method='linear')
            idx, idy = np.where(np.isnan(theta_eval))
            for inan in range(len(idx)):
                theta_eval[idx[inan], idy[inan]] = interpolate.griddata(points, values,
                                                                    (z_eval[idx[inan], idy[inan]],
                                                                        r_eval[idx[inan], idy[inan]]), method='nearest')
        elif method == 'bivariate_spline':
            tck = bisplrep(z.flatten(), r.flatten(), theta.flatten(), s=0)
            theta_eval = bisplev(z_eval.flatten(), r_eval.flatten(), tck)
            theta_eval = np.reshape(theta_eval, self.z_cambSurface.shape)

        else:
            raise ValueError('Unknown method')


        return theta_eval

    def fix_the_borders(self, theta, z, r):
        """
        Extrapolate the nan values
        """
        # theta[0, :] = theta[1, :] - (z[1, :] - z[0, :]) * (theta[2, :] - theta[1, :]) / (z[2, :] - z[1, :])
        # theta[-1, :] = theta[-1, :] + (z[-1, :] - z[-2, :]) * (theta[-2, :] - theta[-3, :]) / (z[-2, :] - z[-3, :])
        theta[0,:] = theta[1,:]
        theta[-2,:] = theta[-1,:]
        theta[:,0] = theta[:,1]
        theta[:,-1] = theta[:,-2]
        print()
        return theta

    def obtain_quantities_on_meridional_grid(self, smooth=0, degree=3, method='griddata'):
        """
        Find the camber surface via interpolation of the function theta = f(z, r).
        Check the degree of the polynomial if it is ok. It preventively computes the surface bounding all the blade.
        :param blade_block: the block storing the meridional mesh of the bladed domain
        """
        # evaluate the camber surface on the (r,z) points of the primary structured grid
        self.z_camber = self.z_grid
        self.r_camber = self.r_grid
        self.theta_camber = self.twoD_function_evaluation(self.z_cambSurface.flatten(),
                                                         self.r_cambSurface.flatten(),
                                                         (self.r_cambSurface*self.theta_cambSurface).flatten(),
                                                         self.z_grid, self.r_grid,
                                                         method, degree, smooth) / self.r_grid
        self.x_camber = self.r_grid * np.cos(self.theta_camber)
        self.y_camber = self.r_grid * np.sin(self.theta_camber)

        self.blockage = self.twoD_function_evaluation(self.z_cambSurface.flatten(),
                                                     self.r_cambSurface.flatten(),
                                                     self.blockage_cambSurface.flatten(),
                                                     self.z_grid, self.r_grid, method, degree, smooth)

        self.nr = self.twoD_function_evaluation(self.z_cambSurface.flatten(),
                                               self.r_cambSurface.flatten(),
                                               self.n_camber_r.flatten(),
                                               self.z_grid, self.r_grid, method, degree, smooth)

        self.nt = self.twoD_function_evaluation(self.z_cambSurface.flatten(),
                                                   self.r_cambSurface.flatten(),
                                                   self.n_camber_t.flatten(),
                                                   self.z_grid, self.r_grid, method, degree, smooth)

        self.nz = self.twoD_function_evaluation(self.z_cambSurface.flatten(),
                                               self.r_cambSurface.flatten(),
                                               self.n_camber_z.flatten(),
                                               self.z_grid, self.r_grid, method, degree, smooth)

    def add_meridional_grid(self, zgrid, rgrid):
        """
        Add the meridional grid taken from the block object
        """
        self.z_grid, self.r_grid = zgrid, rgrid

    def compute_streamline_length(self, normalize=True):
        """
        Compute the streamline length (meridional projection) of the streamlines going from leading edge to trailing edge.
        The leading edge is the starting point.
        :param projection: if True, the length is calculated as projection on the meridional plane, not the real 3D path.
        :param normalize: if True, normalize every streamline from 0 to 1
        """
        self.streamline_length = np.zeros_like(self.z_grid)
        for ii in range(1, self.streamline_length.shape[0]):
            ds = np.sqrt((self.z_grid[ii, :] - self.z_grid[ii - 1, :]) ** 2 + (self.r_grid[ii, :] - self.r_grid[ii - 1, :]) ** 2)
            self.streamline_length[ii, :] = self.streamline_length[ii - 1, :] + ds

        if normalize:
            for jj in range(0, self.streamline_length.shape[1]):
                self.streamline_length[:, jj] /= self.streamline_length[-1, jj]

    def compute_spanline_length(self, normalize=True):
        """
        Compute the spanline length (meridional projection) of the streamlines going from leading edge to trailing edge.
        The leading edge is the starting point.
        :param projection: if True, the length is calculated as projection on the meridional plane, not the real 3D path.
        :param normalize: if True, normalize every streamline from 0 to 1
        """
        self.spanline_length = np.zeros_like(self.streamline_length)
        for jj in range(1, self.spanline_length.shape[1]):
            ds = np.sqrt((self.z_grid[:, jj] - self.z_grid[:, jj -1]) ** 2 + (self.r_grid[:, jj] - self.r_grid[:, jj -1]) ** 2)
            self.spanline_length[:, jj] = self.spanline_length[:, jj -1] + ds

        if normalize:
            for ii in range(0, self.spanline_length.shape[0]):
                self.spanline_length[ii, :] /= self.spanline_length[ii, -1]

    def plot_streamline_length_contour(self, save_filename=None, folder_name=None):
        """
        plot the streamline length contour
        """
        plt.figure()
        plt.contourf(self.z_grid, self.r_grid, self.streamline_length, levels=N_levels)
        plt.xlabel(r'$z \ \rm{[-]}$')
        plt.ylabel(r'$r \ \rm{[-]}$')
        cbar = plt.colorbar()
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title(r'$\bar{s}_{stw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_streamline_length.pdf', bbox_inches='tight')

    def plot_spanline_length_contour(self, save_filename=None, folder_name=None):
        """
        plot the spanline length contour
        """
        plt.figure()
        plt.contourf(self.z_grid, self.r_grid, self.spanline_length, levels=N_levels)
        plt.xlabel(r'$z \ \rm{[-]}$')
        plt.ylabel(r'$r \ \rm{[-]}$')
        cbar = plt.colorbar()
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title(r'$\bar{s}_{spw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_spanline_length.pdf', bbox_inches='tight')

    def find_ss_surface(self, blade_block, smooth, method, degree):
        """
        Find the suction surface via regression of the function theta = f(z, r), using only the main blade ss points.
        :param blade_block: the block storing the meridional mesh of the bladed domain
        :param degree: degree of the regression
        """
        self.z_ss = blade_block.z_grid_points
        self.r_ss = blade_block.r_grid_points
        self.theta_ss = self.compute_surface(self.zss_points, self.rss_points, self.thetass_points, self.z_ss, self.r_ss, method, degree, smooth)
        self.x_ss = self.r_ss * np.cos(self.theta_ss)
        self.y_ss = self.r_ss * np.sin(self.theta_ss)



    def find_ps_surface(self, blade_block, smooth, method, degree):
        """
        Find the suction surface via regression of the function theta = f(z, r), using only the main blade ss points.
        :param blade_block: the block storing the meridional mesh of the bladed domain
        :param degree: degree of the regression
        """
        self.z_ps = blade_block.z_grid_points
        self.r_ps = blade_block.r_grid_points
        self.theta_ps = self.compute_surface(self.zps_points, self.rps_points, self.thetaps_points, self.z_ps, self.r_ps, method, degree, smooth)
        self.x_ps = self.r_ps * np.cos(self.theta_ps)
        self.y_ps = self.r_ps * np.sin(self.theta_ps)

    def plot_camber_surface(self, save_filename=None, folder_name=None, sides=False, points=True):
        """
        plot the main blade points and the camber surface
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x_cambSurface, self.y_cambSurface, self.z_cambSurface, alpha=0.5, color='red', label='reference camber')
        ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, alpha=0.5, color='blue', label='regressed camber')
        if points:
            ax.scatter(self.x_main, self.y_main, self.z_main, c='black', s=1, label='points')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')

        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def plot_camber_meridional_grid(self, save_filename=None, folder_name=None):
        """
        plot the main camber meridional grid
        """
        plt.figure()
        plt.scatter(self.z_camber, self.r_camber)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        # plt.axis('equal')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def plot_camber_normal_contour_on_loft(self):
        """
        plot the camber normal vector contours
        """
        plt.figure()
        plt.contourf(self.z_cambSurface, self.r_cambSurface, self.n_camber_r, levels=N_levels)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title(r'$n_r$ reference')
        plt.colorbar()

        plt.figure()
        plt.contourf(self.z_cambSurface, self.r_cambSurface, self.n_camber_t, levels=N_levels)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title(r'$n_{\theta}$ reference')
        plt.colorbar()

        plt.figure()
        plt.contourf(self.z_cambSurface, self.r_cambSurface, self.n_camber_z, levels=N_levels)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title(r'$n_z$ reference')
        plt.colorbar()

    def plot_blockage_contour(self, save_filename=None, folder_name=None):
        """
        plot the blockage
        """
        plt.figure()
        plt.contourf(self.z_grid, self.r_grid, self.blockage, levels=N_levels)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title(r'$b$')
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '.pdf', bbox_inches='tight')

    def plot_camber_normal_contour(self, save_filename=None, folder_name=None):
        """
        plot the camber normal vector contours
        """
        plt.figure()
        plt.contourf(self.z_grid, self.r_grid, self.nr, levels=N_levels)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title(r'$n_r$')
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + 'normal_r.pdf', bbox_inches='tight')

        plt.figure()
        plt.contourf(self.z_grid, self.r_grid, self.nt, levels=N_levels)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title(r'$n_{\theta}$')
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + 'normal_theta.pdf', bbox_inches='tight')

        plt.figure()
        plt.contourf(self.z_grid, self.r_grid, self.nz, levels=N_levels)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title(r'$n_z$')
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + 'normal_z.pdf', bbox_inches='tight')

    def write_bfm_input_file(self, filename=None, rescale=True):

        # before writing the cordinates, rescale them to meters
        if rescale:
            self.z_camber *= self.config.get_reference_length()
            self.r_camber *= self.config.get_reference_length()

        if filename is None:
            filename = 'BFM_Input.drg'
            with open(filename, 'w') as file:
                file.write('<header>\n')
                file.write('\n')

                file.write('[version inputfile]\n')
                file.write('1.0.0\n')
                file.write('\n')

                file.write('[number of blade rows]\n')
                file.write('1\n')
                file.write('\n')

                file.write('[row blade count]\n')
                file.write('%i\n' % self.config.get_blades_number())
                file.write('\n')

                file.write('[rotation factor]\n')
                file.write('1\n')
                file.write('\n')

                file.write('[number of tangential locations]\n')
                file.write('1\n')
                file.write('\n')

                file.write('[number of data entries in chordwise direction]\n')
                file.write('%i\n' % self.z_camber.shape[0])
                file.write('\n')

                file.write('[number of data entries in spanwise direction]\n')
                file.write('%i\n' % self.z_camber.shape[1])
                file.write('\n')

                file.write('[variable names]\n')
                file.write('1:axial_coordinate 2:radial_coordinate 3:n_ax 4:n_tang 5:n_rad 6:blockage_factor 7:x_LE '
                           '8:axial_chord\n')
                file.write('\n')

                file.write('</header>\n')
                file.write('\n')

                file.write('<data>\n')
                file.write('<blade row>\n')
                file.write('<tang section>\n')

                for j in range(self.z_camber.shape[1]):
                    file.write('<radial section>\n')
                    for i in range(self.z_camber.shape[0]):
                        file.write('%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\n' % (
                        self.z_camber[i, j], self.r_camber[i, j], self.n_camber_z[i, j], self.n_camber_t[i, j],
                        self.n_camber_r[i, j], self.blockage[i, j], self.z_camber[0, j],
                        self.z_camber[-1, j] - self.z_camber[0, j]))
                    file.write('</radial section>\n')
                file.write('</tang section>\n')
                file.write('</blade section>\n')
                file.write('</data>')

    def compute_camber_vector(self, i, j, xgrid, ygrid, zgrid, check=False):
        """
        For a certain point (x,y) on the camber surface z=f(x,y), find the normal vector through vectorial product
        of the vectors connecting streamwise and spanwise points. Preserve the directions to have consistent vectors
        :param i: i index of the point on the blade grid
        :param j: j index of the point on the blade grid
        :param check: if True plots the result
        """
        ni = xgrid.shape[0] - 1  # last element index
        nj = xgrid.shape[1] - 1  # last element index

        # compute versor along the first direction
        if i == ni:
            stream_v = np.array([xgrid[i, j] - xgrid[i - 1, j],
                                 ygrid[i, j] - ygrid[i - 1, j],
                                 zgrid[i, j] - zgrid[i - 1, j]])
        elif i == 0:
            stream_v = np.array([xgrid[i + 1, j] - xgrid[i, j],
                                 ygrid[i + 1, j] - ygrid[i, j],
                                 zgrid[i + 1, j] - zgrid[i, j]])
        else:
            stream_v = np.array([xgrid[i + 1, j] - xgrid[i - 1, j],
                                 ygrid[i + 1, j] - ygrid[i - 1, j],
                                 zgrid[i + 1, j] - zgrid[i - 1, j]])
        stream_v /= np.linalg.norm(stream_v)

        # compute versor along the second direction
        if j == nj:
            span_v = np.array([xgrid[i, j] - xgrid[i, j - 1],
                               ygrid[i, j] - ygrid[i, j - 1],
                               zgrid[i, j] - zgrid[i, j - 1]])
        elif j == 0:
            span_v = np.array([xgrid[i, j + 1] - xgrid[i, j],
                               ygrid[i, j + 1] - ygrid[i, j],
                               zgrid[i, j + 1] - zgrid[i, j]])
        else:
            span_v = np.array([xgrid[i, j + 1] - xgrid[i, j - 1],
                               ygrid[i, j + 1] - ygrid[i, j - 1],
                               zgrid[i, j + 1] - zgrid[i, j - 1]])
        span_v /= np.linalg.norm(span_v)

        # the normal is the vectorial product of the two
        normal = np.cross(stream_v, span_v)
        normal /= np.linalg.norm(normal)

        if normal[2] < 0:  # if the axial component of the normal is negative, invert the direction
            normal *= -1

        if check:
            fig = plt.figure(figsize=self.picture_size_blank)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, alpha=0.3)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_zlabel(r'$z$')
            ax.quiver(self.x_camber[i, j], self.y_camber[i, j], self.z_camber[i, j], vec_1[0], vec_1[1], vec_1[2], length=0.004)
            ax.quiver(self.x_camber[i, j], self.y_camber[i, j], self.z_camber[i, j], vec_2[0], vec_2[1], vec_2[2], length=0.004)
            ax.quiver(self.x_camber[i, j], self.y_camber[i, j], self.z_camber[i, j], normal[0], normal[1], normal[2],
                      length=0.004)
        return normal, stream_v, span_v

    def render_full_annulus(self, n_blades, render_splitter=False, save_filename=None, folder_name=None):
        """
        it plots all the blades around the full annulus of the machine.
        :param n_blades: how many blades the machines has.
        :param render_splitter: if True plots also the splitter blade, if present
        :param save_filename: if specified, saves the plots with the given name
        """

        fig = plt.figure(figsize=self.picture_size_blank)
        ax = fig.add_subplot(111, projection='3d')
        for i in range(0, n_blades):
            ax.scatter(self.r_main * np.cos(self.theta_main + i * 2 * np.pi / n_blades),
                       self.r_main * np.sin(self.theta_main + i * 2 * np.pi / n_blades), self.z_main,
                       label='blade %1.d' % (i + 1))

            if (self.splitter and render_splitter):
                ax.scatter(self.r_splitter * np.cos(self.theta_splitter + i * 2 * np.pi / n_blades),
                           self.r_splitter * np.sin(self.theta_splitter + i * 2 * np.pi / n_blades), self.z_splitter)

        ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, alpha=0.3)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        fig.legend()
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def find_inlet_points(self, iblade):
        """
        Find the points defining the inlet from the coordinates of the blade points.
        """
        self.inlet_z = []
        self.inlet_r = []
        self.profile_types = np.unique(self.profile)
        self.profile_types = sorted(self.profile_types, key=lambda x: float(x.strip('%')))  # to sort the list in correct way
        # in order to have span percentages in ascending order

        for span in self.profile_types:  # for each profile
            idx = np.where(np.logical_and(self.profile == span, self.blade == 'MAIN'))
            z = self.z_main[idx]
            r = self.r_main[idx]

            blade_inlet_type = self.config.get_blade_inlet_type()
            if isinstance(blade_inlet_type, list):
                blade_inlet_type = blade_inlet_type[iblade]
            else:
                pass

            if blade_inlet_type == 'axial':
                # leading edge point
                min_z = np.min(z)  # minimum axial cordinate
                min_r_id = np.argmin(z)  # corresponding index for the r cordinate
                min_r = r[min_r_id]
            elif blade_inlet_type == 'radial':
                min_r = np.min(r)
                min_z_id = np.argmin(r)
                min_z = z[min_z_id]
            else:
                raise ValueError('Set a geometry type of the blade leading edge')

            self.inlet_z.append(min_z)
            self.inlet_r.append(min_r)
        self.inlet = np.stack((self.inlet_z, self.inlet_r), axis=1)

    def extract_inlet_points(self, iblade):
        """
        Find the points defining the inlet from the coordinates of the blade points.
        """
        inlet_z, inlet_r = [], []
        for i in range(len(self.leading_edge)):
            inlet_z.append(self.leading_edge[i][0])
            inlet_r.append(self.leading_edge[i][1])
        plt.figure()
        plt.plot(self.z_main, self.r_main, 'o')
        plt.plot(inlet_z, inlet_r, 's')
        self.inlet = np.stack((inlet_z, inlet_r), axis=1)

    def extract_outlet_points(self, iblade):
        """
        Find the points defining the inlet from the coordinates of the blade points.
        """
        outlet_z, outlet_r = [], []
        for i in range(len(self.leading_edge)):
            outlet_z.append(self.trailing_edge[i][0])
            outlet_r.append(self.trailing_edge[i][1])
        plt.figure()
        plt.plot(self.z_main, self.r_main, 'o')
        plt.plot(outlet_z, outlet_r, 's')
        self.outlet = np.stack((outlet_z, outlet_r), axis=1)

    def find_outlet_points(self, iblade):
        """
        find the points defining the inlet are taken as
        the points with minimum z cordinates for each profile of the blade.
        """
        self.outlet_z = []
        self.outlet_r = []
        self.profile_types = sorted(self.profile_types, key=lambda x: float(x.strip('%')))  # to sort the list in correct way
        # in order to have span percentages in ascending order

        for span in self.profile_types:  # for each profile
            idx = np.where(np.logical_and(self.profile == span, self.blade == 'MAIN'))
            z = self.z_main[idx]
            r = self.r_main[idx]

            blade_outlet_type = self.config.get_blade_outlet_type()
            if isinstance(blade_outlet_type, list):
                blade_outlet_type = blade_outlet_type[iblade]
            else:
                pass

            if blade_outlet_type == 'radial':
                # trailing edge points
                max_r = np.max(r)
                max_z_id = np.argmax(r)
                max_z = z[max_z_id]
            elif blade_outlet_type == 'axial':
                # trailing edge points
                max_z = np.max(z)
                max_r_id = np.argmax(z)
                max_r = r[max_r_id]
            else:
                raise ValueError('Set a geometry type of the blade leading edge')

            self.outlet_z.append(max_z)
            self.outlet_r.append(max_r)
        self.outlet = np.stack((self.outlet_z, self.outlet_r), axis=1)

    def compute_camber_vectors(self):
        """
        for every point discretized on the camber surface, compute the normal vector, the streamline vector and the
        spanline vector, all in cartesian and cylindrical reference systems.
        """
        self.x_cambSurface = self.r_cambSurface*np.cos(self.theta_cambSurface)
        self.y_cambSurface = self.r_cambSurface * np.sin(self.theta_cambSurface)

        # Create 2D NumPy array of empty arrays
        self.normal_vectors = np.empty(self.z_cambSurface.shape, dtype=object)
        self.streamline_vectors = np.empty(self.z_cambSurface.shape, dtype=object)
        self.spanline_vectors = np.empty(self.z_cambSurface.shape, dtype=object)

        # compute also the vector in cylindrical cordinates
        self.normal_vectors_cyl = np.empty(self.z_cambSurface.shape, dtype=object)
        self.streamline_vectors_cyl = np.empty(self.z_cambSurface.shape, dtype=object)
        self.spanline_vectors_cyl = np.empty(self.z_cambSurface.shape, dtype=object)

        for i in range(0, self.z_cambSurface.shape[0]):
            for j in range(0, self.z_cambSurface.shape[1]):
                self.normal_vectors[i, j], self.streamline_vectors[i, j], self.spanline_vectors[
                    i, j] = self.compute_camber_vector(i, j, self.x_cambSurface, self.y_cambSurface, self.z_cambSurface)

                self.normal_vectors_cyl[i, j] = cartesian_to_cylindrical(self.x_cambSurface[i, j],
                                                                         self.y_cambSurface[i, j],
                                                                         self.z_cambSurface[i, j],
                                                                         self.normal_vectors[i, j])
                # self.streamline_vectors_cyl[i, j] = cartesian_to_cylindrical(self.x_cambSurface[i, j],
                #                                                              self.y_cambSurface[i, j],
                #                                                              self.z_cambSurface[i, j],
                #                                                              self.streamline_vectors[i, j])
                # self.spanline_vectors_cyl[i, j] = cartesian_to_cylindrical(self.x_cambSurface[i, j],
                #                                                            self.y_cambSurface[i, j],
                #                                                            self.z_cambSurface[i, j],
                #                                                            self.spanline_vectors[i, j])

        # reorder the vectors in 2d arrays
        self.n_camber_r = np.zeros_like(self.z_cambSurface)
        self.n_camber_t = np.zeros_like(self.z_cambSurface)
        self.n_camber_z = np.zeros_like(self.z_cambSurface)
        for i in range(0, self.z_cambSurface.shape[0]):
            for j in range(0, self.z_cambSurface.shape[1]):
                self.n_camber_r[i, j] = self.normal_vectors_cyl[i, j][0]
                self.n_camber_t[i, j] = self.normal_vectors_cyl[i, j][1]
                self.n_camber_z[i, j] = self.normal_vectors_cyl[i, j][2]

        if np.mean(self.n_camber_z)<0:
            self.n_camber_z *= -1
            self.n_camber_r *= -1
            self.n_camber_t *= -1

    def show_normal_vectors(self, save_filename=None, folder_name=None):
        """
        Show all the normal vectors on the camber surface.
        :param save_filename: if specified, saves the plots with the given name
        :param folder_name: folder name of the pictures
        """
        self.scale = (np.max(self.z_camber) - np.min(self.z_camber)) / 15
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, alpha=0.5)
        for i in range(0, self.x_camber.shape[0]):
            for j in range(0, self.x_camber.shape[1]):
                ax.quiver(self.x_camber[i, j], self.y_camber[i, j], self.z_camber[i, j], self.normal_vectors[i, j][0],
                          self.normal_vectors[i, j][1], self.normal_vectors[i, j][2], length=self.scale, color='red')
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)
        surf.set_edgecolor('none')  # Remove edges
        surf.set_linewidth(0.1)  # Set linewidth
        surf.set_antialiased(True)  # Enable antialiasing for smoother edges
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        ax.set_title('normal vectors')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def show_streamline_vectors(self, save_filename=None, folder_name=None):
        """
        Show all the streamline vectors on the camber surface.
        :param save_filename: if specified, saves the plots with the given name
        :param folder_name: folder name of the pictures
        """
        fig = plt.figure(figsize=self.picture_size_blank)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, alpha=0.5)
        for i in range(0, self.x_camber.shape[0]):
            for j in range(0, self.x_camber.shape[1]):
                ax.quiver(self.x_camber[i, j], self.y_camber[i, j], self.z_camber[i, j], self.streamline_vectors[i, j][0],
                          self.streamline_vectors[i, j][1], self.streamline_vectors[i, j][2], length=self.scale, color='green')
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)
        surf.set_edgecolor('none')  # Remove edges
        surf.set_linewidth(0.1)  # Set linewidth
        surf.set_antialiased(True)  # Enable antialiasing for smoother edges
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        ax.set_box_aspect([1, 1, 1])
        ax.set_title('streamline vectors')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def show_spanline_vectors(self, save_filename=None, folder_name=None):
        """
        Show all the spanline vectors on the camber surface.
        :param save_filename: if specified, saves the plots with the given name
        :param folder_name: folder name of the pictures
        """
        fig = plt.figure(figsize=self.picture_size_blank)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, alpha=0.5)
        for i in range(0, self.x_camber.shape[0]):
            for j in range(0, self.x_camber.shape[1]):
                ax.quiver(self.x_camber[i, j], self.y_camber[i, j], self.z_camber[i, j], self.spanline_vectors[i, j][0],
                          self.spanline_vectors[i, j][1], self.spanline_vectors[i, j][2], length=self.scale, color='purple')
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)
        surf.set_edgecolor('none')  # Remove edges
        surf.set_linewidth(0.1)  # Set linewidth
        surf.set_antialiased(True)  # Enable antialiasing for smoother edges
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        ax.set_box_aspect([1, 1, 1])
        ax.set_title('spanline vectors')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '.pdf', bbox_inches='tight')

    def plot_blade_thickness(self, save_filename=None, folder_name=None):
        """
        Compute blade thickness in the tangential direction
        """
        plt.figure()
        plt.contourf(self.z_grid, self.r_grid, self.thk_tang, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        plt.title(r'$t$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'blade_thickness.pdf', bbox_inches='tight')

    def compute_blade_thickness_normal_to_camber(self, xc, yc, xps, yps, xss, yss, visual_debug=False):
        """
        Compute the blade thickness in the direction perpendicular to the local camber
        """
        if visual_debug:
            plt.figure()
            plt.plot(xc, yc, '-o', label='camber', mec='C0', mfc='none', ms=2)
            plt.plot(xps, yps, '-^', label='PSide', mec='C1', mfc='none', ms=2)
            plt.plot(xss, yss, '--s', label='SSide', mec='C2', mfc='none', ms=2)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend()

        dx, dy = np.zeros_like(xc), np.zeros_like(yc)
        dx[1:-1], dy[1:-1] = xc[2:] - xc[0:-2], yc[2:] - yc[0:-2]
        dx[0], dy[0] = xc[1] - xc[0], yc[1] - yc[0]
        dx[-1], dy[-1] = xc[-1] - xc[-2], yc[-1] - yc[-2]
        x_vers = dx / np.sqrt(dx ** 2 + dy ** 2)
        y_vers = dy / np.sqrt(dx ** 2 + dy ** 2)
        t = np.linspace(-(xc[-1] - xc[0]), (xc[-1] - xc[0]))
        thk_normal = np.zeros_like(xc)
        for iPoint in range(len(dx)):
            x_line = xc[iPoint] - y_vers[iPoint] * t
            y_line = yc[iPoint] + x_vers[iPoint] * t

            # Find intersection between the two curves
            x_int_ps, y_int_ps = find_intersection(x_line, y_line, xps, yps)
            x_int_ss, y_int_ss = find_intersection(x_line, y_line, xss, yss)

            if visual_debug:
                plt.plot(x_line, y_line, 'k', lw=0.1)
                plt.plot(x_int_ps, y_int_ps, 'ro', ms=5)
                plt.plot(x_int_ss, y_int_ss, 'ro', ms=5)

            if len(x_int_ss)>0 and len(x_int_ps)>0:
                thk_normal[iPoint] = np.sqrt((x_int_ps[0]-x_int_ss[0])**2 + (y_int_ps[0]-y_int_ss[0])**2)
            else:
                thk_normal[iPoint] = 0
        return thk_normal

    def compute_blade_thickness_tangential(self, xc, yc, xps, yps, xss, yss, visual_debug=False):
        """
        Compute the blade thickness in the direction perpendicular to the local camber
        """
        if visual_debug:
            plt.figure()
            plt.plot(xc, yc, '-o', label='camber', mec='C0', mfc='none', ms=2)
            plt.plot(xps, yps, '-^', label='PSide', mec='C1', mfc='none', ms=2)
            plt.plot(xss, yss, '--s', label='SSide', mec='C2', mfc='none', ms=2)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend()

        t = np.linspace(-(xc[-1] - xc[0]), (xc[-1] - xc[0]))
        thk = np.zeros_like(xc)
        for iPoint in range(len(xc)):
            x_line = xc[iPoint] + np.zeros_like(t)
            y_line = yc[iPoint] + t

            # Find intersection between the two curves
            x_int_ps, y_int_ps = find_intersection(x_line, y_line, xps, yps)
            x_int_ss, y_int_ss = find_intersection(x_line, y_line, xss, yss)

            if visual_debug:
                plt.plot(x_line, y_line, 'k', lw=0.1)
                plt.plot(x_int_ps, y_int_ps, 'ro', ms=5)
                plt.plot(x_int_ss, y_int_ss, 'ro', ms=5)

            try:
                if (len(x_int_ss)*len(x_int_ps)*len(y_int_ss)*len(y_int_ps)>0):
                    thk[iPoint] = np.sqrt((x_int_ps[0]-x_int_ss[0])**2 + (y_int_ps[0]-y_int_ss[0])**2)
                else:
                    thk[iPoint] = 0
            except:
                thk[iPoint] = 0
        return thk


    def compute_blade_blockage_on_camber_loft(self, Nb):
        """
        Compute blade blockage based on the thickness of the blade in tangential direction
        """
        self.blockage_cambSurface = 1 - Nb * self.thk_tang_cambSurface / (2*np.pi*self.r_cambSurface)
        plt.figure()
        plt.contourf(self.z_cambSurface, self.r_cambSurface, self.blockage_cambSurface, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        plt.title(r'$b$ reference')
        plt.gca().set_aspect('equal', adjustable='box')


    def compute_blade_blockage_gradient(self, save_filename=None, folder_name=None):
        """
        Compute the blockage gradient via finite difference on the meridional grid
        """
        self.db_dz, self.db_dr = compute_2d_curvilinear_gradient(self.z_camber, self.r_camber, self.blockage)

        plt.figure()
        plt.contourf(self.z_camber, self.r_camber, self.db_dz, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.contour(self.z_camber, self.r_camber, self.db_dz, colors='white', linestyles='dashed', linewidths=2)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        plt.title(r'$dbdz$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'dbdz.pdf', bbox_inches='tight')

        plt.figure()
        plt.contourf(self.z_camber, self.r_camber, self.db_dr, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.contour(self.z_camber, self.r_camber, self.db_dr, colors='white', linestyles='dashed', linewidths=2)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        plt.title(r'$dbdr$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'dbdr.pdf', bbox_inches='tight')

        plt.figure()
        plt.contourf(self.z_camber, self.r_camber, np.sqrt(self.db_dr**2+self.db_dz**2), cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.contour(self.z_camber, self.r_camber, np.sqrt(self.db_dr**2+self.db_dz**2), colors='white', linestyles='dashed', linewidths=2)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        plt.title(r'$| \nabla b|$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'bgrad_magnitude.pdf', bbox_inches='tight')

    def plot_blockage_and_grad_leading_to_trailing(self, jump=10, save_filename=None, folder_name=None):
        """
        plot slices of the blockage and its gradient along streamwise direction from leading to trailing edge
        :param jump: jump between streamlines from hub to shroud
        """
        stations = np.arange(0, self.blockage.shape[1], jump)
        if self.blockage.shape[1]-1 not in stations:
            stations = np.concatenate((stations, np.array([self.blockage.shape[1]-1])))

        plt.figure()
        for ispan in stations:
            plt.plot(self.streamline_length[:, ispan], self.blockage[:, ispan], '-s', ms=3, label=r'$i_{span}: \ %i/%i$' %(ispan, self.blockage.shape[1]-1))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylabel(r'$b \ \rm{[-]}$')
        plt.xlabel(r'$\bar{s}_{stw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'blockage_slices.pdf', bbox_inches='tight')

        plt.figure()
        for ispan in stations:
            plt.plot(self.streamline_length[:, ispan], self.db_dz[:, ispan], '-s', ms=3, label=r'$i_{span}: \ %i/%i$' %(ispan, self.blockage.shape[1]-1))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylabel(r'$db/dz \ \rm{[1/m]}$')
        plt.xlabel(r'$\bar{s}_{stw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'dbdz_slices.pdf', bbox_inches='tight')

        plt.figure()
        for ispan in stations:
            plt.plot(self.streamline_length[:, ispan], self.db_dr[:, ispan], '-s', ms=3, label=r'$i_{span}: \ %i/%i$' %(ispan, self.blockage.shape[1]-1))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylabel(r'$db/dr \ \rm{[1/m]}$')
        plt.xlabel(r'$\bar{s}_{stw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'dbdr_slices.pdf', bbox_inches='tight')

    def plot_blockage_and_grad_hub_to_shroud(self, jump=10, save_filename=None, folder_name=None):
        """
        plot slices of the blockage and its gradient along spanwise direction from hub to shroud
        :param jump: jump between streamlines from hub to shroud
        """
        stations = np.arange(0, self.blockage.shape[0], jump)
        if self.blockage.shape[0]-1 not in stations:
            stations = np.concatenate((stations, np.array([self.blockage.shape[0] - 1])))

        plt.figure()
        for istream in stations:
            plt.plot(self.blockage[istream, :], self.spanline_length[istream, :], '-s', ms=3, label=r'$i_{stream}: \ %i/%i$' % (istream, self.blockage.shape[0]-1))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlabel(r'$b \ \rm{[-]}$')
        plt.ylabel(r'$\bar{s}_{spw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'blockage_slices_hub_to_shroud.pdf', bbox_inches='tight')

        plt.figure()
        for istream in stations:
            plt.plot(self.db_dz[istream, :], self.spanline_length[istream, :], '-s', ms=3, label=r'$i_{stream}: \ %i/%i$' % (istream, self.blockage.shape[0]-1))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlabel(r'$db/dz \ \rm{[1/m]}$')
        plt.ylabel(r'$\bar{s}_{spw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'dbdz_hub_to_shroud.pdf', bbox_inches='tight')

        plt.figure()
        for istream in stations:
            plt.plot(self.db_dr[istream, :], self.spanline_length[istream, :], '-s', ms=3, label=r'$i_{stream}: \ %i/%i$' % (istream, self.blockage.shape[0]-1))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlabel(r'$db/dr \ \rm{[1/m]}$')
        plt.ylabel(r'$\bar{s}_{spw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'dbdr_hub_to_shroud.pdf', bbox_inches='tight')

    def plot_bladetoblade_section(self, span_idx, save_filename=None, folder_name=None):
        """
        View of the blade section in the blade to blade plane, to check the camber angles.
        :param span: percentage of the span you want to visualize the profile
        """
        span_percent = (span_idx) / (self.z_camber.shape[1] - 1) * 100

        xs = self.streamline_length_ss[:, span_idx]
        ys = self.r_ss[:, span_idx] * self.theta_ss[:, span_idx]

        xp = self.streamline_length_ps[:, span_idx]
        yp = self.r_ps[:, span_idx] * self.theta_ps[:, span_idx]

        xc = self.streamline_length[:, span_idx]
        yc = self.r_camber[:, span_idx] * self.theta_camber[:, span_idx]

        plt.figure()
        plt.plot(xs, ys, '-o', label='suction side')
        plt.plot(xp, yp, '-s', label='pressure side')
        plt.plot(xc, yc, '-^', label='camber line')
        plt.legend()
        plt.xlabel(r'$s \ \rm{[-]}$')
        plt.ylabel(r'$r \theta \ \rm{[-]}$')
        # plt.xticks([xc.min(), xc.max()])
        # plt.yticks([yc.min(), yc.max()])
        plt.xticks([])
        plt.yticks([])
        plt.grid(alpha=0.3)
        ax = plt.gca()
        # ax.set_aspect('equal', adjustable='box')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_%.1f' % span_percent + '%_span.pdf', bbox_inches='tight')

    def plot_bladetoblade_profile(self, span=50, save_filename=None, folder_name=None):
        """
        View of the blade section in the blade to blade plane, to check the camber angles.
        :param span: percentage of the span you want to visualize the profile. If all, plot all of them
        """
        n_spans = self.z_camber.shape[1]

        if span == 'all':
            for i in range(n_spans):
                self.plot_bladetoblade_section(i, save_filename, folder_name)
        elif span >= 0 and span <= 100:
            span_idx = int(n_spans * span / 100)
            self.plot_bladetoblade_section(span_idx, save_filename, folder_name)
        else:
            raise ValueError('Span value not recognized')

    def compute_blade_camber_angles(self, convention='rotation-wise'):
        """
        From the normal and streamline vectors of the camber compute:
        -gas_path_angle: gas path angle (angle in the meridional plane between streamline and axial direction)
        -blade_metal_angle: angle between the camber 3d streamline vector and its meridional projection
        -lean_angle: angle between the camber 3D spanwise direction and its meridional projection
        -blade_blockage: as defined by Kottapalli. For the moment not ready yet.
        :param convention: neutral doesn't care about the sign, but rotation-wise takes positive the angles in the
        direction of rotation
        """
        self.gas_path_angle = np.zeros_like(self.x_camber)
        self.blade_metal_angle = np.zeros_like(self.x_camber)
        self.blade_lean_angle = np.zeros_like(self.x_camber)

        for i in range(0, self.x_camber.shape[0]):
            for j in range(0, self.x_camber.shape[1]):
                self.gas_path_angle[i, j] = np.arctan(self.streamline_vectors_cyl[i, j][0] / self.streamline_vectors_cyl[i, j][2])

                meridional_sl_vec = np.array([self.streamline_vectors_cyl[i, j][0], 0, self.streamline_vectors_cyl[i, j][2]])
                meridional_sl_vec /= np.linalg.norm(meridional_sl_vec)

                meridional_sp_vec = np.array([self.spanline_vectors_cyl[i, j][0], 0, self.spanline_vectors_cyl[i, j][2]])
                meridional_sp_vec /= np.linalg.norm(meridional_sp_vec)

                if convention == 'neutral':
                    self.blade_metal_angle[i, j] = np.arccos(np.dot(self.streamline_vectors_cyl[i, j], meridional_sl_vec))
                    self.blade_lean_angle[i, j] = np.arccos(np.dot(self.spanline_vectors_cyl[i, j], meridional_sp_vec))
                elif convention == 'rotation-wise':
                    self.blade_metal_angle[i, j] = -np.arccos(np.dot(self.streamline_vectors_cyl[i, j], meridional_sl_vec))
                    self.blade_lean_angle[i, j] = -np.arccos(np.dot(self.spanline_vectors_cyl[i, j], meridional_sp_vec))
                else:
                    raise ValueError('Choose a convention for the angles')

    def show_blade_angles_contour(self, save_filename=None, folder_name=None):
        """
        Contour of the blade angles.
        :param save_filename: if specified, saves the plots with the given name
        """

        fig, ax = plt.subplots()
        cs = ax.contourf(self.z_camber, self.r_camber, 180 / np.pi * self.gas_path_angle, N_levels, cmap=color_map)
        ax.set_title(r'$\varphi$')
        cb = fig.colorbar(cs)
        cb.set_label(r'$\varphi \quad \mathrm{[deg]}$')
        if save_filename is not None:
            fig.savefig(folder_name + '/' + save_filename + 'gas_path_angle.pdf', bbox_inches='tight')

        fig, ax = plt.subplots()
        cs = ax.contourf(self.z_camber, self.r_camber, 180 / np.pi * self.blade_metal_angle, N_levels, cmap=color_map)
        ax.set_title(r'$\kappa$')
        cb = fig.colorbar(cs)
        cb.set_label(r'$\kappa \quad \mathrm{[deg]}$')
        if save_filename is not None:
            fig.savefig(folder_name + '/' + save_filename + 'blade_metal_angle.pdf', bbox_inches='tight')

        fig, ax = plt.subplots()
        cs = ax.contourf(self.z_camber, self.r_camber, 180 / np.pi * self.blade_lean_angle, N_levels, cmap=color_map)
        ax.set_title(r'$\lambda$')
        cb = fig.colorbar(cs)
        cb.set_label(r'$\lambda \quad \mathrm{[deg]}$')
        if save_filename is not None:
            fig.savefig(folder_name + '/' + save_filename + 'blade_lean_angle.pdf', bbox_inches='tight')

    def plot_inlet_outlet_metal_angle(self, save_filename=None, folder_name=None, spans=(0, 0.25, 0.5, 0.75, 1)):
        """
        Plot inlet and metal angle
        """
        plt.figure()
        plt.plot(self.spanline_length[0,:], self.blade_metal_angle[0,:]*180/np.pi, '-o', ms=3, label='leading edge')
        plt.plot(self.spanline_length[-1, :], self.blade_metal_angle[-1, :]*180/np.pi, '-s', ms=3, label='trailing edge')
        plt.xlabel('Normalized Span Hub-to-Shroud [-]')
        plt.ylabel(r'Blade Metal Angle [deg]')
        plt.grid(alpha=0.2)
        plt.legend()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + 'inlet_outlet_metal_angle.pdf', bbox_inches='tight')

        idx_spans = self.compute_blade_span_indexes(spans)

        plt.figure()
        for ispan in idx_spans:
            plt.plot(self.streamline_length[:, ispan], -self.blade_metal_angle[:, ispan] * 180 / np.pi, '-o', ms=3, label='span %.3f' %(self.spanline_length[0,ispan]))
        plt.xlabel('Meridional Length LE-to-TE [-]')
        plt.ylabel(r'Blade Metal Angle [deg]')
        plt.grid(alpha=0.2)
        plt.legend()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + 'metal_angle_spans.pdf', bbox_inches='tight')

    def plot_inlet_outlet_normal_thickness(self, save_filename=None, folder_name=None, spans=(0, 0.25, 0.5, 0.75, 1)):
        """
        Plot normal thickness
        """
        idx_spans = self.compute_blade_span_indexes(spans)
        plt.figure()
        for ispan in idx_spans:
            plt.plot(self.streamline_length[:, ispan], self.thk_normal[:, ispan], '-o', ms=3, label='span %.3f' %(self.spanline_length[0,ispan]))
        plt.xlabel('Meridional Length LE-to-TE [-]')
        plt.ylabel(r'Blade Thickness [-]')
        plt.grid(alpha=0.2)
        plt.legend()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + 'blade_thickness.pdf', bbox_inches='tight')

    def compute_blade_span_indexes(self, spans):
        """
        Given a tuple of spans (normalized from 0-hub to 1-tip), return the indexes of the meridional grid as close as possible
        to those values.
        """
        idx_spans = np.zeros(len(spans), dtype=int)
        span_len = self.spanline_length[0, :]
        for ii, span in enumerate(spans):
            idx_spans[ii] = int(min(range(len(span_len)), key=lambda kk: abs(span_len[kk] - span)))
        return idx_spans

    def compute_paraview_grid_points(self, coeff, debug_visual=False):
        """
        compute the grid points on the meridional plane that will be used in the paraview macro. The borders are treated
        in order to avoid the spline to not cross the volume of the .vtu file. This is needed to avoid nans.
        :param coeff: interpolation coefficient for the borders treatment
        """
        self.x_paraview = self.r_camber.copy()
        self.y_paraview = np.zeros_like(self.x_paraview)
        self.z_paraview = self.z_camber.copy()

        # treat the borders
        self.z_paraview[0, :] = self.z_paraview[0, :] + coeff*(self.z_paraview[1, :]-self.z_paraview[0, :])
        self.z_paraview[-1, :] = self.z_paraview[-1, :] + coeff * (self.z_paraview[-2, :] - self.z_paraview[-1, :])
        self.x_paraview[:, 0] = self.x_paraview[:, 0] + coeff * (self.x_paraview[:, 1] - self.x_paraview[:, 0])
        self.x_paraview[:, -1] = self.x_paraview[:, -1] + coeff * (self.x_paraview[:, -2] - self.x_paraview[:, -1])

        if debug_visual:
            plt.figure()
            plt.scatter(self.z_camber, self.r_camber, marker='o', edgecolors='black', facecolors='none')
            plt.scatter(self.z_paraview, self.x_paraview, marker='^', edgecolors='red', facecolors='none')

    def write_paraview_grid_file(self, filename='meridional_grid.csv', foldername='Grid'):
        """
        write the file requireed by Paraview to run the circumferential avg.
        The format of the file generated is:
        istream, ispan, x, y, z
        """
        os.makedirs(foldername, exist_ok=True)
        with open(foldername + '/' + filename, 'w') as file:
            for istream in range(0, self.x_paraview.shape[0]):
                for ispan in range(0, self.x_paraview.shape[1]):
                    file.write(
                        '%i,%i,%.6f,%.6f,%.6f\n' % (istream, ispan, self.x_paraview[istream, ispan],
                                                    self.y_paraview[istream, ispan], self.z_paraview[istream, ispan]))

    def read_paraview_processed_dataset(self, folder_path):
       """
       Read the processed dataset stored in folder_path location obtained by the Paraview Macro.
       :param folder_path: folder where the dataset is saved
       """

       def extract_grid_location(file_name):
           print('Elaborating Filename: ' + file_name)
           file_name = file_name.strip('spline_data_')
           file_name = file_name.strip('.csv')
           file_name = file_name.split('_')
           nz = int(file_name[0])
           nr = int(file_name[1])
           return nz, nr

       available_avg_types = ['raw', 'density', 'axialMomentum']
       self.avg_type = 'raw'

       if self.avg_type not in available_avg_types:
           raise ValueError('Not valid average type')
       print('Weighted average type: %s' % self.avg_type)

       data_dir = folder_path
       files = [f for f in os.listdir(data_dir) if '.csv' in f]
       files = sorted(files)
       fields = ['Density', 'Mach', 'Pressure', 'Temperature', 'Velocity_Radial', 'Velocity_Tangential', 'Velocity_2', 'Entropy']
       nz, nr = extract_grid_location(files[-1])
       field_grids = {}
       for field in fields:
           field_grids[field] = np.zeros((nz+1, nr+1))
       z_grid = np.zeros((nz+1, nr+1))
       r_grid = np.zeros((nz+1, nr+1))

       for file in files:
           df = pd.read_csv(data_dir + file)
           data_dict = df.to_dict('list')
           data_dict = {key: np.array(value) for key, value in data_dict.items()}

           x = data_dict['Points_0']
           y = data_dict['Points_1']
           z = data_dict['Points_2']
           r = np.sqrt(x ** 2 + y ** 2)
           theta = np.arctan2(y, x)
           stream_id, span_id = extract_grid_location(file)
           z_grid[stream_id, span_id] = np.sum(z) / len(z)
           r_grid[stream_id, span_id] = np.sum(r) / len(r)

           for field in fields:
               f = data_dict[field]

               if self.avg_type == 'raw':
                   field_grids[field][stream_id, span_id] = np.sum(f) / len(f)
               elif self.avg_type == 'density':
                   field_grids[field][stream_id, span_id] = np.sum(f * data_dict['Density']) / np.sum(
                       data_dict['Density'])
               elif self.avg_type == 'axialMomentum':
                   field_grids[field][stream_id, span_id] = np.sum(f * data_dict['Momentum_2']) / np.sum(
                       data_dict['Momentum_2'])

       self.meridional_fields = field_grids


    def contour_meridional_fields(self, output_folder = 'Contours'):
        """
        contour of the fields stored in meridional fields
        """
        os.makedirs(output_folder, exist_ok=True)

        for key, values in self.meridional_fields.items():
            fig, ax = plt.subplots()
            contour = ax.contourf(self.z_camber, self.r_camber, values, levels=N_levels, cmap=color_map)
            # ax.set_xticks([])
            # ax.set_yticks([])
            cbar = fig.colorbar(contour)
            plt.title(key)
            ax.set_aspect('equal', adjustable='box')
            plt.savefig(output_folder + '/%s_%sAvg.pdf' % (key, self.avg_type), bbox_inches='tight')

    def compute_additional_meridional_fields(self):
        """
        Compute additional Meridional Fields
        """
        self.meridional_fields['Velocity_Meridional'] = np.sqrt(self.meridional_fields['Velocity_2']**2+
                                                                self.meridional_fields['Velocity_Radial']**2)
        self.meridional_fields['Velocity_Tangential_Relative'] = self.meridional_fields['Velocity_Tangential'] - self.config.get_omega_shaft()*self.r_camber
        self.meridional_fields['Absolute_Flow_Angle'] = np.arctan2(self.meridional_fields['Velocity_Tangential'],
                                                                 self.meridional_fields['Velocity_2'])
        self.meridional_fields['Relative_Flow_Angle'] = np.arctan2(self.meridional_fields['Velocity_Tangential_Relative'],
                                                                 self.meridional_fields['Velocity_2'])
        self.meridional_fields['Velocity_Magnitude'] = np.sqrt(self.meridional_fields['Velocity_Radial']**2 +
                                                               self.meridional_fields['Velocity_Tangential']**2 +
                                                               self.meridional_fields['Velocity_2']**2)
        self.meridional_fields['Velocity_Magnitude_Relative'] = np.sqrt(self.meridional_fields['Velocity_Radial'] ** 2 +
                                                                        self.meridional_fields['Velocity_Tangential_Relative'] ** 2 +
                                                                        self.meridional_fields['Velocity_2'] ** 2)
        self.meridional_fields['Mach_Relative'] = self.meridional_fields['Velocity_Magnitude_Relative']/np.sqrt(
            self.config.get_fluid_gamma()*self.meridional_fields['Pressure']/self.meridional_fields['Density']
        )


    def extract_body_forces(self, f_turn_method='Thermodynamic', f_loss_method='Thermodynamic'):
        """
        From the meridional fields, extract the body forces
        :param f_turn_method: method selected to extract the turning component of the body force
        :param f_loss_method: method selected to extract the loss component of the body force
        """

        # loss component
        self.meridional_fields['Force_Loss'] = np.zeros_like(self.z_camber)
        self.meridional_fields['Force_Tangential'] = np.zeros_like(self.z_camber)

        if f_loss_method.lower()=='thermodynamic':
            for jj in range(self.z_camber.shape[1]):
                self.meridional_fields['Force_Loss'][:, jj] = ((self.meridional_fields['Temperature'][:,jj] * (
                                 self.meridional_fields['Entropy'][-1,jj]-self.meridional_fields['Entropy'][0,jj])
                                 / (self.streamline_length[-1,jj]-self.streamline_length[0,jj])) *
                                 np.cos(self.meridional_fields['Relative_Flow_Angle'][:,jj]))

        self.meridional_fields['Force_Loss'] = clip_negative_values(self.meridional_fields['Force_Loss'])
        self.meridional_fields['Force_Loss_Axial'] = -self.meridional_fields['Force_Loss']*self.meridional_fields['Velocity_2']/self.meridional_fields['Velocity_Magnitude_Relative']
        self.meridional_fields['Force_Loss_Radial'] = -self.meridional_fields['Force_Loss'] * self.meridional_fields[
            'Velocity_Radial'] / self.meridional_fields['Velocity_Magnitude_Relative']
        self.meridional_fields['Force_Loss_Tangential'] = -self.meridional_fields['Force_Loss'] * self.meridional_fields[
            'Velocity_Tangential_Relative'] / self.meridional_fields['Velocity_Magnitude_Relative']
        self.meridional_fields['Force_Loss_Normalized'] = self.meridional_fields['Force_Loss']/(self.config.get_omega_shaft()**2*self.r_camber[0,-1])

        if f_turn_method.lower()=='thermodynamic':
            self.meridional_fields['Specific_Angular_Momentum'] = self.r_camber * self.meridional_fields['Velocity_Tangential']
            for jj in range(self.z_camber.shape[1]):
                self.meridional_fields['Force_Tangential'][:, jj] = (self.meridional_fields['Specific_Angular_Momentum'][-1, jj] - self.meridional_fields['Specific_Angular_Momentum'][0, jj]) / (self.streamline_length[-1,jj]-self.streamline_length[0,jj]) * self.meridional_fields['Velocity_Meridional'][:,jj]/self.r_camber[:,jj]
        self.meridional_fields['Force_Turning_Tangential'] = self.meridional_fields['Force_Tangential']-self.meridional_fields['Force_Loss_Tangential']
        self.meridional_fields['Force_Turning'] = self.meridional_fields['Force_Turning_Tangential']/self.n_camber_t
        self.meridional_fields['Force_Turning'] = clip_negative_values(self.meridional_fields['Force_Turning'])
        self.meridional_fields['Force_Turning_Axial'] = self.meridional_fields['Force_Turning'] * self.n_camber_z
        self.meridional_fields['Force_Turning_Radial'] = self.meridional_fields['Force_Turning'] * self.n_camber_r
        self.meridional_fields['Force_Turning_Tangential'] = self.meridional_fields['Force_Turning'] * self.n_camber_t
        self.meridional_fields['Force_Turning_Normalized'] = self.meridional_fields['Force_Turning']/(self.config.get_omega_shaft()**2*self.r_camber[0,-1])
        self.meridional_fields['Total_Force_Magnitude'] = np.sqrt(self.meridional_fields['Force_Loss_Axial']**2+self.meridional_fields['Force_Turning_Axial']**2 +
                                                                  self.meridional_fields['Force_Loss_Radial']**2+self.meridional_fields['Force_Turning_Radial']**2 +
                                                                  self.meridional_fields['Force_Loss_Tangential']**2+self.meridional_fields['Force_Turning_Tangential']**2)
        self.meridional_fields['Total_Force_Radial'] = self.meridional_fields['Force_Loss_Radial'] + self.meridional_fields[
            'Force_Turning_Radial']
        self.meridional_fields['Total_Force_Tangential'] = self.meridional_fields['Force_Loss_Tangential'] + self.meridional_fields[
            'Force_Turning_Tangential']
        self.meridional_fields['Total_Force_Axial'] = self.meridional_fields['Force_Loss_Axial'] + self.meridional_fields[
            'Force_Turning_Axial']

    def plot_body_forces_leading_to_trailing(self, jump=10, save_filename=None, folder_name=None):
        """
        plot slices of the body forces and its gradient along streamwise direction from leading to trailing edge
        :param jump: jump between streamlines from hub to shroud
        """
        if folder_name is not None:
            os.makedirs(folder_name, exist_ok=True)

        stations = np.arange(0, self.z_camber.shape[1], jump)
        if self.z_camber.shape[1] - 1 not in stations:
            stations = np.concatenate((stations, np.array([self.z_camber.shape[1] - 1])))

        plt.figure()
        for ispan in stations:
            plt.plot(self.streamline_length[:, ispan]/self.streamline_length[-1, ispan], self.meridional_fields['Force_Turning_Normalized'][:, ispan], '-s', ms=3,
                     label=r'$i_{span}: \ %i/%i$' % (ispan, self.blockage.shape[1] - 1))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylabel(r'$\bar{f}_{turn} \ \rm{[-]}$')
        plt.xlabel(r'$\bar{s}_{stw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'f_turning_slices.pdf', bbox_inches='tight')


    def plot_body_forces_hub_to_shroud(self, jump=5, save_filename=None, folder_name=None):
        """
        plot slices of the loss force along spanwise direction from hub to shroud
        :param jump: jump between streamlines from hub to shroud
        """
        if folder_name is not None:
            os.makedirs(folder_name, exist_ok=True)

        stations = np.arange(0, self.z_camber.shape[0], jump)
        if self.z_camber.shape[0]-1 not in stations:
            stations = np.concatenate((stations, np.array([self.z_camber.shape[0] - 1])))

        plt.figure()
        for istream in stations:
            plt.plot(self.meridional_fields['Force_Loss_Normalized'][istream, :], self.spanline_length[istream, :]/self.spanline_length[istream, -1], '-s', ms=3, label=r'$i_{stream}: \ %i/%i$' % (istream, self.blockage.shape[0]-1))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlabel(r'$\bar{f}_{loss} \ \rm{[-]}$')
        plt.ylabel(r'$\bar{s}_{spw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'f_loss_slices.pdf', bbox_inches='tight')

    def compute_streamwise_meridional_projection_length(self, z1, r1, theta1, z2, r2, theta2, debug_visual=False):
        """
        Given the coordinates defining the two sides of the blade, compute the associated curvilinear abscissa of their projection
        on the meridional plane (z,r)
        """
        blade_type = self.config.get_blade_outlet_type()
        s1 = np.zeros_like(z1)
        s2 = np.zeros_like(z2)

        # leading edge index of the minimum-z coordinate, and bookkeping of the associated curve
        if np.min(z1)<np.min(z2):
            id_LE = np.argmin(z1)
            inlet_line = 1
        else:
            id_LE = np.argmin(z2)
            inlet_line = 2

        # trailing edge index of the last point, and bookkeping of the associated curve
        if blade_type == 'axial':
            if np.max(z1) >= np.max(z2):
                id_TE = np.argmax(z1)
                outlet_line = 1
            else:
                id_TE = np.argmax(z2)
                outlet_line = 2
        elif blade_type == 'radial':
            if np.max(r1) >= np.max(r2):
                id_TE = np.argmax(r1)
                outlet_line = 1
            else:
                id_TE = np.argmax(r2)
                outlet_line = 2
        else:
            raise ValueError('Unknown blade type')

        # generate the curve from leading edge to trailing edge, deciding automatically which data using thanks to previous bookkeping
        if inlet_line==1 and outlet_line==2:
            zmeridional, rmeridional = z1[id_LE:], r1[id_LE:]
            zmeridional = np.append(zmeridional, np.array([z2[id_TE]]))
            rmeridional = np.append(rmeridional, np.array([r2[id_TE]]))
        elif inlet_line==2 and outlet_line==1:
            zmeridional, rmeridional = z2[id_LE:], r2[id_LE:]
            zmeridional = np.append(zmeridional, np.array([z1[id_TE]]))
            rmeridional = np.append(rmeridional, np.array([r1[id_TE]]))
        elif inlet_line==1 and outlet_line==1:
            zmeridional, rmeridional = z1[id_LE:id_TE+1], r1[id_LE:id_TE+1]
        elif inlet_line==2 and outlet_line==2:
            zmeridional, rmeridional = z2[id_LE:id_TE+1], r2[id_LE:id_TE+1]
        else:
            raise ValueError('Problem')

        # spline of the projection on the (z,r) plane, and associated curvilinear abscissa length
        zs, rs = compute_2dSpline_curve(zmeridional, rmeridional, 5000)
        sref = np.zeros_like(zs)
        sref[0] = 0
        for iPoint in range(1, len(sref)):
            dz = zs[iPoint] - zs[iPoint-1]
            dr = rs[iPoint] - rs[iPoint-1]
            dl = np.sqrt(dz**2+dr**2)
            sref[iPoint] = sref[iPoint-1] + dl

        def find_projected_length(zp, rp, zl, rl, sl):
            """
            for zp,rp coordinate of the points, find the associated value of curvilinear length on the meridional spline
            """
            length = np.sqrt((zp-zl)**2+(rp-rl)**2)
            index = np.argmin(length)
            return sl[index]

        # for each side point, find the related curvilinear length projection on the meridional plane
        for ii in range(len(z1)):
            s1[ii] = find_projected_length(z1[ii], r1[ii], zs, rs, sref)
        for ii in range(len(z2)):
            s2[ii] = find_projected_length(z2[ii], r2[ii], zs, rs, sref)

        if debug_visual == True:
            plt.figure()
            plt.plot(z1, r1, 'o', mec='C0', mfc='none')
            plt.plot(z2, r2, '^', mec='C1', mfc='none')
            plt.plot(zs, rs, '--', c='C2')

        return s1, s2

    def extract_coordinates_from_camber(self, s_camber, s, z, r, theta):
        """
        Extract z,r,theta for the points on the camber, using the values that created the camber in first place
        """
        z_camber, r_camber = np.zeros_like(s_camber), np.zeros_like(s_camber)
        for i in range(len(s_camber)):
            idx = np.argmin((s_camber[i]-s)**2)
            z_camber[i], r_camber[i] = z[idx], r[idx]
        return z_camber, r_camber

    def compute_metal_angle_along_camber(self, xc, yc):
        """
        Compute metal angle considering the camber
        """
        dx, dy = np.zeros_like(xc), np.zeros_like(yc)
        dx[1:-1], dy[1:-1] = xc[2:] - xc[0:-2], yc[2:] - yc[0:-2]
        dx[0], dy[0] = xc[1] - xc[0], yc[1] - yc[0]
        dx[-1], dy[-1] = xc[-1] - xc[-2], yc[-1] - yc[-2]
        x_vers = dx / np.sqrt(dx ** 2 + dy ** 2)
        y_vers = dy / np.sqrt(dx ** 2 + dy ** 2)
        alpha = np.arctan2(y_vers, x_vers)
        return alpha



























