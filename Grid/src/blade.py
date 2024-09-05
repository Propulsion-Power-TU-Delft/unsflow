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
from Grid.src.functions import compute_picture_size, clip_negative_values, compute_curvilinear_abscissa, compute_3dSpline_curve
from Grid.src.profile import Profile
from Utils.styles import *
from scipy import interpolate
import math
import os
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import bisplrep, bisplev




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

        self.read_from_curve_file(iblade, poly_degree)
        self.print_blade_info()

    def read_from_curve_file(self, iblade, poly_degree, blade_dataset='ordered', visual_debug=False, camber_method = 'spline based'):
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
                    profile_span = words_list[-1]

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

        if visual_debug:
            # inspect points, for a 90 degree sector blades
            N_Blades = self.config.get_blades_number()
            if isinstance(N_Blades, list):
                N_Blades = N_Blades[iblade]
            theta_machine = np.linspace(0, np.pi, N_Blades//4)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for theta_position in theta_machine:
                ax.scatter(self.r_main*np.cos(self.theta_main+theta_position),
                           self.r_main*np.sin(self.theta_main+theta_position),
                           self.z_main)
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Z Axis')

        number_main_profiles = np.unique(self.profile).shape[0]
        main_profiles = np.unique(self.profile)

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

        if blade_dataset == 'not ordered':
            """
            Try to guess the right ordering of the points, when the points are not given in an ordered fashion. It doesn't
            look really good, especially for radial blades. If possible use the ordered method, providing ordered data points
            """
            for i in range(number_main_profiles):
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
            for i in range(number_main_profiles):
                idx = np.where(self.profile == main_profiles[i])
                z = self.z_main[idx]
                r = self.r_main[idx]
                theta = self.theta_main[idx]

                z1, r1, theta1 = z[0:len(z)//2], r[0:len(z)//2], theta[0:len(z)//2]
                z2, r2, theta2 = z[len(z)//2:], r[len(z)//2:], theta[len(z)//2:]
                if visual_debug:
                    plt.figure()
                    plt.plot(z1, theta1, '-o', label='1st half', mec='C0', mfc='none')
                    plt.plot(z2, theta2, '-^', label='2nd half', mec='C1', mfc='none')
                    plt.xlabel('z')
                    plt.ylabel('theta')
                    plt.legend()

                def generate_intermediate_point(z, r, theta):
                    """
                    Include a point just before the last, intermediate between the last and the one before.
                    """
                    zm, rm, thetam = np.zeros(len(z)+1), np.zeros(len(z)+1), np.zeros(len(z)+1)
                    zm[0:-1], rm[0:-1], thetam[0:-1] = z, r, theta
                    zm[-1], rm[-1], thetam[-1] = z[-1], r[-1], theta[-1]
                    zm[-2], rm[-2], thetam[-2] = 0.5*(z[-1]+z[-2]), 0.5*(r[-1]+r[-2]), 0.5*(theta[-1]+theta[-2])
                    return zm, rm, thetam

                # order from inlet to outlet
                if z1[0]<z1[-1]:
                    z2, r2, theta2 = np.flip(z2), np.flip(r2), np.flip(theta2)
                else:
                    z1, r1, theta1 = np.flip(z1), np.flip(r1), np.flip(theta1)

                if camber_method == 'brutal':
                    # compute the camber points
                    try:
                        zc = 0.5 * (z1 + z2)
                        rc = 0.5 * (r1 + r2)
                        thetac = 0.5 * (theta1 + theta2)
                    except:
                        # handle the case in which one of the two sides has less points than the other
                        if (len(z1)>len(z2)):
                            z2, r2, theta2 = generate_intermediate_point(z2, r2, theta2)
                        else:
                            z1, r1, theta1 = generate_intermediate_point(z1, r1, theta1)
                        zc = 0.5 * (z1 + z2)
                        rc = 0.5 * (r1 + r2)
                        thetac = 0.5 * (theta1 + theta2)
                        z1s, r1s, theta1s = z1, r1, theta1
                        z2s, r2s, theta2s = z2, r2, theta2
                elif camber_method == 'spline based':
                    num_points = len(z1)
                    z1s, r1s, theta1s = compute_3dSpline_curve(z1, r1, theta1, num_points)
                    z2s, r2s, theta2s = compute_3dSpline_curve(z2, r2, theta2, num_points)
                    zc = 0.5 * (z1s + z2s)
                    rc = 0.5 * (r1s + r2s)
                    thetac = 0.5 * (theta1s + theta2s)
                    #store the initial and final points for future reference
                    self.leading_edge.append([zc[0], rc[0], thetac[0]])
                    self.trailing_edge.append([zc[-1], rc[-1], thetac[-1]])

                if visual_debug:
                    plt.figure()
                    plt.plot(z1, theta1, '-o', label='1st half', mec='C0', mfc='none')
                    plt.plot(z2, theta2, '-^', label='2nd half', mec='C1', mfc='none')
                    plt.plot(zc, thetac, '--s', label='camber', mec='C2', mfc='none')
                    plt.plot(z1s, theta1s, '-o', mec='C3', mfc='none')
                    plt.plot(z2s, theta2s, '-^', mec='C4', mfc='none')
                    plt.xlabel('z')
                    plt.ylabel('theta')
                    plt.legend()
                    print()

                zss.append(z1s)
                rss.append(r1s)
                thetass.append(theta1s)

                zps.append(z2s)
                rps.append(r2s)
                thetaps.append(theta2s)

                zcamb.append(zc)
                rcamb.append(rc)
                thetacamb.append(thetac)


        self.zss_points = np.concatenate(zss)
        self.rss_points = np.concatenate(rss)
        self.thetass_points = np.concatenate(thetass)
        self.zps_points = np.concatenate(zps)
        self.rps_points = np.concatenate(rps)
        self.thetaps_points = np.concatenate(thetaps)
        self.zc_points = np.concatenate(zcamb)
        self.rc_points = np.concatenate(rcamb)
        self.thetac_points = np.concatenate(thetacamb)

        if self.splitter:
            raise ValueError('Splitter blade not implemented yet')
            self.idx_splitter = np.where(self.blade == 'SPLITTER')
            self.x_splitter = self.x[self.idx_splitter]
            self.y_splitter = self.y[self.idx_splitter]
            self.z_splitter = self.z[self.idx_splitter]
            self.theta_splitter = self.theta[self.idx_splitter]
            self.r_splitter = self.r[self.idx_splitter]

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

    def compute_surface(self, z, r, theta, z_eval, r_eval, method, degree, smooth):
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
        # elif method == 'griddata':
        #     points = np.array((z.flatten(), r.flatten())).T
        #     values = theta.flatten()
        #     theta_eval = interpolate.griddata(points, values, (z_eval, r_eval), method='cubic')
        #     theta_eval = self.fix_the_borders(theta_eval, z_eval, r_eval)
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

    def find_camber_surface(self, blade_block, smooth, degree, method):
        """
        Find the camber surface via interpolation of the function theta = f(z, r).
        Check the degree of the polynomial if it is ok. It preventively computes the surface bounding all the blade.
        :param blade_block: the block storing the meridional mesh of the bladed domain
        """
        # evaluate the camber surface on the (r,z) points of the primary structured grid
        self.z_camber = blade_block.z_grid_points
        self.r_camber = blade_block.r_grid_points
        self.theta_camber = self.compute_surface(self.zc_points, self.rc_points, self.thetac_points, self.z_camber, self.r_camber, method, degree, smooth)
        self.x_camber = self.r_camber * np.cos(self.theta_camber)
        self.y_camber = self.r_camber * np.sin(self.theta_camber)

    def update_camber_surface(self, blade_block, degree=3):
        """
        Update the camber surface via regression of the function theta = f(stream, span), using only the main blade points.
        Check the degree of the polynomial if it is ok. It preventively computes the surface bounding all the blade.
        :param degree: degree of the regression
        """
        points = np.column_stack((self.z_camber.flatten(), self.r_camber.flatten()))
        stw_values = self.streamline_length.flatten()
        spw_values = self.spanline_length.flatten()
        stw_points = griddata(points, stw_values, (self.z_main, self.r_main), method='nearest')
        spw_points = griddata(points, spw_values, (self.z_main, self.r_main), method='nearest')

        plt.figure()
        plt.scatter(self.z_main, self.r_main, c=stw_points)
        plt.colorbar()

        plt.figure()
        plt.scatter(self.z_main, self.r_main, c=spw_points)
        plt.colorbar()

        x_param = self.z_main
        y_param = self.r_main
        self.camber_degree = degree  # mixed polynomial order
        self.camber_poly_features = PolynomialFeatures(degree=degree)  # object for regression
        X = self.camber_poly_features.fit_transform(np.column_stack((x_param, y_param)))  # dataset in right format
        self.camber_model = LinearRegression()  # object for linear regression (least square fit)
        self.camber_model.fit(X, self.theta_main)  # least square fit of the regression coefficient

        plt.figure()
        plt.scatter(self.z_main, self.r_main, c=self.theta_main, s=50)
        plt.xlabel('z')
        plt.ylabel('r')
        plt.title('theta camber')
        # plt.show()

        plt.figure()
        plt.scatter(stw_points.flatten(), spw_points.flatten(), c=self.theta_main,s=50)
        plt.xlabel('streamwise position')
        plt.ylabel('spanwise position')
        plt.title('theta camber')
        # plt.show()



        self.camber_coefficients = self.camber_model.coef_  # polynomial coefficients
        self.camber_intercept = self.camber_model.intercept_  # constant term
        self.z_camber = blade_block.z_grid_points
        self.r_camber = blade_block.r_grid_points
        z_eval = self.z_camber.flatten()
        r_eval = self.r_camber.flatten()
        X_eval = self.camber_poly_features.fit_transform(np.column_stack((z_eval, r_eval)))
        camber_surface_values = np.dot(X_eval, self.camber_coefficients) + self.camber_intercept
        self.theta_camber = camber_surface_values.reshape(self.z_camber.shape)
        self.x_camber = self.r_camber * np.cos(self.theta_camber)
        self.y_camber = self.r_camber * np.sin(self.theta_camber)





    def find_camber_surface2(self, blade_block, degree=4):
        """
        Find the camber surface as the surface sitting in between the pressure and the suction side
        """

        self.z_camber = blade_block.z_grid_points
        self.r_camber = blade_block.r_grid_points
        self.theta_camber = (self.theta_ss+self.theta_ps)/2
        self.x_camber = self.r_camber * np.cos(self.theta_camber)
        self.y_camber = self.r_camber * np.sin(self.theta_camber)

    def compute_streamline_length(self, projection=True, normalize=True):
        """
        Compute the streamline length (meridional projection) of the streamlines going from leading edge to trailing edge.
        The leading edge is the starting point.
        :param projection: if True, the length is calculated as projection on the meridional plane, not the real 3D path.
        :param normalize: if True, normalize every streamline from 0 to 1
        """
        self.streamline_length = np.zeros_like(self.z_camber)
        self.streamline_length_ps = np.zeros_like(self.z_camber)
        self.streamline_length_ss = np.zeros_like(self.z_camber)
        for ii in range(1, self.z_camber.shape[0]):
            if not projection:
                ds = np.sqrt((self.x_camber[ii, :] - self.x_camber[ii - 1, :]) ** 2 + (
                            self.y_camber[ii, :] - self.y_camber[ii - 1, :]) ** 2 + (
                                         self.z_camber[ii, :] - self.z_camber[ii - 1, :]) ** 2)
            else:
                ds = np.sqrt((self.r_camber[ii, :] - self.r_camber[ii - 1, :]) ** 2 + (
                            self.z_camber[ii, :] - self.z_camber[ii - 1, :]) ** 2)
            self.streamline_length[ii, :] = self.streamline_length[ii - 1, :] + ds

            if not projection:
                ds = np.sqrt((self.x_ps[ii, :] - self.x_ps[ii - 1, :]) ** 2 + (self.y_ps[ii, :] - self.y_ps[ii - 1, :]) ** 2 + (
                            self.z_ps[ii, :] - self.z_ps[ii - 1, :]) ** 2)
            else:
                ds = np.sqrt((self.r_ps[ii, :] - self.r_ps[ii - 1, :]) ** 2 + (self.z_ps[ii, :] - self.z_ps[ii - 1, :]) ** 2)
            self.streamline_length_ps[ii, :] = self.streamline_length_ps[ii - 1, :] + ds

            if not projection:
                ds = np.sqrt((self.x_ss[ii, :] - self.x_ss[ii - 1, :]) ** 2 + (self.y_ss[ii, :] - self.y_ss[ii - 1, :]) ** 2 + (
                            self.z_ss[ii, :] - self.z_ss[ii - 1, :]) ** 2)
            else:
                ds = np.sqrt((self.r_ss[ii, :] - self.r_ss[ii - 1, :]) ** 2 + (self.z_ss[ii, :] - self.z_ss[ii - 1, :]) ** 2)
            self.streamline_length_ss[ii, :] = self.streamline_length_ss[ii - 1, :] + ds

        if normalize:
            for jj in range(0, self.z_camber.shape[1]):
                self.streamline_length[:, jj] /= self.streamline_length[-1, jj]

    def compute_spanline_length(self, normalize=True):
        """
        Compute the spanline length (meridional projection) of the streamlines going from leading edge to trailing edge.
        The leading edge is the starting point.
        :param projection: if True, the length is calculated as projection on the meridional plane, not the real 3D path.
        :param normalize: if True, normalize every streamline from 0 to 1
        """
        self.spanline_length = np.zeros_like(self.z_camber)
        for jj in range(1, self.z_camber.shape[1]):
            ds = np.sqrt((self.r_camber[:, jj] - self.r_camber[:, jj -1]) ** 2 +
                         (self.z_camber[:, jj] - self.z_camber[:, jj -1]) ** 2)
            self.spanline_length[:, jj] = self.spanline_length[:, jj -1] + ds

        if normalize:
            for ii in range(0, self.z_camber.shape[0]):
                self.spanline_length[ii, :] /= self.spanline_length[ii, -1]

    def plot_streamline_length_contour(self, save_filename=None, folder_name=None):
        """
        plot the streamline length contour
        """
        plt.figure()
        plt.contourf(self.z_camber, self.r_camber, self.streamline_length, levels=N_levels)
        plt.xlabel(r'$z \ \rm{[-]}$')
        plt.ylabel(r'$r \ \rm{[-]}$')
        bar_ticks = np.linspace(0, 1, 5)
        cbar = plt.colorbar()
        cbar.set_ticks(bar_ticks)  # Optional: set specific ticks
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
        plt.contourf(self.z_camber, self.r_camber, self.spanline_length, levels=N_levels)
        plt.xlabel(r'$z \ \rm{[-]}$')
        plt.ylabel(r'$r \ \rm{[-]}$')
        bar_ticks = np.linspace(0, 1, 5)
        cbar = plt.colorbar()
        cbar.set_ticks(bar_ticks)  # Optional: set specific ticks
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

    def plot_camber_surface(self, save_filename=None, folder_name=None, sides=True, points=True, render_plotly=False):
        """
        plot the main blade points and the camber surface
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, alpha=0.45, color='red', label='camber')
        if points:
            ax.scatter(self.x_main, self.y_main, self.z_main, c='black', s=1)
        if sides:
            ax.plot_surface(self.x_ss, self.y_ss, self.z_ss, alpha=0.25, color='blue', label='ss')
            ax.plot_surface(self.x_ps, self.y_ps, self.z_ps, alpha=0.25, color='green', label='ps')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        # fig.legend()

        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

        if render_plotly:
            fig = go.Figure(data=[go.Surface(z=self.z_camber, x=self.x_camber, y=self.y_camber)])
            fig.update_layout()
            fig.show()

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

    def plot_camber_normal_contour(self, save_filename=None, folder_name=None):
        """
        plot the camber normal vector contours
        """
        plt.figure()
        plt.contourf(self.z_camber, self.r_camber, self.n_camber_r, levels=N_levels)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title(r'$n_r$')
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_r.pdf', bbox_inches='tight')

        plt.figure()
        plt.contourf(self.z_camber, self.r_camber, self.n_camber_t, levels=N_levels)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title(r'$n_{\theta}$')
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_theta.pdf', bbox_inches='tight')

        plt.figure()
        plt.contourf(self.z_camber, self.r_camber, self.n_camber_z, levels=N_levels)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title(r'$n_z$')
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_z.pdf', bbox_inches='tight')

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

    def compute_camber_vector(self, i, j, check=False):
        """
        For a certain point (x,y) on the camber surface z=f(x,y), find the normal vector through vectorial product
        of the vectors connecting streamwise and spanwise points. Preserve the directions to have consistent vectors
        :param i: i index of the point on the blade grid
        :param j: j index of the point on the blade grid
        :param check: if True plots the result
        """
        ni = self.z_camber.shape[0] - 1  # last element index
        nj = self.r_camber.shape[1] - 1  # last element index

        # vector along the streamline
        if i == ni:
            stream_v = np.array([self.x_camber[i, j] - self.x_camber[i - 1, j], self.y_camber[i, j] - self.y_camber[i - 1, j],
                                 self.z_camber[i, j] - self.z_camber[i - 1, j]])
        else:
            stream_v = np.array([self.x_camber[i + 1, j] - self.x_camber[i, j], self.y_camber[i + 1, j] - self.y_camber[i, j],
                                 self.z_camber[i + 1, j] - self.z_camber[i, j]])
        stream_v /= np.linalg.norm(stream_v)

        # vector along the spanline
        if j == nj:
            span_v = np.array([self.x_camber[i, j] - self.x_camber[i, j - 1], self.y_camber[i, j] - self.y_camber[i, j - 1],
                               self.z_camber[i, j] - self.z_camber[i, j - 1]])
        else:
            span_v = np.array([self.x_camber[i, j + 1] - self.x_camber[i, j], self.y_camber[i, j + 1] - self.y_camber[i, j],
                               self.z_camber[i, j + 1] - self.z_camber[i, j]])
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

    def compute_camber_vectors(self, fix=None):
        """
        for every point discretized on the camber surface, compute the normal vector, the streamline vector and the
        spanline vector, all in cartesian and cylindrical reference systems.
        :param fix: parameter needed to artificially fix the sign of the normal vector
        """

        # Create 2D NumPy array of empty arrays
        self.normal_vectors = np.empty(self.z_camber.shape, dtype=object)
        self.streamline_vectors = np.empty(self.z_camber.shape, dtype=object)
        self.spanline_vectors = np.empty(self.z_camber.shape, dtype=object)

        # compute also the vector in cylindrical cordinates
        self.normal_vectors_cyl = np.empty(self.z_camber.shape, dtype=object)
        self.streamline_vectors_cyl = np.empty(self.z_camber.shape, dtype=object)
        self.spanline_vectors_cyl = np.empty(self.z_camber.shape, dtype=object)

        for i in range(0, self.z_camber.shape[0]):
            for j in range(0, self.z_camber.shape[1]):
                self.normal_vectors[i, j], self.streamline_vectors[i, j], self.spanline_vectors[
                    i, j] = self.compute_camber_vector(i, j)

                self.normal_vectors_cyl[i, j] = cartesian_to_cylindrical(self.x_camber[i, j], self.y_camber[i, j],
                                                                         self.z_camber[i, j], self.normal_vectors[i, j])
                self.streamline_vectors_cyl[i, j] = cartesian_to_cylindrical(self.x_camber[i, j], self.y_camber[i, j],
                                                                             self.z_camber[i, j], self.streamline_vectors[i, j])
                self.spanline_vectors_cyl[i, j] = cartesian_to_cylindrical(self.x_camber[i, j], self.y_camber[i, j],
                                                                           self.z_camber[i, j], self.spanline_vectors[i, j])

        # reorder the vectors in 2d arrays
        self.n_camber_r = np.zeros_like(self.z_camber)
        self.n_camber_t = np.zeros_like(self.z_camber)
        self.n_camber_z = np.zeros_like(self.z_camber)
        for i in range(0, self.z_camber.shape[0]):
            for j in range(0, self.z_camber.shape[1]):
                self.n_camber_r[i, j] = self.normal_vectors_cyl[i, j][0]
                self.n_camber_t[i, j] = self.normal_vectors_cyl[i, j][1]
                self.n_camber_z[i, j] = self.normal_vectors_cyl[i, j][2]

        if fix == 'plus':
            warnings.warn('Attention, camber normal vector artificially corrected to positive on all the domain')
            self.n_camber_r = np.abs(self.n_camber_r)
            self.n_camber_t = np.abs(self.n_camber_t)
            self.n_camber_z = np.abs(self.n_camber_z)
        elif fix == 'minus':
            warnings.warn('Attention, camber normal vector artificially corrected to negative on all the domain')
            self.n_camber_r = -np.abs(self.n_camber_r)
            self.n_camber_t = -np.abs(self.n_camber_t)
            self.n_camber_z = -np.abs(self.n_camber_z)
        else:
            pass

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

    def compute_blade_thickness(self, save_filename=None, folder_name=None):
        """
        Compute blade thickness in the tangential direction
        """
        self.thk = self.r_ss * self.theta_ss - self.r_ps * self.theta_ps
        plt.figure()
        plt.contourf(self.z_ss, self.r_ss, self.thk, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        plt.title(r'$t$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'blade_thickness.pdf', bbox_inches='tight')

    def compute_blade_thickness_normal_to_camber(self, save_filename=None, folder_name=None):
        """
        Compute the blade thickness in the direction perpendicular to the local camber
        """
        self.thk_normal = self.thk * np.cos(self.blade_metal_angle)
        plt.figure()
        plt.contourf(self.z_camber, self.r_camber, self.thk_normal, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        plt.title(r'$t$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'blade_thickness_normal.pdf', bbox_inches='tight')

    def compute_blade_blockage(self, Nb, save_filename=None, folder_name=None):
        self.blockage = 1 - Nb * (np.abs(self.theta_ss - self.theta_ps)) / 2 / np.pi
        #
        # #artifically fix leading and trailing edge
        # self.blockage[0, :] = np.zeros_like(self.blockage[0, :])+1
        # self.blockage[-1, :] = np.zeros_like(self.blockage[-1, :]) + 1
        plt.figure()
        plt.contourf(self.z_ss, self.r_ss, self.blockage, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        plt.title(r'$b$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'blockage_factor.pdf', bbox_inches='tight')

    def compute_blade_blockage_gradient(self, save_filename=None, folder_name=None):
        self.db_dz, self.db_dr = compute_2d_curvilinear_gradient(self.z_camber, self.r_camber, self.blockage)

        # levels = np.linspace(-8, 8, N_levels)
        plt.figure()
        plt.contourf(self.z_camber, self.r_camber, self.db_dz, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.contour(self.z_camber, self.r_camber, self.db_dz, levels=[0], colors='white', linestyles='dashed', linewidths=2)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        plt.title(r'$dbdz$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'dbdz.pdf', bbox_inches='tight')

        # levels = np.linspace(0, 0.4, N_levels)
        plt.figure()
        plt.contourf(self.z_camber, self.r_camber, self.db_dr, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.contour(self.z_camber, self.r_camber, self.db_dr, levels=[0], colors='white', linestyles='dashed', linewidths=2)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        plt.title(r'$dbdr$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'dbdr.pdf', bbox_inches='tight')

        plt.figure()
        plt.contourf(self.z_camber, self.r_camber, np.sqrt(self.db_dr**2+self.db_dz**2), cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.contour(self.z_camber, self.r_camber, np.sqrt(self.db_dr**2+self.db_dz**2), levels=[0], colors='white', linestyles='dashed', linewidths=2)
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




























