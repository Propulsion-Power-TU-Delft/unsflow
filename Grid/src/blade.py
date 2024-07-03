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
from .functions import cartesian_to_cylindrical
from Sun.src.general_functions import print_banner_begin, print_banner_end
from Utils.styles import total_chars, total_chars_mid
from Grid.src.functions import compute_picture_size
from Grid.src.profile import Profile
from Utils.styles import *
from scipy.interpolate import griddata
import math


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

        self.read_from_curve_file(iblade, poly_degree)
        self.print_blade_info()

    def read_from_curve_file(self, iblade, poly_degree):
        """
        Reads from a specific format of file, which has been generated during blade generation (e.g. BladeGen).
        :param iblade: number of the blade row
        """
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

        self.picture_size_blank, self.picture_size_contour = compute_picture_size(self.z, self.r)

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

        # inspect points, for three blades
        N_Blades = self.config.get_blades_number()
        theta_machine = np.linspace(0, np.pi, N_Blades//2)
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

        def compute_unstructured_streamline_length(r, theta, z, le_idx, te_idx):
            """
            function to compute the stremaline length of an nustructured dataset, starting from the leading edge point
            """
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            swl = np.zeros_like(x)
            z_stream = np.linspace(z[le_idx], z[te_idx], 100)
            for istream in range(len(z_stream)-1):
                z_min = z_stream[istream]
                z_max = z_stream[istream+1]
                mask1 = z < z_max
                mask2 = z > z_min
                combined_mask = mask1 & mask2
                idx = np.where(combined_mask)

                plt.figure()
                plt.scatter(z, r*theta, c='black')
                plt.scatter(z[le_idx], r[le_idx]*theta[le_idx], c='red')
                plt.plot(np.zeros(10)+z_min, np.linspace(-2,2,10),'--r')
                plt.plot(np.zeros(10) + z_max, np.linspace(-2, 2, 10),'--r')

                warnings.warn('Beta feature. Not ready yet')


        for i in range(number_main_profiles-1):
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
            #
            # swl = compute_unstructured_streamline_length(r, theta, z, le_idx, te_idx)

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

    def find_camber_surface(self, blade_block, degree=4):
        """
        Find the camber surface via regression of the function theta = f(z, r), using only the main blade points.
        Check the degree of the polynomial if it is ok. It preventively computes the surface bounding all the blade.
        :param blade_block: the block storing the meridional mesh of the bladed domain
        :param degree: degree of the regression
        """

        self.camber_degree = degree  # mixed polynomial order
        self.camber_poly_features = PolynomialFeatures(degree=degree)  # object for regression
        X = self.camber_poly_features.fit_transform(np.column_stack((self.zc_points, self.rc_points)))  # dataset in right format
        self.camber_model = LinearRegression()  # object for linear regression (least square fit)
        self.camber_model.fit(X, self.thetac_points)  # least square fit of the regression coefficient
        self.camber_coefficients = self.camber_model.coef_  # polynomial coefficients
        self.camber_intercept = self.camber_model.intercept_  # constant term

        # evaluate the camber surface on the (r,z) points of the primary structured grid
        self.z_camber = blade_block.z_grid_points
        self.r_camber = blade_block.r_grid_points
        z_eval = self.z_camber.flatten()
        r_eval = self.r_camber.flatten()
        X_eval = self.camber_poly_features.fit_transform(np.column_stack((z_eval, r_eval)))
        camber_surface_values = np.dot(X_eval, self.camber_coefficients) + self.camber_intercept
        self.theta_camber = camber_surface_values.reshape(self.z_camber.shape)
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
            plt.savefig(folder_name + '/' + save_filename + '.pdf', bbox_inches='tight')

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
            plt.savefig(folder_name + '/' + save_filename + '.pdf', bbox_inches='tight')

    def find_ss_surface(self, blade_block, degree=4):
        """
        Find the suction surface via regression of the function theta = f(z, r), using only the main blade ss points.
        :param blade_block: the block storing the meridional mesh of the bladed domain
        :param degree: degree of the regression
        """

        self.camber_degree = degree  # mixed polynomial order
        self.camber_poly_features = PolynomialFeatures(degree=degree)  # object for regression
        X = self.camber_poly_features.fit_transform(
            np.column_stack((self.zss_points, self.rss_points)))  # dataset in right format
        self.camber_model = LinearRegression()  # object for linear regression (least square fit)
        self.camber_model.fit(X, self.thetass_points)  # least square fit of the regression coefficient
        self.camber_coefficients = self.camber_model.coef_  # polynomial coefficients
        self.camber_intercept = self.camber_model.intercept_  # constant term

        # evaluate the camber surface on the (r,z) points of the primary structured grid
        self.z_ss = blade_block.z_grid_points
        self.r_ss = blade_block.r_grid_points
        z_eval = self.z_ss.flatten()
        r_eval = self.r_ss.flatten()
        X_eval = self.camber_poly_features.fit_transform(np.column_stack((z_eval, r_eval)))
        camber_surface_values = np.dot(X_eval, self.camber_coefficients) + self.camber_intercept
        self.theta_ss = camber_surface_values.reshape(blade_block.z_grid_points.shape)
        self.x_ss = self.r_ss * np.cos(self.theta_ss)
        self.y_ss = self.r_ss * np.sin(self.theta_ss)



    def find_ps_surface(self, blade_block, degree=4):
        """
        Find the suction surface via regression of the function theta = f(z, r), using only the main blade ss points.
        :param blade_block: the block storing the meridional mesh of the bladed domain
        :param degree: degree of the regression
        """

        self.camber_degree = degree  # mixed polynomial order
        self.camber_poly_features = PolynomialFeatures(degree=degree)  # object for regression
        X = self.camber_poly_features.fit_transform(
            np.column_stack((self.zps_points, self.rps_points)))  # dataset in right format
        self.camber_model = LinearRegression()  # object for linear regression (least square fit)
        self.camber_model.fit(X, self.thetaps_points)  # least square fit of the regression coefficient
        self.camber_coefficients = self.camber_model.coef_  # polynomial coefficients
        self.camber_intercept = self.camber_model.intercept_  # constant term

        # evaluate the camber surface on the (r,z) points of the primary structured grid
        self.z_ps = blade_block.z_grid_points
        self.r_ps = blade_block.r_grid_points
        z_eval = self.z_ps.flatten()
        r_eval = self.r_ps.flatten()
        X_eval = self.camber_poly_features.fit_transform(np.column_stack((z_eval, r_eval)))
        camber_surface_values = np.dot(X_eval, self.camber_coefficients) + self.camber_intercept
        self.theta_ps = camber_surface_values.reshape(blade_block.z_grid_points.shape)
        self.x_ps = self.r_ps * np.cos(self.theta_ps)
        self.y_ps = self.r_ps * np.sin(self.theta_ps)

    def plot_camber_surface(self, save_filename=None, folder_name=None, sides=False, points=True):
        """
        plot the main blade points and the camber surface
        """
        fig = plt.figure(figsize=self.picture_size_blank)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, alpha=0.5, color='green', label='camber')
        if points:
            ax.scatter(self.x_main, self.y_main, self.z_main)
        if sides:
            ax.plot_surface(self.x_ss, self.y_ss, self.z_ss, alpha=0.5, color='red', label='ss')
            ax.plot_surface(self.x_ps, self.y_ps, self.z_ps, alpha=0.5, color='blue', label='ps')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        # fig.legend()
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
        self.thk = self.r_ss * self.theta_ss - self.r_ps * self.theta_ps
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_ss, self.r_ss, self.thk, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        plt.title(r'$t$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'blade_thickness.pdf', bbox_inches='tight')

    def compute_blade_blockage(self, Nb, save_filename=None, folder_name=None):
        self.blockage = 1 - Nb * (np.abs(self.theta_ss - self.theta_ps)) / 2 / np.pi
        #
        # #artifically fix leading and trailing edge
        # self.blockage[0, :] = np.zeros_like(self.blockage[0, :])+1
        # self.blockage[-1, :] = np.zeros_like(self.blockage[-1, :]) + 1
        plt.figure(figsize=self.picture_size_contour)
        plt.contourf(self.z_ss, self.r_ss, self.blockage, cmap=color_map, levels=N_levels)
        plt.colorbar()
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        plt.title(r'$b$')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_' + 'blockage_factor.pdf', bbox_inches='tight')

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

        fig, ax = plt.subplots(figsize=self.picture_size_blank)
        cs = ax.contourf(self.z_camber, self.r_camber, 180 / np.pi * self.gas_path_angle, N_levels, cmap=color_map)
        ax.set_title(r'$\varphi$')
        cb = fig.colorbar(cs)
        cb.set_label(r'$\varphi \quad \mathrm{[deg]}$')
        if save_filename is not None:
            fig.savefig(folder_name + '/' + save_filename + 'gas_path_angle.pdf', bbox_inches='tight')

        fig, ax = plt.subplots(figsize=self.picture_size_blank)
        cs = ax.contourf(self.z_camber, self.r_camber, 180 / np.pi * self.blade_metal_angle, N_levels, cmap=color_map)
        ax.set_title(r'$\kappa$')
        cb = fig.colorbar(cs)
        cb.set_label(r'$\kappa \quad \mathrm{[deg]}$')
        if save_filename is not None:
            fig.savefig(folder_name + '/' + save_filename + 'blade_metal_angle.pdf', bbox_inches='tight')

        fig, ax = plt.subplots(figsize=self.picture_size_blank)
        cs = ax.contourf(self.z_camber, self.r_camber, 180 / np.pi * self.blade_lean_angle, N_levels, cmap=color_map)
        ax.set_title(r'$\lambda$')
        cb = fig.colorbar(cs)
        cb.set_label(r'$\lambda \quad \mathrm{[deg]}$')
        if save_filename is not None:
            fig.savefig(folder_name + '/' + save_filename + 'blade_lean_angle.pdf', bbox_inches='tight')
