#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:41:53 2023
@author: F. Neri, TU Delft
"""
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


class Blade:
    """
    class that stores the information regarding the blade topology.
    """

    def __init__(self, config):
        """
        reads the info from the blade file .curve, which is created during blade generation, e.g. with BladeGen.
        """
        self.config = config
        self.x = []
        self.y = []
        self.z = []
        self.blade = []  # main or splitter type
        self.profile = []  # span level
        self.mark = []  # leading, trailing edge

        self.read_from_curve_file()
        self.print_blade_info()

    def read_from_curve_file(self):
        """
        Reads from a specific format of file, which has been generated during blade generation (e.g. BladeGen).
        """
        blade_type = 'MAIN'
        with open(self.config.get_blade_curve_filepath()) as f:
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

        # inspect points
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(self.x_main, self.y_main, self.z_main, c='b', marker='o')
        # ax.set_xlabel('X Axis')
        # ax.set_ylabel('Y Axis')
        # ax.set_zlabel('Z Axis')

        number_main_profiles = np.unique(self.profile).shape[0]
        if len(self.x_main) % number_main_profiles != 0:
            raise ValueError('Something is wrong with the blade profiles cordinates')
        else:
            number_points_per_profile = len(self.x_main) // number_main_profiles

        # create a list of profiles, which store information of the pressure and suction side
        profiles = []
        zss = []
        rss = []
        thetass = []
        zps = []
        rps = []
        thetaps = []
        for i in range(number_main_profiles):
            ss_idxs = slice(i * number_points_per_profile, i * number_points_per_profile + number_points_per_profile // 2)
            ps_idxs = slice(i * number_points_per_profile + number_points_per_profile // 2,
                            i * number_points_per_profile + number_points_per_profile)
            profiles.append(Profile(self.x_main[ss_idxs], self.y_main[ss_idxs], self.z_main[ss_idxs],
                                    self.x_main[ps_idxs], self.y_main[ps_idxs], self.z_main[ps_idxs]))
            # profiles[i].plot_profile()
            zss.append(profiles[i].zss)
            rss.append(profiles[i].rss)
            thetass.append(profiles[i].thetass)

            zps.append(profiles[i].zps)
            rps.append(profiles[i].rps)
            thetaps.append(profiles[i].thetaps)
        self.zss_points = np.concatenate(zss)
        self.rss_points = np.concatenate(rss)
        self.thetass_points = np.concatenate(thetass)
        self.zps_points = np.concatenate(zps)
        self.rps_points = np.concatenate(rps)
        self.thetaps_points = np.concatenate(thetaps)

        if self.splitter:
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
        X = self.camber_poly_features.fit_transform(np.column_stack((self.z_main, self.r_main)))  # dataset in right format
        self.camber_model = LinearRegression()  # object for linear regression (least square fit)
        self.camber_model.fit(X, self.theta_main)  # least square fit of the regression coefficient
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
        self.theta_ss = camber_surface_values.reshape(self.z_camber.shape)
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
        self.theta_ps = camber_surface_values.reshape(self.z_camber.shape)
        self.x_ps = self.r_ps * np.cos(self.theta_ps)
        self.y_ps = self.r_ps * np.sin(self.theta_ps)

    def plot_camber_surface(self, save_filename=None, folder_name=None):
        """
        plot the main blade points and the camber surface
        """
        fig = plt.figure(figsize=self.picture_size_blank)
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(self.x_main, self.y_main, self.z_main, s=scatter_point_size, label='main blade points')
        ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, alpha=0.5, color='green', label='camber')
        ax.plot_surface(self.x_ss, self.y_ss, self.z_ss, alpha=0.5, color='red', label='ss')
        ax.plot_surface(self.x_ps, self.y_ps, self.z_ps, alpha=0.5, color='blue', label='ps')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        # fig.legend()
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

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

        # # Plot the surface
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, cmap='viridis')
        # # Customize labels
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # vector along the streamline
        if i == ni:
            stream_v = np.array([self.x_camber[i, j] - self.x_camber[i - 1, j],
                                self.y_camber[i, j] - self.y_camber[i - 1, j],
                                self.z_camber[i, j] - self.z_camber[i - 1, j]])
        else:
            stream_v = np.array([self.x_camber[i + 1, j] - self.x_camber[i, j],
                                self.y_camber[i + 1, j] - self.y_camber[i, j],
                                self.z_camber[i + 1, j] - self.z_camber[i, j]])
        stream_v /= np.linalg.norm(stream_v)

        # vector along the spanline
        if j == nj:
            span_v = np.array([self.x_camber[i, j] - self.x_camber[i, j - 1],
                               self.y_camber[i, j] - self.y_camber[i, j - 1],
                               self.z_camber[i, j] - self.z_camber[i, j - 1]])
        else:
            span_v = np.array([self.x_camber[i, j + 1] - self.x_camber[i, j],
                              self.y_camber[i, j + 1] - self.y_camber[i, j],
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
            ax.quiver(self.x_camber[i, j], self.y_camber[i, j], self.z_camber[i, j],
                      vec_1[0], vec_1[1], vec_1[2], length=0.004)
            ax.quiver(self.x_camber[i, j], self.y_camber[i, j], self.z_camber[i, j],
                      vec_2[0], vec_2[1], vec_2[2], length=0.004)
            ax.quiver(self.x_camber[i, j], self.y_camber[i, j], self.z_camber[i, j],
                      normal[0], normal[1], normal[2], length=0.004)
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
                       self.r_main * np.sin(self.theta_main + i * 2 * np.pi / n_blades),
                       self.z_main,
                       label='blade %1.d' % (i + 1))

            if (self.splitter and render_splitter):
                ax.scatter(self.r_splitter * np.cos(self.theta_splitter + i * 2 * np.pi / n_blades),
                           self.r_splitter * np.sin(self.theta_splitter + i * 2 * np.pi / n_blades),
                           self.z_splitter)

        ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, alpha=0.3)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        fig.legend()
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def find_inlet_points(self):
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

            if self.config.get_blade_inlet_type() == 'axial':
                # leading edge point
                min_z = np.min(z)  # minimum axial cordinate
                min_r_id = np.argmin(z)  # corresponding index for the r cordinate
                min_r = r[min_r_id]
            elif self.config.get_blade_inlet_type() == 'radial':
                min_r = np.min(r)
                min_z_id = np.argmin(r)
                min_z = z[min_z_id]
            else:
                raise ValueError('Set a geometry type of the blade leading edge')

            self.inlet_z.append(min_z)
            self.inlet_r.append(min_r)
        self.inlet = np.stack((self.inlet_z, self.inlet_r), axis=1)

    def find_outlet_points(self):
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

            if self.config.get_blade_outlet_type() == 'radial':
                # trailing edge points
                max_r = np.max(r)
                max_z_id = np.argmax(r)
                max_z = z[max_z_id]
            elif self.config.get_blade_outlet_type() == 'axial':
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
                self.normal_vectors[i, j], self.streamline_vectors[i, j], self.spanline_vectors[i, j] = \
                    self.compute_camber_vector(i, j)

                self.normal_vectors_cyl[i, j] = cartesian_to_cylindrical(self.x_camber[i, j], self.y_camber[i, j],
                                                                         self.z_camber[i, j], self.normal_vectors[i, j])
                self.streamline_vectors_cyl[i, j] = cartesian_to_cylindrical(self.x_camber[i, j], self.y_camber[i, j],
                                                                             self.z_camber[i, j], self.streamline_vectors[i, j])
                self.spanline_vectors_cyl[i, j] = cartesian_to_cylindrical(self.x_camber[i, j], self.y_camber[i, j],
                                                                           self.z_camber[i, j], self.spanline_vectors[i, j])

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
                ax.quiver(self.x_camber[i, j], self.y_camber[i, j], self.z_camber[i, j],
                          self.normal_vectors[i, j][0], self.normal_vectors[i, j][1], self.normal_vectors[i, j][2],
                          length=self.scale, color='red')
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
                ax.quiver(self.x_camber[i, j], self.y_camber[i, j], self.z_camber[i, j],
                          self.streamline_vectors[i, j][0], self.streamline_vectors[i, j][1], self.streamline_vectors[i, j][2],
                          length=self.scale, color='green')
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
                ax.quiver(self.x_camber[i, j], self.y_camber[i, j], self.z_camber[i, j],
                          self.spanline_vectors[i, j][0], self.spanline_vectors[i, j][1],
                          self.spanline_vectors[i, j][2], length=self.scale, color='purple')
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
                self.gas_path_angle[i, j] = np.arctan(self.streamline_vectors_cyl[i, j][0] /
                                                      self.streamline_vectors_cyl[i, j][2])

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
