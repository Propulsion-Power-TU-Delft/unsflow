#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:41:53 2023
@author: F. Neri, TU Delft
"""
import warnings

import matplotlib.pyplot as plt
from numpy import array, sin, cos, tan, pi
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from .functions import cartesian_to_cylindrical, compute_gradient_least_square
from Sun.src.general_functions import print_banner_begin, print_banner_end
from Utils.styles import total_chars, total_chars_mid
from Grid.src.functions import clip_negative_values, compute_curvilinear_abscissa, compute_3dSpline_curve, compute_2dSpline_curve, find_intersection, eriksson_stretching_function_both, rotate_cartesian_to_cylindric_tensor, compute_gradient_least_square, griddata_interpolation_with_nearest_filler
from Grid.src.profile import Profile
from Utils.styles import *
from scipy import interpolate
import math
import os
import pandas as pd
import pickle
import plotly.graph_objects as go
from scipy.interpolate import bisplrep, bisplev
from shapely.geometry import LineString
from scipy.spatial import KDTree
from Grid.src.surface import Surface
from scipy.interpolate import bisplev, bisplrep, griddata
from scipy.interpolate import splprep, splev





class Blade:
    """
    class that stores the information regarding the blade topology.
    """

    def __init__(self, config, iblock, iblade, poly_degree=3):
        """
        Class used to model the blade from the file .curve, which is created during blade generation. The usual format for that file is the one of Ansys.

        Parameters
        -----------------------------------

        `config` : configuration object
        
        `iblock` : grid block counter
        
        `iblade`: blade counter

        `poly_degree`: degree of the polynomial fitting the blade
        """
        print_banner_begin('BLADE %02i' %(iblade))
        self.config = config
        self.x = []
        self.y = []
        self.z = []
        self.blade = []  # main or splitter type
        self.profile = []  # span level
        self.mark = []  # leading, trailing edge
        self.leading_edge = []
        self.trailing_edge = []
        self.pressureSurface = Surface('Pressure Surface', config)
        self.suctionSurface = Surface('Suctions Surface', config)
        self.iblock = iblock
        self.iblade = iblade

        self.read_from_curve_file(iblade, iblock, poly_degree)
        print(f"{'Rescale Factor [-]:':<{total_chars_mid}}{self.config.get_coordinates_rescaling_factor():>{total_chars_mid}.3f}")
        print(f"{'Reference Length [m]:':<{total_chars_mid}}{self.config.get_reference_length():>{total_chars_mid}.3f}")
        print(f"{'Splitter Blade:':<{total_chars_mid}}{self.splitter:>{total_chars_mid}}")
        print(f"{'Blade inlet type:':<{total_chars_mid}}{self.config.get_blade_inlet_type()[iblade]:>{total_chars_mid}}")
        print(f"{'Blade outlet type:':<{total_chars_mid}}{self.config.get_blade_outlet_type()[iblade]:>{total_chars_mid}}")
        print(f"{'Method used for blade camber reconstruction:':<{total_chars_mid}}{self.config.get_blades_camber_reconstruction()[self.iblade]:>{total_chars_mid}}")
        print_banner_end()


    def read_from_curve_file(self, iblade, iblock, poly_degree, camber_stream_points=100):
        """
        Reads from a specific format of file, which has been generated during blade generation (e.g. BladeGen).
        
        Parameters
        -----------------------------------

        `config` : configuration object
        
        `iblock` : grid block counter
        
        `iblade`: blade counter

        `poly_degree`: degree of the polynomial fitting the blade

        `blade_dataset`: ordered if the points are on one side, and then on the other side. As it should be. Not order is deprecated, it could work badly.

        `camber_stream_points`: points in the streamwise direction to obtain the curvilinear absicssa for each profile
        """
        blade_type = 'MAIN'
        filepath = self.config.get_blade_curve_filepath()[iblade]
        print(f"{'Blade coordinate file:':<{total_chars_mid}}{filepath:>{total_chars_mid}}")

        with open(filepath) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            words_list = line.split()
            if len(words_list) > 0:

                if words_list[0] == '##':                               # this rows defines the blade type (main or splitter)
                    blade_type = words_list[1].upper()
                elif words_list[0] == '#':                              # this rows define the span of the profile described below
                    profile_span = words_list[2]
                elif (len(words_list) == 3 or len(words_list) == 4):    # this rows has the coordinate of the blade points
                    self.x.append(float(words_list[0]))
                    self.y.append(float(words_list[1]))
                    if self.config.invert_axial_coordinates():
                        self.z.append(-float(words_list[2]))
                    else:
                        self.z.append(float(words_list[2]))
                    self.blade.append(blade_type)
                    self.profile.append(profile_span)

                    if len(words_list) == 3:
                        self.mark.append('')                            # it is a normal point on the surface
                    else:
                        self.mark.append(words_list[-1])                # it is defined as leading or trailing edge point (optional)
                else:
                    pass                                                # ignore different type of lines

        # convert in numpy arrays
        self.x = array(self.x, dtype=float)
        self.y = array(self.y, dtype=float)
        self.z = array(self.z, dtype=float)
        self.blade = array(self.blade)
        self.profile = array(self.profile)
        self.mark = array(self.mark)

        # rescale the coordinates to SI units
        print(f"{'Coordinates rescaled to SI units by factor: ':<{total_chars_mid}}{self.config.get_coordinates_rescaling_factor():>{total_chars_mid}.3f}")
        self.x *= self.config.get_coordinates_rescaling_factor()
        self.y *= self.config.get_coordinates_rescaling_factor()
        self.z *= self.config.get_coordinates_rescaling_factor()

        if self.config.get_normalize_coordinates():
            self.x /= self.config.get_reference_length()
            self.y /= self.config.get_reference_length()
            self.z /= self.config.get_reference_length()

        self.theta = np.arctan2(self.y, self.x)
        self.r = np.sqrt(self.x ** 2 + self.y ** 2)

        # check if the blade has a splitter blade
        if np.unique(self.blade).shape[0] > 1:
            self.splitter = True
        else:
            self.splitter = False
        print(f"{'Blade coordinate file contains splitter blade:':<{total_chars_mid}}{self.splitter:>{total_chars_mid}}")

        self.idx_main = np.where(self.blade == 'MAIN')
        self.x_main = self.x[self.idx_main]
        self.y_main = self.y[self.idx_main]
        self.z_main = self.z[self.idx_main]
        self.r_main = self.r[self.idx_main]
        self.theta_main = self.theta[self.idx_main]

        self.number_profiles = np.unique(self.profile).shape[0]
        print(f"{'Number of profiles:':<{total_chars_mid}}{self.number_profiles:>{total_chars_mid}}")
        main_profiles = np.unique(self.profile)
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
        
        self.thickness = {}
        self.rc_data, self.thetac_data, self.zc_data, self.thk_data = [], [], [], []
        self.rss_data, self.thetass_data, self.zss_data = [], [], []
        self.rps_data, self.thetaps_data, self.zps_data = [], [], []
        for i in range(self.number_profiles):
            idx = np.where((self.profile == str(main_profiles[i])) & (self.blade == 'MAIN'))
            z = self.z_main[idx]
            r = self.r_main[idx]
            theta = self.theta_main[idx]
            x = self.x_main[idx]
            y = self.y_main[idx]

            # spline of the profile in 3D
            tck, u = splprep([x, y, z], k=self.config.get_blade_profiles_spline_order()[self.iblade], s=0, per=0)
            u_fine = np.linspace(0, 1, 1000)
            spline_points = splev(u_fine, tck)
            if self.config.get_visual_debug():
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, c='b', marker='o')
                ax.plot(*spline_points, c='r')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.set_aspect('equal')            

            # obtain the coordinates of the spline in the blade to blade view
            r1,t1,m1,z1, r2,t2,m2,z2, rc,tc,mc,zc = self.compute_meridional_coordinate(spline_points)
            
            # distinguish the two sides between pressure and suction
            dum = len(z1)//2
            if np.mean(t1[0:dum] - t2[0:dum]) * self.config.get_omega_shaft()[iblock] > 0:
                z_ps, r_ps, theta_ps, m_ps = z1,r1,t1,m1
                z_ss, r_ss, theta_ss, m_ss = z2,r2,t2,m2
            else:
                z_ps, r_ps, theta_ps, m_ps = z2,r2,t2,m2
                z_ss, r_ss, theta_ss, m_ss = z1,r1,t1,m1
            
            # add surface data to dataset
            self.rss_data.append(r_ss)
            self.zss_data.append(z_ss)
            self.thetass_data.append(theta_ss)
            self.rps_data.append(r_ps)
            self.zps_data.append(z_ps)
            self.thetaps_data.append(theta_ps)

            t_norm = self.compute_blade_thickness_normal_to_camber(mc, rc*tc, 
                                                                    m_ps, r_ps * theta_ps, 
                                                                    m_ss, r_ss * theta_ss)
            
            t_tang = self.compute_blade_thickness_tangential(mc, rc*tc,
                                                                m_ps, r_ps * theta_ps, 
                                                                m_ss, r_ss * theta_ss)

            # append data to dataset
            self.rc_data.append(rc)
            self.zc_data.append(zc)
            self.thetac_data.append(tc)
            self.thk_data.append(t_tang)
            
            # if self.config.get_visual_debug():
            #     plt.figure()
            #     plt.plot(mc, t_norm, label='normal thickness')
            #     plt.plot(mc, t_tang, label='tangential thickness')
            #     plt.grid(alpha=0.2)
            #     plt.legend()
            #     plt.xlabel(r'$s$ [m]')
            #     plt.ylabel(r'$t$ [m]')

            # metal_angle = self.compute_metal_angle_along_camber(s_camber, rtheta_camber)

            if self.config.get_visual_debug():
                plt.figure()
                plt.plot(m_ps, r_ps*theta_ps, '-', color='C0', label='Pressure Side')
                plt.plot(m_ss, r_ss*theta_ss, '-', color='C1', label='Suction Side')
                # plt.plot(mc, rc*tc, '-o', color='C2', ms=2, label='Camber')
                plt.xlabel(r'$s_{m}$ [m]')
                plt.ylabel(r'$r \theta$ [m]')
                plt.legend()
                plt.title(f'Profile {i+1} of {self.number_profiles}')
                plt.grid(alpha=grid_opacity)
                plt.gca().set_aspect('equal', adjustable='box')

            # self.thickness['z'] = z_camber
            # self.thickness['r'] = r_camber
            # self.thickness['t'] = t_tang

            # self.camberSurf.add_curve(r_camber*np.cos(theta_camber), r_camber*np.sin(theta_camber), z_camber)
            self.pressureSurface.add_curve(r_ps*np.cos(theta_ps), r_ps*np.sin(theta_ps), z_ps)
            self.suctionSurface.add_curve(r_ss*np.cos(theta_ss), r_ss*np.sin(theta_ss), z_ss)

            # tCamb.append(t_tang)
            # kappaCamb.append(metal_angle)

            # self.camberSurf.loft_through_profiles(extension=0.0)
            # self.camberSurf.bspline_surface_generation()
            # if self.config.get_visual_debug(): self.camberSurf.plot_bspline_surface()
            # self.r_cambSurface, self.theta_cambSurface, self.z_cambSurface = self.camberSurf.get_global_bspline_surface(method='cylindrical')

        self.pressureSurface.bspline_surface_generation()
        if self.config.get_visual_debug(): self.pressureSurface.plot_bspline_surface()
        self.r_psSurface, self.theta_psSurface, self.z_psSurface = self.pressureSurface.get_global_bspline_surface(method='cylindrical')
        self.x_psSurface, self.y_psSurface, self.z_psSurface = self.pressureSurface.get_global_bspline_surface(method='cartesian')

        self.suctionSurface.bspline_surface_generation()
        if self.config.get_visual_debug(): self.suctionSurface.plot_bspline_surface()
        self.r_ssSurface, self.theta_ssSurface, self.z_ssSurface = self.suctionSurface.get_global_bspline_surface(method='cylindrical')
        self.x_ssSurface, self.y_ssSurface, self.z_ssSurface = self.suctionSurface.get_global_bspline_surface(method='cartesian')


        # self.thickness_camber = tCamb

        # check the full reconstructed blade
        if self.config.get_visual_debug():
            def cartesian_points(r, t, z):
                return r*np.cos(t), r*np.sin(t), z
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax.plot_surface(*cartesian_points(self.r_cambSurface, self.theta_cambSurface, self.z_cambSurface), alpha=0.1)
            ax.plot_surface(*(self.x_psSurface, self.y_psSurface, self.z_psSurface), alpha=0.4)
            ax.plot_surface(*(self.x_ssSurface, self.y_ssSurface, self.z_ssSurface), alpha=0.4)
            ax.plot(*(self.x_main, self.y_main, self.z_main), 'o', color='k')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')      
            ax.set_title('Reconstructed blade')
            ax.set_aspect('equal')
    

    def compute_meridional_coordinate(self, spline_points):
        """
        For the x,y,z points in the spline_points list, compute the associated mprime coordinate (curvilinear abscissa), 
        distinguishing also between the two sides of the blade.
        """
        x,y,z = spline_points
        r = np.sqrt(x**2+y**2)

        if self.config.get_blade_inlet_type()[self.iblade].lower()=='axial':
            le = np.argmin(z)
        else:
            le = np.argmin(r)
        
        if self.config.get_blade_outlet_type()[self.iblade].lower()=='axial':
            te = np.argmax(z)
        else:
            te = np.argmax(r)
        
        if le>te: #swap if the list of points goes from trailing to leading edge
            le_copy = le.copy()
            le = te.copy()
            te = le_copy
            
        x1,y1,z1 = x[le:te+1], y[le:te+1], z[le:te+1]
        x2,y2,z2 = x[te:], y[te:], z[te:]
        x2 = np.concatenate((x2, x[0:le+1]))
        y2 = np.concatenate((y2, y[0:le+1]))
        z2 = np.concatenate((z2, z[0:le+1]))

        def flip_orders(xp,yp,zp):
            # flip the points if they were ordered in from trailing to leading edge
            if zp[0]>zp[-1]:
                return np.flip(xp), np.flip(yp), np.flip(zp)
            else:
                return xp, yp, zp
        x1,y1,z1 = flip_orders(x1,y1,z1)
        x2,y2,z2 = flip_orders(x2,y2,z2)

        def compute_mprime_coords(xp, yp, zp):
            """
            Compute the blade to blade coordinates
            """
            rp = np.sqrt(xp**2+yp**2)
            thetap = np.arctan2(yp,xp)
            sp = np.zeros_like(rp)
            for i in range(1,len(sp)):
                dr = rp[i]-rp[i-1]
                dz = zp[i]-zp[i-1]
                dm = np.sqrt(dr**2+dz**2)
                sp[i] = sp[i-1]+dm
            return rp, thetap, sp
        r1,t1,m1 = compute_mprime_coords(x1,y1,z1)
        r2,t2,m2 = compute_mprime_coords(x2,y2,z2)

        rglob = np.concatenate((r1,r2))
        tglob = np.concatenate((t1,t2))
        mglob = np.concatenate((m1,m2))
        zglob = np.concatenate((z1,z2))

        s_camber = np.linspace(0,np.max(mglob), (len(x1)+len(x2))//2) # camber line with same number of points of the two surfaces
        coeff = np.polyfit(mglob, rglob*tglob, deg=13) 
        rt_camber = np.polyval(coeff, s_camber)
        coeff = np.polyfit(mglob, rglob, deg=3)  
        r_camber = np.polyval(coeff, s_camber)
        theta_camber = rt_camber/r_camber
        coeff = np.polyfit(mglob, zglob, deg=3)  
        z_camber = np.polyval(coeff, s_camber)

        return r1, t1, m1, z1, r2, t2, m2, z2, r_camber, theta_camber, s_camber, z_camber



    def compute_thickness(self):
        """
        Compute the blade thickness evaluating theta of the pressure and suction side on the meridional points of the camber surface
        """
        theta_ps = griddata_interpolation_with_nearest_filler(self.z_psSurface, self.r_psSurface, self.theta_psSurface, self.z_cambSurface, self.r_cambSurface)
        theta_ss = griddata_interpolation_with_nearest_filler(self.z_ssSurface, self.r_ssSurface, self.theta_ssSurface, self.z_cambSurface, self.r_cambSurface)
        
        self.thk_tang_cambSurface = self.r_cambSurface * np.abs(theta_ps - theta_ss)
                
        if self.config.get_visual_debug():
            self.contour_template(self.z_cambSurface, self.r_cambSurface, self.thk_tang_cambSurface, 'Tangential Thickness')

        # if np.mean(self.thk_tang_cambSurface) > 0:
        #     for ii in range(self.thk_tang_cambSurface.shape[0]):
        #         for jj in range(self.thk_tang_cambSurface.shape[1]):
        #             if self.thk_tang_cambSurface[ii, jj] < 0:
        #                 self.thk_tang_cambSurface[ii, jj] = 0
        # else:
        #     for ii in range(self.thk_tang_cambSurface.shape[0]):
        #         for jj in range(self.thk_tang_cambSurface.shape[1]):
        #             if self.thk_tang_cambSurface[ii, jj] < 0:
        #                 self.thk_tang_cambSurface[ii, jj] *= -1
        #             else:
        #                 self.thk_tang_cambSurface[ii, jj] = 0


    def compute_thickness_along_camber(self):
        """
        Compute thickness for each points on the spline along the camber
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
        Find and return the intersection between 2 curves. static method because it is bound to the class, not to an instance
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


    def twoD_function_evaluation(self, z, r, theta, z_eval, r_eval, method):
        """
        Routine to evaluate whatever dataset (here called theta) as a function of the z and r.

        Parameters
        --------------------------------------
        
        `z`: np.ndarray of z coordinates where `theta` is stored

        `r`: np.ndarray of r coordinates where `theta` is stored

        `theta`: np.ndarray of theta values corresponding to `z` and `r`

        `z_eval`: np.ndarray of z coordinates where evaluating `theta`

        `r_eval`: np.ndarray of r coordinates where evaluating `theta`

        `method`: regression or interpolation (linear) of the function theta(z,r)

        """
        smooth = 1 # rbf interpolation smoother
        degree = self.config.get_blade_reconstruction_regression_order()

        if method == 'regression': # polynomial regression of order <degree>
            poly_features = PolynomialFeatures(degree)  
            X = poly_features.fit_transform(np.column_stack((z, r)))
            model = LinearRegression()
            model.fit(X, theta)
            coefficients = model.coef_
            intercept = model.intercept_
            X_eval = poly_features.fit_transform(np.column_stack((z_eval.flatten(), r_eval.flatten())))
            surface_values = np.dot(X_eval, coefficients) + intercept
            theta_eval = surface_values.reshape(z_eval.shape)
        elif method == 'interpolation': # linear interpolation, with nearest-neighbor for the extrapolated points
            theta_eval = griddata_interpolation_with_nearest_filler(z, r, theta, z_eval, r_eval)
        else:
            raise ValueError('Unknown method')

        return theta_eval


    def obtain_quantities_on_meridional_grid(self, smooth=1):
        """
        Find the camber information on the blade grid via interpolation of the various functions stored on the camber grid.
        Check the degree of the polynomial if it is ok.
        """
        self.z_camber = self.z_grid
        self.r_camber = self.r_grid
        
        method = self.config.get_blades_camber_reconstruction()[self.iblade].lower()
        
        self.theta_camber = self.twoD_function_evaluation(self.z_cambSurface.flatten(),
                                                         self.r_cambSurface.flatten(),
                                                         (self.theta_cambSurface).flatten(),
                                                         self.z_grid, self.r_grid,
                                                         method)
        self.x_camber = self.r_grid * np.cos(self.theta_camber)
        self.y_camber = self.r_grid * np.sin(self.theta_camber)

        self.blockage = self.twoD_function_evaluation(self.z_cambSurface.flatten(),
                                                     self.r_cambSurface.flatten(),
                                                     self.blockage_cambSurface.flatten(),
                                                     self.z_grid, self.r_grid, method)

        self.nr = self.twoD_function_evaluation(self.z_cambSurface.flatten(),
                                               self.r_cambSurface.flatten(),
                                               self.n_camber_r.flatten(),
                                               self.z_grid, self.r_grid, method)

        self.nt = self.twoD_function_evaluation(self.z_cambSurface.flatten(),
                                                   self.r_cambSurface.flatten(),
                                                   self.n_camber_t.flatten(),
                                                   self.z_grid, self.r_grid, method)

        self.nz = self.twoD_function_evaluation(self.z_cambSurface.flatten(),
                                               self.r_cambSurface.flatten(),
                                               self.n_camber_z.flatten(),
                                               self.z_grid, self.r_grid, method)
    

    def obtain_quantities_on_meridional_grid_secondversion(self, smooth=1):
        """
        Find the camber information on the blade grid via interpolation of the various functions stored on the camber grid.
        Check the degree of the polynomial if it is ok.
        """
        self.z_camber = self.z_grid
        self.r_camber = self.r_grid
        
        method = self.config.get_blades_camber_reconstruction()[self.iblade].lower()
        
        def unroll_list_in_nparray(l):
            arr = np.concatenate(l)
            return arr
        
        z_data = unroll_list_in_nparray(self.zss_data)
        r_data = unroll_list_in_nparray(self.rss_data)
        theta_data = unroll_list_in_nparray(self.thetass_data)
        theta_ss = self.twoD_function_evaluation(z_data, r_data, (theta_data), self.z_grid, self.r_grid, method)
        self.contour_template(self.z_grid, self.r_grid, theta_ss*180/np.pi, r'$\theta_{ss}$ [deg]')

        z_data = unroll_list_in_nparray(self.zps_data)
        r_data = unroll_list_in_nparray(self.rps_data)
        theta_data = unroll_list_in_nparray(self.thetaps_data)
        theta_ps = self.twoD_function_evaluation(z_data, r_data, (theta_data), self.z_grid, self.r_grid, method)
        self.contour_template(self.z_grid, self.r_grid, theta_ps*180/np.pi, r'$\theta_{ps}$ [deg]')

        self.theta_camber = 0.5*(theta_ps+theta_ss)
        self.contour_template(self.z_grid, self.r_grid, self.theta_camber*180/np.pi, r'$\theta_{c}$ [deg]')
        self.thk = self.r_grid*np.abs(theta_ps-theta_ss)
        self.contour_template(self.z_grid, self.r_grid, self.thk, r'$t$ [m]')

        
        # z_data = unroll_list_in_nparray(self.zc_data)
        # r_data = unroll_list_in_nparray(self.rc_data)
        # theta_data = unroll_list_in_nparray(self.thetac_data)
        # thk_data = unroll_list_in_nparray(self.thk_data)

        # self.theta_camber = self.twoD_function_evaluation(z_data, r_data, (theta_data),
        #                                                  self.z_grid, self.r_grid,
        #                                                  method)
        # self.contour_template(self.z_grid, self.r_grid, self.theta_camber*180/np.pi, r'$\theta_c$ [deg]')
        # self.x_camber = self.r_grid * np.cos(self.theta_camber)
        # self.y_camber = self.r_grid * np.sin(self.theta_camber)

        # self.thk = self.twoD_function_evaluation(z_data, r_data, (thk_data),
        #                                         self.z_grid, self.r_grid,
        #                                         method)
        # self.contour_template(self.z_grid, self.r_grid, self.thk, r'$t$ [m]')

        Nb = self.config.get_blades_number()[self.iblade]
        self.blockage = 1 - Nb * self.thk / (2*np.pi*self.r_grid)
        self.contour_template(self.z_grid, self.r_grid, self.blockage, r'$b$ [-]')

        # self.nr = self.twoD_function_evaluation(self.z_cambSurface.flatten(),
        #                                        self.r_cambSurface.flatten(),
        #                                        self.n_camber_r.flatten(),
        #                                        self.z_grid, self.r_grid, method)

        # self.nt = self.twoD_function_evaluation(self.z_cambSurface.flatten(),
        #                                            self.r_cambSurface.flatten(),
        #                                            self.n_camber_t.flatten(),
        #                                            self.z_grid, self.r_grid, method)

        # self.nz = self.twoD_function_evaluation(self.z_cambSurface.flatten(),
        #                                        self.r_cambSurface.flatten(),
        #                                        self.n_camber_z.flatten(),
        #                                        self.z_grid, self.r_grid, method)
    

    def obtain_quantities_on_meridional_grid_thirdversion(self, smooth=1):
        """
        Find the camber information on the blade grid via interpolation of the various functions stored on the camber grid.
        Check the degree of the polynomial if it is ok.
        """
        self.z_camber = self.z_grid
        self.r_camber = self.r_grid
        
        method = self.config.get_blades_camber_reconstruction()[self.iblade].lower()
        
        theta_ss = self.twoD_function_evaluation(self.z_ssSurface, self.r_ssSurface, (self.theta_ssSurface), self.z_grid, self.r_grid, method)
        self.contour_template(self.z_grid, self.r_grid, theta_ss*180/np.pi, r'$\theta_{ss}$ [deg]')

        theta_ps = self.twoD_function_evaluation(self.z_psSurface, self.r_psSurface, (self.theta_psSurface), self.z_grid, self.r_grid, method)
        self.contour_template(self.z_grid, self.r_grid, theta_ps*180/np.pi, r'$\theta_{ps}$ [deg]')

        self.theta_camber = 0.5*(theta_ps+theta_ss)
        self.contour_template(self.z_grid, self.r_grid, self.theta_camber*180/np.pi, r'$\theta_{c}$ [deg]')
        self.thk = self.r_grid*np.abs(theta_ps-theta_ss)
        self.contour_template(self.z_grid, self.r_grid, self.thk, r'$t$ [m]')

        Nb = self.config.get_blades_number()[self.iblade]
        self.blockage = 1 - Nb * self.thk / (2*np.pi*self.r_grid)
        self.contour_template(self.z_grid, self.r_grid, self.blockage, r'$b$ [-]')

    
    def compute_thollet_angles(self):
        dtdz, dtdr = compute_gradient_least_square(self.z_camber, self.r_camber, self.theta_camber)
        beta = np.arctan(self.r_grid*dtdz)
        self.contour_template(self.z_camber, self.r_camber, beta*180/np.pi, 'metal angle thollet [deg]')
        lmbda = np.arctan(self.r_grid*dtdr)
        self.contour_template(self.z_camber, self.r_camber, lmbda*180/np.pi, 'lean angle thollet [deg]')


    def add_meridional_grid(self, zgrid, rgrid):
        """
        Add the meridional grid to the blade taken from the block object
        """
        self.z_grid, self.r_grid = zgrid, rgrid


    def compute_streamline_length(self, normalize=False):
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
                self.streamline_length[:, jj] /= self.streamline_length[-1, jj]-self.streamline_length[0, jj]


    def compute_spanline_length(self, normalize=False):
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
                self.spanline_length[ii, :] /= self.spanline_length[ii, -1]-self.spanline_length[ii, 0]


    def plot_streamline_length_contour(self, save_filename=None):
        """
        plot the streamline length contour
        """
        self.contour_template(self.z_grid, self.r_grid, self.streamline_length, name=r'$\bar{s}_{stw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_streamline_length.pdf', bbox_inches='tight')


    def plot_spanline_length_contour(self, save_filename=None):
        """
        plot the spanline length contour
        """
        self.contour_template(self.z_grid, self.r_grid, self.spanline_length, name=r'$\bar{s}_{spw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_spanline_length.pdf', bbox_inches='tight')


    def plot_camber_surface(self, save_filename=None, sides=False, points=True):
        """
        Plot the main blade points and the camber surface
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x_cambSurface, self.y_cambSurface, self.z_cambSurface, alpha=0.5, color='red', label='reference camber')
        ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, alpha=0.5, color='blue', label='regressed camber')
        if points:
            ax.scatter(self.x_main, self.y_main, self.z_main, c='black', s=1, label='points')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')

        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + save_filename + '.pdf', bbox_inches='tight')


    def plot_camber_meridional_grid(self, save_filename=None):
        """
        plot the main camber meridional grid
        """
        plt.figure()
        plt.scatter(self.z_camber, self.r_camber)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + save_filename + '.pdf', bbox_inches='tight')


    def plot_camber_normal_contour_on_loft(self):
        """
        plot the camber normal vector contours
        """
        self.contour_template(self.z_cambSurface, self.r_cambSurface, self.n_camber_r, r'$n_r$ reference')
        self.contour_template(self.z_cambSurface, self.r_cambSurface, self.n_camber_t, r'$n_{\theta}$ reference')
        self.contour_template(self.z_cambSurface, self.r_cambSurface, self.n_camber_z, r'$n_z$ reference')


    def plot_blockage_contour(self, save_filename=None):
        """
        plot the blockage
        """
        self.contour_template(self.z_grid, self.r_grid, self.blockage, name=r'$b$', vmax=1)
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_blockage.pdf', bbox_inches='tight')


    def plot_camber_normal_contour(self, save_filename=None):
        """
        plot the camber normal vector contours
        """
        self.contour_template(self.z_camber, self.r_camber, self.n_camber_r, name=r'$n_r$')
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_normal_r.pdf', bbox_inches='tight')

        self.contour_template(self.z_camber, self.r_camber, self.n_camber_t, name=r'$n_{\theta}$')
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_normal_theta.pdf', bbox_inches='tight')

        self.contour_template(self.z_camber, self.r_camber, self.n_camber_z, name=r'$n_z$')
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_normal_z.pdf', bbox_inches='tight')


    def write_bfm_input_file(self, filename=None, rescale=True):
        """
        Write the SU2 BFM input file
        """
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
            stream_v = np.array([xgrid[i, j] - xgrid[i - 2, j],
                                 ygrid[i, j] - ygrid[i - 2, j],
                                 zgrid[i, j] - zgrid[i - 2, j]])
        elif i == 0:
            stream_v = np.array([xgrid[i + 2, j] - xgrid[i, j],
                                 ygrid[i + 2, j] - ygrid[i, j],
                                 zgrid[i + 2, j] - zgrid[i, j]])
        else:
            stream_v = np.array([xgrid[i + 1, j] - xgrid[i - 1, j],
                                 ygrid[i + 1, j] - ygrid[i - 1, j],
                                 zgrid[i + 1, j] - zgrid[i - 1, j]])
        stream_v /= np.linalg.norm(stream_v)

        # compute versor along the second direction
        if j == nj:
            span_v = np.array([xgrid[i, j] - xgrid[i, j - 2],
                               ygrid[i, j] - ygrid[i, j - 2],
                               zgrid[i, j] - zgrid[i, j - 2]])
        elif j == 0:
            span_v = np.array([xgrid[i, j + 2] - xgrid[i, j],
                               ygrid[i, j + 2] - ygrid[i, j],
                               zgrid[i, j + 2] - zgrid[i, j]])
        else:
            span_v = np.array([xgrid[i, j + 1] - xgrid[i, j - 1],
                               ygrid[i, j + 1] - ygrid[i, j - 1],
                               zgrid[i, j + 1] - zgrid[i, j - 1]])
        span_v /= np.linalg.norm(span_v)

        # the normal is the vectorial product of the two
        normal = np.cross(stream_v, span_v)
        normal /= np.linalg.norm(normal)

        if check:
            arrow_len = (np.max(zgrid)-np.min(zgrid))/3
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(xgrid, ygrid, zgrid, alpha=0.3)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_zlabel(r'$z$')
            ax.set_aspect('equal', adjustable='box')
            ax.quiver(xgrid[i, j], ygrid[i, j], zgrid[i, j], stream_v[0], stream_v[1], stream_v[2], length=arrow_len, color='red')
            ax.quiver(xgrid[i, j], ygrid[i, j], zgrid[i, j], span_v[0], span_v[1], span_v[2], length=arrow_len, color='green')
            ax.quiver(xgrid[i, j], ygrid[i, j], zgrid[i, j], normal[0], normal[1], normal[2], length=arrow_len, color='blue')
            pass
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


    def find_inlet_points(self):
        """
        Find the points defining the inlet from the coordinates of the blade points.
        """
        iblade = self.iblade
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


    def find_outlet_points(self):
        """
        find the points defining the inlet are taken as
        the points with minimum z cordinates for each profile of the blade.
        """
        iblade = self.iblade
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


    def compute_normal_vectors_on_reference_surface(self, visual_debug=True):
        """
        for every point discretized on the camber surface, compute the normal vector, the streamline vector and the
        spanline vector, all in cartesian and cylindrical reference systems.
        """
        self.x_cambSurface = self.r_cambSurface*np.cos(self.theta_cambSurface)
        self.y_cambSurface = self.r_cambSurface * np.sin(self.theta_cambSurface)

        # Create 2D NumPy array of empty arrays
        self.normal_vectors = np.empty(self.z_cambSurface.shape, dtype=object)

        # compute also the vector in cylindrical cordinates
        self.normal_vectors_cyl = np.empty(self.z_cambSurface.shape, dtype=object)

        for i in range(0, self.z_cambSurface.shape[0]):
            for j in range(0, self.z_cambSurface.shape[1]):
                self.normal_vectors[i, j] = self.compute_camber_vector(i, j, self.x_cambSurface, self.y_cambSurface, self.z_cambSurface, check=False)[0]

                self.normal_vectors_cyl[i, j] = cartesian_to_cylindrical(self.x_cambSurface[i, j],
                                                                         self.y_cambSurface[i, j],
                                                                         self.z_cambSurface[i, j],
                                                                         self.normal_vectors[i, j])

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

        if visual_debug:
            arrow_len = (np.max(self.z_cambSurface) - np.min(self.z_cambSurface)) / 5
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.x_cambSurface, self.y_cambSurface, self.z_cambSurface, alpha=0.3)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_zlabel(r'$z$')
            ax.set_aspect('equal', adjustable='box')
            for i in range(0, self.z_cambSurface.shape[0], 10):
                for j in range(0, self.z_cambSurface.shape[1], 5):
                    ax.quiver(self.x_cambSurface[i, j], self.y_cambSurface[i, j], self.z_cambSurface[i, j], self.normal_vectors[i,j][0], self.normal_vectors[i,j][1], self.normal_vectors[i,j][2], length=arrow_len, color='blue')
            pass


    def compute_camber_vectors(self):
        """
        for every point discretized on the camber surface, compute the normal vector, the streamline vector and the
        spanline vector, all in cartesian and cylindrical reference systems.
        """
        self.x_camber = self.r_camber * np.cos(self.theta_camber)
        self.y_camber = self.r_camber * np.sin(self.theta_camber)

        # Create 2D NumPy array of empty arrays
        self.normal_vectors = np.empty(self.x_camber.shape, dtype=object)
        self.streamline_vectors = np.empty(self.x_camber.shape, dtype=object)
        self.spanline_vectors = np.empty(self.x_camber.shape, dtype=object)

        # compute also the vector in cylindrical cordinates
        self.normal_vectors_cyl = np.empty(self.x_camber.shape, dtype=object)
        self.streamline_vectors_cyl = np.empty(self.x_camber.shape, dtype=object)
        self.spanline_vectors_cyl = np.empty(self.x_camber.shape, dtype=object)

        for i in range(0, self.x_camber.shape[0]):
            for j in range(0, self.x_camber.shape[1]):
                self.normal_vectors[i, j], self.streamline_vectors[i, j], self.spanline_vectors[
                    i, j] = self.compute_camber_vector(i, j, self.x_camber, self.y_camber, self.z_camber, check=False)

                self.normal_vectors_cyl[i, j] = cartesian_to_cylindrical(self.x_camber[i, j],
                                                                         self.y_camber[i, j],
                                                                         self.z_camber[i, j],
                                                                         self.normal_vectors[i, j])
                self.streamline_vectors_cyl[i, j] = cartesian_to_cylindrical(self.x_camber[i, j],
                                                                             self.y_camber[i, j],
                                                                             self.z_camber[i, j],
                                                                             self.streamline_vectors[i, j])
                self.spanline_vectors_cyl[i, j] = cartesian_to_cylindrical(self.x_camber[i, j],
                                                                           self.y_camber[i, j],
                                                                           self.z_camber[i, j],
                                                                           self.spanline_vectors[i, j])

        # reorder the vectors in 2d arrays
        self.n_camber_r = np.zeros_like(self.x_camber)
        self.n_camber_t = np.zeros_like(self.x_camber)
        self.n_camber_z = np.zeros_like(self.x_camber)
        for i in range(0, self.x_camber.shape[0]):
            for j in range(0, self.x_camber.shape[1]):
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


    def compute_blade_thickness_normal_to_camber(self, xc, yc, xps, yps, xss, yss, debug=False):
        """
        Compute the blade thickness in the direction perpendicular to the local camber
        """
        visual_debug = debug
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

            try:
                if (len(x_int_ss)*len(x_int_ps)*len(y_int_ss)*len(y_int_ps)>0):
                    thk_normal[iPoint] = np.sqrt((x_int_ps[0]-x_int_ss[0])**2 + (y_int_ps[0]-y_int_ss[0])**2)
                else:
                    thk_normal[iPoint] = 0
            except:
                thk_normal[iPoint] = 0

        return thk_normal

    def compute_blade_thickness_tangential(self, xc, yc, xps, yps, xss, yss, debug=False):
        """
        Compute the blade thickness in the circumferential direction
        """
        visual_debug = debug

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
        
        if debug:
            plt.figure()
            plt.plot(xc, thk, '-o', markersize=3)
            plt.xlabel(r'$z$')
            plt.title('Tangential Thickness')
        return thk


    def compute_blade_blockage_on_camber_loft(self):
        """
        Compute blade blockage based on the thickness of the blade in tangential direction
        """
        Nb = self.config.get_blades_number()[self.iblade]
        self.blockage_cambSurface = 1 - Nb * self.thk_tang_cambSurface / (2*np.pi*self.r_cambSurface)
        self.contour_template(self.z_cambSurface, self.r_cambSurface, self.blockage_cambSurface, r'$b$ reference')


    def compute_blade_blockage_gradient(self, save_filename=None):
        """
        Compute the blockage gradient via finite difference on the meridional grid
        """
        self.db_dz, self.db_dr = compute_gradient_least_square(self.z_camber, self.r_camber, self.blockage)

        self.contour_template(self.z_camber, self.r_camber, self.db_dz, r'$\partial_z b$')
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_' + '_dbdz.pdf', bbox_inches='tight')

        self.contour_template(self.z_camber, self.r_camber, self.db_dr, r'$\partial_r b$')
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_' + '_dbdr.pdf', bbox_inches='tight')

        self.contour_template(self.z_camber, self.r_camber, np.sqrt(self.db_dr**2+self.db_dz**2), r'$| \nabla b|$')
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_bgrad_magnitude.pdf', bbox_inches='tight')


    def plot_blockage_and_grad_leading_to_trailing(self, jump=10, save_filename=None):
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
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_' + 'blockage_slices.pdf', bbox_inches='tight')

        plt.figure()
        for ispan in stations:
            plt.plot(self.streamline_length[:, ispan], self.db_dz[:, ispan], '-s', ms=3, label=r'$i_{span}: \ %i/%i$' %(ispan, self.blockage.shape[1]-1))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylabel(r'$db/dz \ \rm{[1/m]}$')
        plt.xlabel(r'$\bar{s}_{stw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_' + 'dbdz_slices.pdf', bbox_inches='tight')

        plt.figure()
        for ispan in stations:
            plt.plot(self.streamline_length[:, ispan], self.db_dr[:, ispan], '-s', ms=3, label=r'$i_{span}: \ %i/%i$' %(ispan, self.blockage.shape[1]-1))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylabel(r'$db/dr \ \rm{[1/m]}$')
        plt.xlabel(r'$\bar{s}_{stw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_' + 'dbdr_slices.pdf', bbox_inches='tight')


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


    def compute_blade_camber_angles(self, convention='neutral'):
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
                self.gas_path_angle[i, j] = np.arctan2(self.streamline_vectors_cyl[i, j][0] , self.streamline_vectors_cyl[i, j][2])

                # meridional_sl_vec = np.array([self.streamline_vectors_cyl[i, j][0], 0, self.streamline_vectors_cyl[i, j][2]])
                # meridional_sl_vec /= np.linalg.norm(meridional_sl_vec)

                # meridional_sp_vec = np.array([self.spanline_vectors_cyl[i, j][0], 0, self.spanline_vectors_cyl[i, j][2]])
                # meridional_sp_vec /= np.linalg.norm(meridional_sp_vec)

                # if convention == 'neutral':
                #     self.blade_metal_angle[i, j] = np.arccos(np.dot(self.streamline_vectors_cyl[i, j], meridional_sl_vec))
                #     self.blade_lean_angle[i, j] = np.arccos(np.dot(self.spanline_vectors_cyl[i, j], meridional_sp_vec))
                # elif convention == 'rotation-wise':
                #     self.blade_metal_angle[i, j] = -np.arccos(np.dot(self.streamline_vectors_cyl[i, j], meridional_sl_vec))
                #     self.blade_lean_angle[i, j] = -np.arccos(np.dot(self.spanline_vectors_cyl[i, j], meridional_sp_vec))
                # else:
                #     raise ValueError('Choose a convention for the angles')
        
        self.blade_metal_angle = np.arctan2(self.n_camber_t, self.n_camber_z)
        self.blade_lean_angle = np.arctan2(self.n_camber_r, np.sqrt(self.n_camber_t**2+self.n_camber_z**2))


    def show_blade_angles_contour(self, save_filename=None, folder_name=None):
        """
        Contour of the blade angles.
        :param save_filename: if specified, saves the plots with the given name
        """
        self.contour_template(self.z_camber, self.r_camber, 180 / np.pi * self.gas_path_angle, r'$\varphi \quad \mathrm{[deg]}$')
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_gas_path_angle.pdf', bbox_inches='tight')

        self.contour_template(self.z_camber, self.r_camber, 180 / np.pi * self.blade_metal_angle, r'$\kappa \quad \mathrm{[deg]}$')
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_blade_metal_angle.pdf', bbox_inches='tight')

        self.contour_template(self.z_camber, self.r_camber, 180 / np.pi * self.blade_lean_angle, r'$\lambda \quad \mathrm{[deg]}$')
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_blade_lean_angle.pdf', bbox_inches='tight')


    def plot_inlet_outlet_metal_angle(self, save_filename=None, spans=(0, 0.25, 0.5, 0.75, 1)):
        """
        Plot inlet and metal angle
        """
        def normalize(f, dir):
            ni,nj = f.shape
            fnew = np.zeros_like(f)
            if dir==0:
                for j in range(nj):
                    fnew[:,j] = (f[:,j]-f[:,j].min())/(f[:,j].max()-f[:,j].min())
            else:
                for i in range(ni):
                    fnew[i,:] = (f[i,:]-f[i,:].min())/(f[i,:].max()-f[i,:].min())
            return fnew
        
        stream_len = normalize(self.streamline_length, dir=0)
        span_len = normalize(self.spanline_length, dir=1)

        plt.figure()
        plt.plot(span_len[0,:], self.blade_metal_angle[0,:]*180/np.pi, '-o', ms=3, mfc='none', label='leading edge')
        plt.plot(span_len[-1, :], self.blade_metal_angle[-1, :]*180/np.pi, '-s', ms=3, mfc='none', label='trailing edge')
        plt.xlabel('Normalized Span Length [-]')
        plt.ylabel(r'Blade Metal Angle [deg]')
        plt.grid(alpha=0.2)
        plt.legend()
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_metal_angle_span.pdf', bbox_inches='tight')

        idx_spans = self.compute_blade_span_indexes(spans, span_len)

        plt.figure()
        for ispan in idx_spans:
            plt.plot(stream_len[:, ispan], self.blade_metal_angle[:, ispan] * 180 / np.pi, '-o', ms=3, mfc='none', label='span %.3f' %(span_len[0,ispan]))
        plt.xlabel('Normalized Stream Length [-]')
        plt.ylabel(r'Blade Metal Angle [deg]')
        plt.grid(alpha=0.2)
        plt.legend()
        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_metal_angle_stream.pdf', bbox_inches='tight')


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


    def compute_blade_span_indexes(self, spans, span_len):
        """
        Given a tuple of spans (normalized from 0-hub to 1-tip), return the indexes of the meridional grid as close as possible
        to those values.
        """
        idx_spans = np.zeros(len(spans), dtype=int)
        span_len = span_len[0, :]
        for ii, span in enumerate(spans):
            idx_spans[ii] = np.argmin(np.abs(span_len-span))
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


    def process_paraview_dataset_kiwada(self, folder_path, average_type='raw', CP=1005, R=287, TREF=288.15, PREF=101300, inviscid=False):
        """
        Read the processed dataset stored in folder_path location obtained by the Paraview Macro, for The Kiwada Extraction procedure.
        Average type distinguish the type of average used, raw for standard circumferential.
        Inviscid=True sets the viscous stresses to zero, leading to inviscid force extraction.
        """
        available_avg_types = ['raw', 'density', 'axialMomentum']
        self.avg_type = average_type
        if self.avg_type not in available_avg_types:
            raise ValueError('Not valid average type')
        print('Weighted average type: %s' % self.avg_type)

        def extract_grid_location(file_name):
            print('Elaborating Filename: ' + file_name)
            file_name = file_name.strip('spline_data_')
            file_name = file_name.strip('.csv')
            file_name = file_name.split('_')
            nz = int(file_name[0])
            nr = int(file_name[1])
            return nz, nr
       
        data_dir = folder_path
        files = [f for f in os.listdir(data_dir) if '.csv' in f]
        files = sorted(files)

        # give the name of the fields to average
        fields = ['Density', 'Energy', 'Mach', 'Eddy_Viscosity', 'Pressure', 'Temperature', 'Grid_Velocity_Tangential',
                    'Velocity_Radial', 'Velocity_Tangential', 'Velocity_Tangential_Relative', 'Velocity_Axial', 'Entropy',
                    'A1', 'A2', 'R2', 'R3', 'T1', 'T2', 'T3',
                    'Tau_rr', 'Tau_tt', 'Tau_zz', 'Tau_rt', 'Tau_rz', 'Tau_tz']
        
        nz, nr = extract_grid_location(files[-1])
        field_grids = {}
        for field_name in fields:
            field_grids[field_name] = np.zeros((nz+1, nr+1))
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

            # Compute additional fields that will be circumferentially averaged 
            data_dict['Velocity_Radial'] = data_dict['Velocity_0']*cos(theta)+data_dict['Velocity_1']*sin(theta)

            data_dict['Velocity_Tangential'] = -data_dict['Velocity_0']*sin(theta)+data_dict['Velocity_1']*cos(theta)

            data_dict['Grid_Velocity_Tangential'] = -data_dict['Grid_Velocity_0']*sin(theta)+data_dict['Grid_Velocity_1']*cos(theta)

            data_dict['Velocity_Tangential_Relative'] = data_dict['Velocity_Tangential']-data_dict['Grid_Velocity_Tangential'] 
            
            data_dict['Velocity_Axial'] = data_dict['Velocity_2']

            data_dict['Entropy'] = CP*np.log(data_dict['Temperature']/TREF)-R*np.log(data_dict['Pressure']/PREF)

            duxdx = data_dict['Velocity_X_Grad_0']
            duxdy = data_dict['Velocity_X_Grad_1']
            duxdz = data_dict['Velocity_X_Grad_2']
            duydx = data_dict['Velocity_Y_Grad_0']
            duydy = data_dict['Velocity_Y_Grad_1']
            duydz = data_dict['Velocity_Y_Grad_2']
            duzdx = data_dict['Velocity_Z_Grad_0']
            duzdy = data_dict['Velocity_Z_Grad_1']
            duzdz = data_dict['Velocity_Z_Grad_2']
            div_u = duxdx+duydy+duzdz
            mu = (data_dict['Laminar_Viscosity']+data_dict['Eddy_Viscosity'])*data_dict['Density']

            tauxx = mu*(2*duxdx)-2/3*mu*div_u
            tauyy = mu*(2*duydy)-2/3*mu*div_u
            tauzz_cart = mu*(2*duzdz)-2/3*mu*div_u
            tauxy = mu*(duydx+duxdy)
            tauyz = mu*(duydz+duzdy)
            tauxz = mu*(duxdz+duzdx)
            trace_cart = tauxx+tauyy+tauzz_cart

            # plt.figure()
            # plt.plot(theta*180/pi, tauxx, '.', label=r'$\tau_{xx}$')
            # plt.plot(theta*180/pi, tauyy, '.', label=r'$\tau_{yy}$')
            # plt.plot(theta*180/pi, tauzz_cart, '.', label=r'$\tau_{zz}$')
            # plt.plot(theta*180/pi, trace_cart, '.', label=r'$\tau_{xx} + \tau_{yy} + \tau_{zz}$')
            # plt.xlabel(r'$\theta$ [deg]')
            # plt.ylabel(r'$\tau$ [Pa]')
            # plt.legend()

            # plt.figure()
            # plt.plot(theta*180/pi, tauxx, '.', label=r'$\tau_{xx}$')
            # plt.plot(theta*180/pi, tauyy, '.', label=r'$\tau_{yy}$')
            # plt.plot(theta*180/pi, tauzz_cart, '.', label=r'$\tau_{zz}$')
            # plt.plot(theta*180/pi, tauxy, '.', label=r'$\tau_{xy}$')
            # plt.plot(theta*180/pi, tauxz, '.', label=r'$\tau_{xz}$')
            # plt.plot(theta*180/pi, tauyz, '.', label=r'$\tau_{yz}$')
            # plt.xlabel(r'$\theta$ [deg]')
            # plt.ylabel(r'$\tau$ [Pa]')
            # plt.legend()

            taurr = np.zeros_like(tauxx) 
            tautt = np.zeros_like(tauxx) 
            tauzz_cyl = np.zeros_like(tauxx) 
            taurt = np.zeros_like(tauxx) 
            taurz = np.zeros_like(tauxx) 
            tautz = np.zeros_like(tauxx) 
            for i in range(len(tauxz)):
                TAU = np.zeros((3,3))
                TAU[0,0] = tauxx[i]
                TAU[0,1] = tauxy[i]
                TAU[0,2] = tauxz[i]
                TAU[1,0] = tauxy[i]
                TAU[1,1] = tauyy[i]
                TAU[1,2] = tauyz[i]
                TAU[2,0] = tauxz[i]
                TAU[2,1] = tauyz[i]
                TAU[2,2] = tauzz_cart[i]
                TAU_cyl = rotate_cartesian_to_cylindric_tensor(theta[i], TAU)
                taurr[i] = TAU_cyl[0,0]
                tautt[i] = TAU_cyl[1,1]
                tauzz_cyl[i] = TAU_cyl[2,2]
                taurt[i] = TAU_cyl[0,1]
                taurz[i] = TAU_cyl[0,2]
                tautz[i] = TAU_cyl[1,2]
            data_dict['Tau_rr'] = taurr
            data_dict['Tau_tt'] = tautt
            data_dict['Tau_zz'] = tauzz_cyl
            data_dict['Tau_rt'] = taurt
            data_dict['Tau_rz'] = taurz
            data_dict['Tau_tz'] = tautz
            # trace_cyl = taurr+tautt+tauzz_cyl
            # plt.figure()
            # plt.plot(theta*180/pi, taurr, '.', label=r'$\tau_{rr}$')
            # plt.plot(theta*180/pi, tautt, '.', label=r'$\tau_{\theta \theta}$')
            # plt.plot(theta*180/pi, tauzz_cyl, '.', label=r'$\tau_{zz}$')
            # plt.plot(theta*180/pi, trace_cyl, '.', label=r'$\tau_{rr} + \tau_{\theta} + \tau_{zz}$')
            # plt.xlabel(r'$\theta$ [deg]')
            # plt.ylabel(r'$\tau$ [Pa]')
            # plt.legend()

            # plt.figure()
            # plt.plot(theta*180/pi, trace_cart, 'o', mfc='none', label='Cartesian')
            # plt.plot(theta*180/pi, trace_cyl, 'x', label='Cylindrical')
            # plt.xlabel(r'$\theta$ [deg]')
            # plt.ylabel(r'Tr$(\tau)$ [Pa]')
            # plt.legend()

            # plt.figure()
            # plt.plot(theta*180/pi, tauzz_cart, 'o', mfc='none', label='Cartesian')
            # plt.plot(theta*180/pi, tauzz_cyl, 'x', label='Cylindrical')
            # plt.xlabel(r'$\theta$ [deg]')
            # plt.ylabel(r'$\tau_{zz}$ [Pa]')
            # plt.legend()
            
            # plt.figure()
            # plt.plot(theta*180/pi, taurr, '.', label=r'$\tau_{rr}$')
            # plt.plot(theta*180/pi, tautt, '.', label=r'$\tau_{\theta \theta}$')
            # plt.plot(theta*180/pi, tauzz_cyl, '.', label=r'$\tau_{zz}$')
            # plt.plot(theta*180/pi, taurt, '.', label=r'$\tau_{r \theta}$')
            # plt.plot(theta*180/pi, taurz, '.', label=r'$\tau_{r z}$')
            # plt.plot(theta*180/pi, tautz, '.', label=r'$\tau_{\theta z}$')
            # plt.xlabel(r'$\theta$ [deg]')
            # plt.ylabel(r'$\tau$ [Pa]')
            # plt.legend()

            # follow nomenclature page 89 of Magrini
            if inviscid:
                taurr *=0
                tautt *=0
                tauzz_cyl *= 0
                tauzz_cart *= 0
                taurt *= 0
                taurz *= 0
                tautz *= 0
                
            data_dict['A1'] = data_dict['Density']*data_dict['Velocity_Axial']**2+data_dict['Pressure']-tauzz_cyl
            data_dict['A2'] = data_dict['Density']*data_dict['Velocity_Axial']*data_dict['Velocity_Radial']-taurz
            data_dict['R2'] = data_dict['Density']*data_dict['Velocity_Radial']**2+data_dict['Pressure']-taurr
            data_dict['R3'] = data_dict['Density']*data_dict['Velocity_Tangential']**2+data_dict['Pressure']-tautt
            data_dict['T1'] = data_dict['Density']*data_dict['Velocity_Axial']*data_dict['Velocity_Tangential']-tautz
            data_dict['T2'] = data_dict['Density']*data_dict['Velocity_Radial']*data_dict['Velocity_Tangential']-taurt
            data_dict['T3'] = data_dict['Density']*data_dict['Velocity_Radial']*data_dict['Velocity_Tangential']-taurt 

            for field in fields:
                f = data_dict[field].copy()
                if self.avg_type.lower() == 'raw':
                    field_grids[field][stream_id, span_id] = np.sum(f) / len(f)
                elif self.avg_type.lower() == 'density':
                    field_grids[field][stream_id, span_id] = np.sum(f * data_dict['Density']) / np.sum(data_dict['Density'])
                elif self.avg_type.lower() == 'axialMomentum':
                    field_grids[field][stream_id, span_id] = np.sum(f * data_dict['Momentum_2']) / np.sum(data_dict['Momentum_2'])

        self.meridional_fields = field_grids
        self.meridional_fields['Z'] = z_grid
        self.meridional_fields['R'] = r_grid
    

    def process_paraview_dataset_marble(self, folder_path, average_type='raw', CP=1005, R=287, TREF=288.15, PREF=101300, inviscid=False):
        """
        Read the processed dataset stored in folder_path location obtained by the Paraview Macro, for The Marble Extraction procedure.
        Average type distinguish the type of average used, raw for standard circumferential.
        Inviscid=True sets the viscous stresses to zero, leading to inviscid force extraction.
        """
        available_avg_types = ['raw', 'density', 'axialMomentum']
        self.avg_type = average_type
        if self.avg_type not in available_avg_types:
            raise ValueError('Not valid average type')
        print('Weighted average type: %s' % self.avg_type)

        def extract_grid_location(file_name):
            print('Elaborating Filename: ' + file_name)
            file_name = file_name.strip('spline_data_')
            file_name = file_name.strip('.csv')
            file_name = file_name.split('_')
            nz = int(file_name[0])
            nr = int(file_name[1])
            return nz, nr
       
        data_dir = folder_path
        files = [f for f in os.listdir(data_dir) if '.csv' in f]
        files = sorted(files)

        # give the name of the fields to average
        fields = ['Density', 'Energy', 'Mach', 'Eddy_Viscosity', 'Pressure', 'Temperature', 'Grid_Velocity_Tangential',
                    'Velocity_Radial', 'Velocity_Tangential', 'Velocity_Tangential_Relative', 'Velocity_Axial', 'Entropy']
        
        nz, nr = extract_grid_location(files[-1])
        field_grids = {}
        for field_name in fields:
            field_grids[field_name] = np.zeros((nz+1, nr+1))
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

            # Compute additional fields that will be circumferentially averaged 
            data_dict['Velocity_Radial'] = data_dict['Velocity_0']*cos(theta)+data_dict['Velocity_1']*sin(theta)

            data_dict['Velocity_Tangential'] = -data_dict['Velocity_0']*sin(theta)+data_dict['Velocity_1']*cos(theta)

            data_dict['Grid_Velocity_Tangential'] = -data_dict['Grid_Velocity_0']*sin(theta)+data_dict['Grid_Velocity_1']*cos(theta)

            data_dict['Velocity_Tangential_Relative'] = data_dict['Velocity_Tangential']-data_dict['Grid_Velocity_Tangential'] 
            
            data_dict['Velocity_Axial'] = data_dict['Velocity_2']

            data_dict['Entropy'] = CP*np.log(data_dict['Temperature']/TREF)-R*np.log(data_dict['Pressure']/PREF)

            for field in fields:
                f = data_dict[field].copy()
                if self.avg_type.lower() == 'raw':
                    field_grids[field][stream_id, span_id] = np.sum(f) / len(f)
                elif self.avg_type.lower() == 'density':
                    field_grids[field][stream_id, span_id] = np.sum(f * data_dict['Density']) / np.sum(data_dict['Density'])
                elif self.avg_type.lower() == 'axialMomentum':
                    field_grids[field][stream_id, span_id] = np.sum(f * data_dict['Momentum_2']) / np.sum(data_dict['Momentum_2'])

        self.meridional_fields = field_grids
        self.meridional_fields['Z'] = z_grid
        self.meridional_fields['R'] = r_grid


    def contour_meridional_fields(self):
        """
        contour of the fields stored in meridional fields
        """
        output_folder = self.config.get_pictures_folder_path()
        os.makedirs(output_folder, exist_ok=True)

        z = self.meridional_fields['Z']
        r = self.meridional_fields['R']

        for key, value in self.meridional_fields.items():
            if key!='Z' and key!='R':
                try:
                    self.contour_template(z, r, value, key)
                    plt.savefig(output_folder + '/%s_%sAvg.pdf' % (key, self.avg_type), bbox_inches='tight')
                except:
                    plt.close()
                    pass


    def compute_additional_meridional_fields(self, CP=1005, R=287, TREF=288.15, PREF=101300):
        """
        Compute additional Meridional Fields.
        WARNING: All these fields are computed using the circumferential averages through some function. For this reason the suffix postAVG is used. 
        It's probably not mathematically correct, since a function of the average is not the same of the average of a function. If that produces error, these fields should be computed
        during the reading process
        """
        self.meridional_fields['Velocity_Meridional_postAVG'] = np.sqrt(self.meridional_fields['Velocity_Axial']**2+
                                                                self.meridional_fields['Velocity_Radial']**2)
        
        self.meridional_fields['Velocity_Tangential_Relative_postAVG'] = self.meridional_fields['Velocity_Tangential'] - self.config.get_omega_shaft()*self.meridional_fields['R']
        
        self.meridional_fields['Absolute_Flow_Angle_postAVG'] = np.arctan2(self.meridional_fields['Velocity_Tangential'],
                                                                 self.meridional_fields['Velocity_Axial'])
        
        self.meridional_fields['Relative_Flow_Angle_postAVG'] = np.arctan2(self.meridional_fields['Velocity_Tangential_Relative'],
                                                                 self.meridional_fields['Velocity_Axial'])
        
        self.meridional_fields['Velocity_Magnitude_postAVG'] = np.sqrt(self.meridional_fields['Velocity_Radial']**2 +
                                                               self.meridional_fields['Velocity_Tangential']**2 +
                                                               self.meridional_fields['Velocity_Axial']**2)
        
        self.meridional_fields['Velocity_Magnitude_Relative_postAVG'] = np.sqrt(self.meridional_fields['Velocity_Radial'] ** 2 +
                                                                        self.meridional_fields['Velocity_Tangential_Relative'] ** 2 +
                                                                        self.meridional_fields['Velocity_Axial'] ** 2)
        
        self.meridional_fields['Mach_Relative_postAVG'] = self.meridional_fields['Velocity_Magnitude_Relative_postAVG']/np.sqrt(
            1.4*self.meridional_fields['Pressure']/self.meridional_fields['Density'])

        self.meridional_fields['Entropy_postAVG'] = CP*np.log(self.meridional_fields['Temperature']/TREF)-R*np.log(self.meridional_fields['Pressure']/PREF)


    def extract_body_forces(self, f_turn_method='Thermodynamic', f_loss_method='Thermodynamic'):
        """
        From the meridional fields, extract the body forces
        :param f_turn_method: method selected to extract the turning component of the body force
        :param f_loss_method: method selected to extract the loss component of the body force
        """

        # loss component
        self.meridional_fields['Force_Loss'] = np.zeros_like(self.meridional_fields['Z'])
        self.meridional_fields['Force_Tangential'] = np.zeros_like(self.meridional_fields['Z'])

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


    def compute_streamwise_meridional_projection_length(self, z1, r1, theta1, z2, r2, theta2):
        """
        Given the coordinates defining the two sides of the blade, compute the associated curvilinear abscissa of their projection
        on the meridional plane (z,r)
        """
        blade_type = self.config.get_blade_outlet_type()[self.iblade]
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
        elif inlet_line==2 and outlet_line==1:
            zmeridional, rmeridional = z2[id_LE:], r2[id_LE:]
        elif inlet_line==1 and outlet_line==1:
            zmeridional, rmeridional = z1[id_LE:id_TE+1], r1[id_LE:id_TE+1]
        elif inlet_line==2 and outlet_line==2:
            zmeridional, rmeridional = z2[id_LE:id_TE+1], r2[id_LE:id_TE+1]
        else:
            raise ValueError('Problem')

        # spline of the projection on the (z,r) plane, and associated curvilinear abscissa length
        zs, rs = compute_2dSpline_curve(zmeridional, rmeridional, 1000)
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

        if self.config.get_visual_debug():
            plt.figure()
            plt.plot(z1, r1, 'o', mfc='none', label='blade side 1')
            plt.plot(z2, r2, '^', mfc='none', label='blade side 2')
            plt.plot(zs, rs, mfc='none', label='projection spline')
            plt.xlabel(r'$z$')
            plt.ylabel(r'$r$')
            plt.legend()

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
    
    
    def compute_lean_angle(self):
        normalPlanar = np.sqrt(self.n_camber_t**2+self.n_camber_z**2)
        self.lean_angle = np.arctan2(self.n_camber_r, normalPlanar)
        self.contour_template(self.z_grid, self.r_grid, self.lean_angle*180/np.pi, r'$\lambda$ [deg]')
    

    def compute_marble_body_force(self):
        """
        Compute the body force density, using the marble thermodynamic approach based on the circumferentially averaged flow field
        """
        ni,nj = self.meridional_fields['Z'].shape
        
        
        self.meridional_fields['Force_Viscous'] = self.compute_marble_loss_force()
        self.meridional_fields['Force_Tangential'] = self.compute_marble_ftheta()

        fp_versor = np.zeros((ni,nj,3))
        for i in range(ni):
            for j in range(nj):
                w = np.array([self.meridional_fields['Velocity_Radial'][i,j],
                              self.meridional_fields['Velocity_Tangential_Relative'][i,j],
                              self.meridional_fields['Velocity_Axial'][i,j]])
                
                fp_versor[i,j,:] = -w/np.linalg.norm(w)

        self.meridional_fields['Force_Viscous_Radial'] = self.meridional_fields['Force_Viscous']*fp_versor[:,:,0]
        self.meridional_fields['Force_Viscous_Tangential'] = self.meridional_fields['Force_Viscous']*fp_versor[:,:,1]
        self.meridional_fields['Force_Viscous_Axial'] = self.meridional_fields['Force_Viscous']*fp_versor[:,:,2]
        
        self.meridional_fields['Force_Inviscid_Tangential'] = self.meridional_fields['Force_Tangential']-self.meridional_fields['Force_Viscous_Tangential']
        self.meridional_fields['Force_Inviscid_Radial'] = np.abs(self.meridional_fields['Force_Inviscid_Tangential'])*np.tan(self.lean_angle)

        self.meridional_fields['Force_Axial'] = self.meridional_fields['Force_Viscous_Axial']-(self.meridional_fields['Force_Tangential']-self.meridional_fields['Force_Viscous_Tangential'])*self.meridional_fields['Velocity_Tangential_Relative']/self.meridional_fields['Velocity_Axial']
        self.meridional_fields['Force_Inviscid_Axial'] = self.meridional_fields['Force_Axial']-self.meridional_fields['Force_Viscous_Axial']

        self.meridional_fields['Force_Inviscid'] = np.sqrt(self.meridional_fields['Force_Inviscid_Axial']**2+
                                                           self.meridional_fields['Force_Inviscid_Radial']**2+
                                                           self.meridional_fields['Force_Inviscid_Tangential']**2)
        
        self.meridional_fields['Force_Radial'] = self.meridional_fields['Force_Viscous_Radial']+self.meridional_fields['Force_Inviscid_Radial']


    def clip_contour(self, fsource, vmin, vmax):
        """
        clip a 2D array between vmin and vmax
        """
        f = fsource.copy()
        ni,nj = f.shape
        for i in range(ni):
            for j in range(nj):
                if f[i,j]<vmin:
                    f[i,j] = vmin
                elif f[i,j]>= vmax:
                    f[i,j]=vmax
                else:
                    pass
        return f


    def compute_marble_ftheta(self, method='local'):
        """
        Compute the global tangential force.
        Method <distributed> spread the gradient of angular momentum linearly from inlet to outlet
        Method <local> uses local gradients of the angular momentum
        """
        self.meridional_fields['Velocity_Meridional'] = np.sqrt(self.meridional_fields['Velocity_Axial']**2 + self.meridional_fields['Velocity_Radial']**2)
        um = self.meridional_fields['Velocity_Meridional'].copy()
        ut = self.meridional_fields['Velocity_Tangential'].copy()
        r = self.meridional_fields['R'].copy()
        z = self.meridional_fields['Z'].copy()
        ftheta = np.zeros_like(um)
        
        if method=='local':
            drut_dz, drut_dr = compute_gradient_least_square(self.meridional_fields['Z'], self.meridional_fields['R'], r*ut)
            self.contour_template(z, r, r, r'$r$')
            self.contour_template(z, r, ut, r'$u_{\theta}$')
            self.contour_template(z, r, r*ut, r'$r u_{\theta}$')
            self.contour_template(z, r, drut_dz, r'$\partial(r u_{\theta}) / \partial z$')
            self.contour_template(z, r, drut_dr, r'$\partial(r u_{\theta}) / \partial r$')
            ftheta = 1/r*(drut_dz*self.meridional_fields['Velocity_Axial']+drut_dr*self.meridional_fields['Velocity_Radial'])
        
        elif method=='distributed':
            for j in range(um.shape[1]):
                deltaF = (r[-1,j]*ut[-1,j])-(r[0,j]*ut[0,j])
                deltaM = self.streamline_length[-1,j]-self.streamline_length[0,j]
                ftheta[:,j] = um[:,j]/r[:,j]*deltaF/deltaM
        
        else:
            raise ValueError('Method unknown')
        
        self.contour_template(z, r, ftheta, r'$f_{\theta}$', vmax=0)
        return ftheta
    

    def compute_marble_loss_force(self):
        floss = np.zeros_like(self.meridional_fields['R'])

        T = self.meridional_fields['Temperature'].copy()
        W = np.sqrt(self.meridional_fields['Velocity_Axial']**2 + self.meridional_fields['Velocity_Radial']**2 + self.meridional_fields['Velocity_Tangential_Relative']**2)
        Um = np.sqrt(self.meridional_fields['Velocity_Axial']**2 + self.meridional_fields['Velocity_Radial']**2)

        for j in range(T.shape[1]):
            deltaS = self.meridional_fields['Entropy'][-1,j]-self.meridional_fields['Entropy'][0,j]
            deltaM = self.streamline_length[-1,j]-self.streamline_length[0,j]

            floss[:,j] = T[:,j]*Um[:,j]/W[:,j]*deltaS/deltaM
        return floss 
    

    def contour_template(self, z: np.ndarray, r: np.ndarray, f: np.ndarray, name: str, vmin=None, vmax=None, save_filename=None):
        """
        Template function for 2D contours
        """
        folder = self.config.get_pictures_folder_path()
        os.makedirs(folder, exist_ok=True)

        if vmin == None:
            minval = np.min(f)
        else:
            minval = vmin
        if vmax == None:
            maxval = np.max(f)
        else:
            maxval = vmax
        
        if minval==maxval:
            maxval += 1e-12
        
        levels = np.linspace(minval, maxval, N_levels)

        fig, ax = plt.subplots()
        contour = ax.contourf(z, r, f, levels=levels, cmap=color_map, vmin = minval, vmax = maxval)
        cbar = fig.colorbar(contour)
        contour = ax.contour(z, r, f, levels=levels, colors='black', vmin = minval, vmax = maxval, linewidths=0.1)
        plt.title(name)
        ax.set_aspect('equal', adjustable='box')
        if save_filename is not None:
            plt.savefig(folder + '/' + save_filename + '_%sAvg.pdf' % (self.avg_type), bbox_inches='tight')
    

    def interpolate_body_force_data(self, filepath, fields_name):
        """
        Interpolate the body force components stored in filepath onto the blade grid
        """
        with open(filepath, 'rb') as file:
            data = pickle.load(file)

        z_data = data['Z']
        r_data = data['R']
        return_fields = []
        for field in fields_name:
            self.contour_template(z_data, r_data, data[field], field+'_reference')
            f = griddata_interpolation_with_nearest_filler(z_data, r_data, data[field], self.z_grid, self.r_grid)
            self.contour_template(z_data, r_data, data[field], field+'_interpolated')
            return_fields.append(f.copy())

        return return_fields
    

    def cut_blade_tip(self, clearance_meters):
        """
        Remove every force component in the gap from the shroud described by clearance_meters
        """
        gap = clearance_meters
        self.compute_spanline_length()
        ni,nj = self.meridional_fields['R'].shape

        for i in range(ni):
            for j in range(nj):
                distance = self.spanline_length[i,-1]-self.spanline_length[i,j]
                if distance<=gap:
                    self.meridional_fields['Force_Axial'][i,j] = 0
                    self.meridional_fields['Force_Tangential'][i,j] = 0
                    self.meridional_fields['Force_Radial'][i,j] = 0
    

    def compute_kiwada_body_force(self):
        """
        Compute the force using the relations of Kiwada, for the global force, already decomposed in in its components.
        """
        R = self.meridional_fields['R'].copy()
        Z = self.meridional_fields['Z'].copy()
        B = self.blockage.copy()
        dbdz, dbdr = compute_gradient_least_square(Z, R, self.blockage)

        self.contour_template(Z, R, B, r'$b$')
        self.contour_template(Z, R, dbdz, r'$\partial_z b$')
        self.contour_template(Z, R, dbdr, r'$\partial_r b$')

        # axial equation
        dA1dz = compute_gradient_least_square(Z, R, B*self.meridional_fields['A1'])[0]
        dA2dr = compute_gradient_least_square(Z, R, B*R*self.meridional_fields['A2'])[1]
        self.meridional_fields['Force_Axial'] = 1/B*dA1dz+1/B/R*dA2dr
        # self.contour_template(Z[2:-2,2:-2], R[2:-2,2:-2], self.meridional_fields['Force_Axial'][2:-2,2:-2], name='f_axial', vmin=0)

        dR1dz = compute_gradient_least_square(Z, R, B*self.meridional_fields['A2'])[0]
        dR2dr = compute_gradient_least_square(Z,R, B*R*self.meridional_fields['R2'])[1]
        self.meridional_fields['Force_Radial'] = 1/B*dR1dz+1/B/R*dR2dr-self.meridional_fields['R3']/R
        # self.contour_template(Z[2:-2,2:-2], R[2:-2,2:-2], self.meridional_fields['Force_Radial'][2:-2,2:-2], name='f_radial')

        dT1dz = compute_gradient_least_square(Z, R, B*self.meridional_fields['T1'])[0]
        dT2dr = compute_gradient_least_square(Z, R, B*R*self.meridional_fields['T2'])[1]
        self.meridional_fields['Force_Tangential'] = 1/B*dT1dz + 1/B/R*dT2dr + self.meridional_fields['T3']/R
        # self.contour_template(Z[2:-2,2:-2], R[2:-2,2:-2], self.meridional_fields['Force_Tangential'][2:-2,2:-2], name='f_tangential', vmax=0)

        fmag = np.sqrt(self.meridional_fields['Force_Radial']**2+self.meridional_fields['Force_Tangential']**2+self.meridional_fields['Force_Axial']**2)
        # self.contour_template(Z[2:-2,2:-2], R[2:-2,2:-2], fmag[2:-2,2:-2], name='f_magnitude')

        ni,nj = R.shape
        self.meridional_fields['Force_Viscous'] = np.zeros((ni,nj))
        for i in range(ni):
            for j in range(nj):
                w = np.array([self.meridional_fields['Velocity_Radial'][i,j],
                              self.meridional_fields['Velocity_Tangential_Relative'][i,j],
                              self.meridional_fields['Velocity_Axial'][i,j]])
                fg = np.array([self.meridional_fields['Force_Radial'][i,j],
                               self.meridional_fields['Force_Tangential'][i,j],
                               self.meridional_fields['Force_Axial'][i,j]])
                
                fp_vers = -w/np.linalg.norm(w)
                self.meridional_fields['Force_Viscous'][i,j] = np.dot(fg, fp_vers)
        self.meridional_fields['Force_Inviscid'] = np.sqrt(fmag**2-self.meridional_fields['Force_Viscous']**2)
    

    def cure_hub(self, span_extent, f):
        """
        For f defined on the meridional grid, cure the field within hub and the span extent. Cure means copying from the first acceptable value outside of the span extent.
        """
        # self.contour_template(self.meridional_fields['Z'], self.meridional_fields['R'], f, 'f_before')
        gap = span_extent
        self.compute_spanline_length(normalize=True)
        ni,nj = f.shape
        for i in range(ni):
            j = 0
            while self.spanline_length[i,j]<span_extent:
                j_id = j
                j += 1
            f[i,0:j_id] = f[i,j_id]
        # self.contour_template(self.meridional_fields['Z'], self.meridional_fields['R'], f, 'f_after')
    
    def cure_shroud(self, span_extent, f):
        """
        For f defined on the meridional grid, cure the field within shroud and the span extent. Cure means copying from the first acceptable value outside of the span extent.
        """
        # self.contour_template(self.meridional_fields['Z'], self.meridional_fields['R'], f, 'f_before')
        self.compute_spanline_length(normalize=True)
        ni,nj = f.shape
        for i in range(ni):
            j = nj-1
            while self.spanline_length[i,j]>1-span_extent:
                j_id = j
                j -= 1
            f[i,j_id:] = f[i,j_id]
        # self.contour_template(self.meridional_fields['Z'], self.meridional_fields['R'], f, 'f_after')
    
    
    def extrapolate_camber(self):
        """
        Extrapolate the blade normal camber over the last portion close to leading and trailing edge
        """
        coefficient = self.config.get_blade_edges_extrapolation_coefficient()[self.iblade]
        normalizedStreamLength = np.zeros_like(self.streamline_length)
        for j in range(normalizedStreamLength.shape[1]):
            normalizedStreamLength[:,j] = self.streamline_length[:,j]/self.streamline_length[-1,j]
        
        def LinearExtrapolation(x, y, xnew):
            ynew = np.zeros_like(xnew)
            yprime = np.gradient(y, x, edge_order=2)
            if xnew[0]<=x[0]:
                # left extrapolation
                for i in range(len(xnew)):
                    ynew[i] = y[0]+yprime[0]*(xnew[i]-x[0])
            else:
                for i in range(len(xnew)):
                    ynew[i] = y[-1]+yprime[0]*(xnew[i]-x[-1])
            return ynew
        
        def ExtrapolateDataStream(f):
            ni,nj = f.shape
            for j in range(nj):
                leadingPoints = np.where(normalizedStreamLength[:,j]<=coefficient)
                trailingPoints = np.where(normalizedStreamLength[:,j]>=1-coefficient)
                internalPoints = np.where((normalizedStreamLength[:, j] > coefficient) & (normalizedStreamLength[:, j] < 1 - coefficient))
                
                f[leadingPoints,j] = LinearExtrapolation(normalizedStreamLength[internalPoints,j].flatten(), f[internalPoints,j].flatten(), normalizedStreamLength[leadingPoints,j].flatten())
                f[trailingPoints,j] = LinearExtrapolation(normalizedStreamLength[internalPoints,j].flatten(), f[internalPoints,j].flatten(), normalizedStreamLength[trailingPoints,j].flatten())                
            return f
        
        self.n_camber_r = ExtrapolateDataStream(self.n_camber_r)
        self.n_camber_t = ExtrapolateDataStream(self.n_camber_t)
        self.n_camber_z = ExtrapolateDataStream(self.n_camber_z)
        
        normalizedSpanLength = np.zeros_like(self.spanline_length)
        for i in range(normalizedSpanLength.shape[0]):
            normalizedSpanLength[i,:] = self.spanline_length[i,:]/self.spanline_length[i,-1]
        
        def ExtrapolateDataSpan(f):
            ni,nj = f.shape
            for i in range(ni):
                hubPoints = np.where(normalizedSpanLength[i,:]<=coefficient)
                shroudPoints = np.where(normalizedSpanLength[i,:]>=1-coefficient)
                internalPoints = np.where((normalizedSpanLength[i,:] > coefficient) & (normalizedSpanLength[i,:] < 1 - coefficient))
                
                f[i, hubPoints] = LinearExtrapolation(normalizedSpanLength[i,internalPoints].flatten(), f[i,internalPoints].flatten(), normalizedSpanLength[i,hubPoints].flatten())
                f[i, shroudPoints] = LinearExtrapolation(normalizedSpanLength[i,internalPoints].flatten(), f[i,internalPoints].flatten(), normalizedSpanLength[i,shroudPoints].flatten())
            return f
        
        self.n_camber_r = ExtrapolateDataSpan(self.n_camber_r)
        self.n_camber_t = ExtrapolateDataSpan(self.n_camber_t)
        self.n_camber_z = ExtrapolateDataSpan(self.n_camber_z)
        
        # renormalize the camber normal after the extrapolation
        self.n_camber_r = self.n_camber_r/np.sqrt(self.n_camber_r**2+self.n_camber_t**2+self.n_camber_z**2)
        self.n_camber_t = self.n_camber_t/np.sqrt(self.n_camber_r**2+self.n_camber_t**2+self.n_camber_z**2)
        self.n_camber_z = self.n_camber_z/np.sqrt(self.n_camber_r**2+self.n_camber_t**2+self.n_camber_z**2)
        
        
        
                    


                
                



























