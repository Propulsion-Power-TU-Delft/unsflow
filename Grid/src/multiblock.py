#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:29:29 2023
@author: F. Neri, TU Delft
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from utils.styles import *
from .functions import cluster_sample_u, elliptic_grid_generation, transfinite_grid_generation, contour_template, compute_meridional_streamwise_coordinates, compute_meridional_spanwise_coordinates
from .curve import Curve
from sun.src.general_functions import print_banner_begin, print_banner_end
from .area_element import AreaElement
from scipy.interpolate import CubicSpline
import pickle
import os
from scipy.ndimage import gaussian_filter
from grid.src.functions import enlarge_domain_array
from grid.src.config import Config


class MultiBlock:
    """
    this class contains a multiblock meridional grid
    """

    def __init__(self, config, blocks, blades):
        """
        Construct the Block object, storing all the data and methods for the meridional grid. There is no need to provide the
        dimensions and scaling factor of the cordinates since they are already used in the hub and shroud curve objects.
        :param config: configuration file
        :param nstream: number of grid points along the streamwise direction
        :param nspan: number of grid points along the spanwise direction
        """
        self.config = config
        self.blocks = blocks
        self.blades = blades

    def assemble_grid(self):
        """
        Put together all the fields of the blocks needed for later
        """
        self.fix_theta_camber_grids()
        
        # Flag: if True, include the last point (slice is [:]), else exclude the last point (slice is [:-1])
        numberBlocks = len(self.blocks)
        if numberBlocks == 1:
            row_slice = slice(None)
        else:
            row_slice = slice(None, -1)

        self.z_grid_cg = self.blocks[0].z_grid_cg[row_slice, :]
        self.r_grid_cg = self.blocks[0].r_grid_cg[row_slice, :]
        
        self.bfmFields = {}
        for key in self.blocks[0].bfmFields.keys():
            self.bfmFields[key] = self.blocks[0].bfmFields[key][row_slice, :]

        # if self.config.get_body_force_calibration_method().lower() == 'lift/drag':
        #     self.BFCalibrationCoefficients = self.blocks[0].BFCalibrationCoefficients.copy()
        #     for key in self.BFCalibrationCoefficients:
        #         if key.lower() != 'model':
        #             self.BFCalibrationCoefficients[key] = self.BFCalibrationCoefficients[key][row_slice, :]
        
        # if self.config.get_body_force_calibration_method().lower() == 'inference':
        #     self.inferenceCoefficients = self.blocks[0].inferenceCoefficients.copy()
        #     for key in self.inferenceCoefficients:
        #         self.inferenceCoefficients[key] = self.inferenceCoefficients[key][row_slice, :]
        
        bladeFlag = 1
        for iBlock, block in enumerate(self.blocks[1:]):
                           
            if bladeFlag > 0: # this is a blade block, or there is only one block
                streamSlice = slice(None)
            elif iBlock == numberBlocks-1: # is the last block
                streamSlice = slice(1, None)
            else:
                streamSlice = slice(1, -1)
            
            self.z_grid_cg = np.concatenate((self.z_grid_cg, block.z_grid_cg[streamSlice, :]), axis=0)
            self.r_grid_cg = np.concatenate((self.r_grid_cg, block.r_grid_cg[streamSlice, :]), axis=0)
            
            for key in self.bfmFields.keys():
                self.bfmFields[key] = np.concatenate((self.bfmFields[key], block.bfmFields[key][streamSlice, :]), axis=0)
            
            # self.blockage = np.concatenate((self.blockage, block.blockage[streamSlice, :]), axis=0)
            # self.rpm = np.concatenate((self.rpm, block.rpm[streamSlice, :]), axis=0)
            # self.streamline_length = np.concatenate((self.streamline_length, block.streamline_length[streamSlice, :]), axis=0)
            # self.normal_camber['Axial'] = np.concatenate((self.normal_camber['Axial'], block.normal_camber['Axial'][streamSlice, :]), axis=0)
            # self.normal_camber['Radial'] = np.concatenate((self.normal_camber['Radial'], block.normal_camber['Radial'][streamSlice, :]), axis=0)
            # self.normal_camber['Tangential'] = np.concatenate((self.normal_camber['Tangential'], block.normal_camber['Tangential'][streamSlice, :]), axis=0)
            # self.force_axial = np.concatenate((self.force_axial, block.bodyForce['Force_Axial'][streamSlice, :]), axis=0)
            # self.force_radial = np.concatenate((self.force_radial, block.bodyForce['Force_Radial'][streamSlice, :]), axis=0)
            # self.force_tangential = np.concatenate((self.force_tangential, block.bodyForce['Force_Tangential'][streamSlice, :]), axis=0)
            # self.angular_momentum_derivative = np.concatenate((self.angular_momentum_derivative, block.bodyForce['AngularMomentumDerivative'][streamSlice, :]), axis=0)
            # self.entropy_derivative = np.concatenate((self.entropy_derivative, block.bodyForce['EntropyDerivative'][streamSlice, :]), axis=0)
            # self.nBlades = np.concatenate((self.nBlades, block.nBlades[streamSlice, :]), axis=0)
            # self.bladePresent = np.concatenate((self.bladePresent, block.bladePresent[streamSlice, :]), axis=0)
            # self.theta_camber = np.concatenate((self.theta_camber, block.theta_camber[streamSlice, :]), axis=0)
            
            # if self.config.get_body_force_calibration_method().lower() == 'lift/drag':
            #     for key in self.BFCalibrationCoefficients.keys():
            #         if key.lower()!='model':
            #             self.BFCalibrationCoefficients[key] = np.concatenate((self.BFCalibrationCoefficients[key], block.BFCalibrationCoefficients[key][streamSlice, :]), axis=0)
            
            # if self.config.get_body_force_calibration_method().lower() == 'inference':
            #     for key in self.inferenceCoefficients.keys():
            #         self.inferenceCoefficients[key] = np.concatenate((self.inferenceCoefficients[key], block.inferenceCoefficients[key][streamSlice, :]), axis=0)
            
            bladeFlag *= -1 # the next block will be unbladed
            
        # self.theta_camber = gaussian_filter(self.theta_camber, sigma=3)
        self.spanline_length_normalized = compute_meridional_spanwise_coordinates(self.z_grid_cg, self.r_grid_cg, normalize=True)
        self.streamline_length_normalized = compute_meridional_streamwise_coordinates(self.z_grid_cg, self.r_grid_cg, normalize=True)
        self.spanline_length = compute_meridional_spanwise_coordinates(self.z_grid_cg, self.r_grid_cg, normalize=False)
        self.streamline_length = compute_meridional_streamwise_coordinates(self.z_grid_cg, self.r_grid_cg, normalize=False)
        self.nstream = self.z_grid_cg.shape[0]
        self.nspan = self.z_grid_cg.shape[1]
        self.z_grid_points = self.z_grid_cg
        self.r_grid_points = self.r_grid_cg

    def remove_inlet_grid_points(self, Ntrim):
        """
        Remove the first Ntrim streamwise stations from the dual grids before writing the grid file
        """
        self.z_grid_dual = self.z_grid_dual[Ntrim:, :]
        self.r_grid_dual = self.r_grid_dual[Ntrim:, :]

    def remove_outlet_grid_points(self, Ntrim):
        """
        Remove the first Ntrim streamwise stations from the dual grids before writing the grid file
        """
        max_id = self.z_grid_dual.shape[0]-Ntrim
        self.z_grid_dual = self.z_grid_dual[0:max_id, :]
        self.r_grid_dual = self.r_grid_dual[0:max_id, :]


    def plot_full_grid(self, save_filename=None, primary_grid=True, primary_grid_points=False, secondary_grid=False,
                       secondary_grid_points=False, hub_shroud=False, outline=False, grid_centers=False, ticks=False,
                       save_foldername=None):
        """
        Plot the obtained grid.
        :param save_filename: specify path of the figures to be saved (if you want to save).
        :param primary_grid: if True plots the primary grid lines
        :param primary_grid_points: if True plots the primary grid points
        :param secondary_grid: if True plots the secondary grid lines
        :param secondary_grid_points: if True plots the secondary grid points
        :param hub_shroud: if True plots hub and shroud highlighted
        :param outline: if True plots the highlighted outline of the domain
        :param grid_centers: if True plots the grid centers
        :param ticks: if True allows ticks to be shown
        :param save_foldername: folder name to save pictures in
        """


        plt.figure()

        # hub and shroud plot
        if hub_shroud:
            plt.plot(self.hub.z_spline, self.hub.r_spline, lw=light_line_width, c='black')
            plt.plot(self.shroud.z_spline, self.shroud.r_spline, lw=light_line_width, c='black')

        # primary grid
        if primary_grid:
            for istream in range(0, self.nstream):
                plt.plot(self.z_grid_points[istream, :], self.r_grid_points[istream, :], lw=light_line_width, c='black')
            for ispan in range(0, self.nspan):
                plt.plot(self.z_grid_points[:, ispan], self.r_grid_points[:, ispan], lw=light_line_width, c='black')
        elif outline:
            plt.plot(self.z_grid_points[0, :], self.r_grid_points[0, :], lw=line_width, label='leading edge')
            plt.plot(self.z_grid_points[-1, :], self.r_grid_points[-1, :], lw=line_width, label='trailing edge')
            plt.plot(self.z_grid_points[:, 0], self.r_grid_points[:, 0], lw=line_width, label='hub')
            plt.plot(self.z_grid_points[:, -1], self.r_grid_points[:, -1], lw=line_width, label='shroud')

        # primary grid points
        if primary_grid_points:
            plt.scatter(self.z_grid_points.flatten(), self.r_grid_points.flatten(), c='black', s=scatter_point_size,
                        label='primary grid nodes')

        # secondary grid
        if secondary_grid:
            for istream in range(0, self.z_grid_dual.shape[0]):
                plt.plot(self.z_grid_dual[istream, :], self.r_grid_dual[istream, :], '--r', lw=light_line_width)
            for ispan in range(0, self.z_grid_dual.shape[1]):
                plt.plot(self.z_grid_dual[:, ispan], self.r_grid_dual[:, ispan], '--r', lw=light_line_width)

        if grid_centers:
            plt.scatter(self.z_grid_cg, self.r_grid_cg, marker='+', s=marker_size_small, c='black')

        if secondary_grid_points:
            plt.scatter(self.z_grid_dual.flatten(), self.r_grid_dual.flatten(), c='red', s=scatter_point_size,
                        label='secondary grid nodes')

        if primary_grid_points or secondary_grid_points or outline:
            plt.legend()
        plt.xlabel(r'$z \ \mathrm{[m]}$')
        plt.ylabel(r'$r \ \mathrm{[m]}$')
        plt.title(r'$(%d \times %d)$' % (self.nstream, self.nspan))

        if not ticks:
            plt.xticks([])
            plt.yticks([])  # plt.xlabel('')  # plt.ylabel('')
        
        ax = plt.gca()
        ax.set_aspect('equal')

        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_meridional_grid.pdf', bbox_inches='tight')
    

    def compute_average_dtheta(self):
        """
        Loop over all the cells on the meridional plane, finds the average length, and obtain dtheta as R_mean*dtheta=Avg_length
        in order to preserve the AR of the mesh when extruded in 3D
        """
        avg_length = 0
        for i in range(self.nstream - 1):
            for j in range(self.nspan - 1):
                avg_length += np.sqrt(np.abs(self.z_grid_points[i + 1, j] - self.z_grid_points[i, j]) * np.abs(
                    self.r_grid_points[i, j + 1] - self.r_grid_points[i, j]))
        avg_length /= (self.nstream - 1) * (self.nspan - 1)
        r_mean = (self.r_grid_points[0, self.nspan // 2] + self.r_grid_points[-1, self.nspan // 2]) / 2
        return avg_length / r_mean

    def compute_min_dtheta(self):
        """
        Loop over all the cells on the meridional plane, finds the average length, and obtain dtheta as R_mean*dtheta=Avg_length
        in order to preserve the AR of the mesh when extruded in 3D
        """
        min_length = 1e9
        for i in range(self.nstream - 1):
            for j in range(self.nspan - 1):
                tmp = np.sqrt(np.abs(self.z_grid_points[i + 1, j] - self.z_grid_points[i, j]) * np.abs(
                    self.r_grid_points[i, j + 1] - self.r_grid_points[i, j]))
                if tmp < min_length:
                    min_length = tmp
        r_mean = (self.r_grid_points[0, self.nspan // 2] + self.r_grid_points[-1, self.nspan // 2]) / 2
        return min_length / r_mean

    def compute_three_dimensional_mesh(self, config, mode='singlezone', conserve_AR=True, theta_max=1, nodes_number=2, dimensional=True):
        """
        Compute the Three-dimensional mesh X,Y,Z as 3D arrays, structured.
        :param config: config object needed to pass reference dimensions
        :param conserve_AR: if True, tries to conserve the AR choosing the correct delta-theta between each tang. node
        :param theta_max: [deg] angle of the mesh sector.
        :param nodes_number: number of nodes from zero to theta_max
        :param dimensional: if True, reconverts the coordinates to original dimensions in [m]
        :param mode: singlezone or multizone mesh pickles.
        """
        if conserve_AR:
            dtheta = self.compute_average_dtheta()
            theta_max = dtheta * (nodes_number - 1)
            theta = np.linspace(0, theta_max, nodes_number)
        else:
            theta = np.linspace(0, theta_max * np.pi / 180, nodes_number)
        self.deltatheta_periodic = theta[-1]*180/np.pi

        if mode.lower()=='singlezone':
            self.X_mesh = np.zeros((self.nstream, self.nspan, nodes_number))
            self.Y_mesh = np.zeros((self.nstream, self.nspan, nodes_number))
            self.Z_mesh = np.zeros((self.nstream, self.nspan, nodes_number))
            for i in range(self.nstream):
                for j in range(self.nspan):
                    for k in range(nodes_number):
                        self.X_mesh[i, j, k] = self.r_grid_cg[i, j] * np.cos(theta[k])
                        self.Y_mesh[i, j, k] = self.r_grid_cg[i, j] * np.sin(theta[k])
                        self.Z_mesh[i, j, k] = self.z_grid_cg[i, j]
            if dimensional:
                self.X_mesh *= config.get_reference_length()
                self.Y_mesh *= config.get_reference_length()
                self.Z_mesh *= config.get_reference_length()

        elif mode.lower()=='multizone':
            for block in self.blocks:
                block.X_mesh = np.zeros((block.nstream, block.nspan, nodes_number))
                block.Y_mesh = np.zeros_like(block.X_mesh)
                block.Z_mesh = np.zeros_like(block.X_mesh)
                for i in range(block.nstream):
                    for j in range(block.nspan):
                        for k in range(nodes_number):
                            block.X_mesh[i, j, k] = block.r_grid_cg[i, j] * np.cos(theta[k])
                            block.Y_mesh[i, j, k] = block.r_grid_cg[i, j] * np.sin(theta[k])
                            block.Z_mesh[i, j, k] = block.z_grid_cg[i, j]


    def save_mesh_pickle(self, mode='singlezone'):
        """
        Save the mesh cordinates in a pickle.
        :param mode: if single or multizone output
        :param filepath: path of the file output
        """
        if mode.lower() == 'singlezone':
            mesh = {'x': self.X_mesh, 'y': self.Y_mesh, 'z': self.Z_mesh}
            filepath = 'mesh_%02i_%02i_%02i_%.3f-deg.pickle' % (
                self.X_mesh.shape[0], self.X_mesh.shape[1], self.X_mesh.shape[2], self.deltatheta_periodic)
            with open(filepath, 'wb') as f:
                pickle.dump(mesh, f)
            print(f"Single Zone Data pickle saved to '{filepath}'")

        elif mode.lower() == 'multizone':
            for kk, block in enumerate(self.blocks):
                mesh = {'x': block.X_mesh, 'y': block.Y_mesh, 'z': block.Z_mesh}
                filepath = 'mesh_zone_%02i_%02i_%02i_%02i_%.3f-deg.pickle' % ( kk,
                    block.X_mesh.shape[0], block.X_mesh.shape[1], block.X_mesh.shape[2], self.deltatheta_periodic)
                with open(filepath, 'wb') as f:
                    pickle.dump(mesh, f)
                print(f"Multi Zone Data pickle saved to '{filepath}'")

    def export_meridional_spline(self, span, folder=None, filename=None, format_file=None):
        """
        export the meridional spline needed by Paraview to perform the blade to blade contour
        """
        if folder is None:
            folder = 'spline'
        os.makedirs(folder, exist_ok=True)

        if filename is None:
            filename = 'spline_span_%.2f' %(span)

        if format_file is None:
            format_file = 'csv'

        n_span = self.z_grid_points.shape[1]
        ispan = int(span * n_span)
        y = self.r_grid_cg[:, ispan]
        z = self.z_grid_points[:, ispan]
        x = np.zeros_like(y)

        # t = np.linspace(0, 1, len(z))
        # splinez = CubicSpline(z, t)
        # spliney = CubicSpline(y, t)
        #
        # t = np.linspace(0, 1, 200)
        # znew = splinez(t)
        # ynew = spliney(t)
        # xnew = np.zeros_like(znew)

        filepath = folder + '/' + filename + '.' + format_file
        with open(filepath, 'w') as file:
            if format_file == 'csv':
                for ii in range(len(x)):
                    file.write('%.6f,%.6f,%.6f\n' % (x[ii], y[ii], z[ii]))
                print('Meridional spline at span %.2f saved to %s' % (span, filepath))
            else:
                raise ValueError('Format not supported.')

    def compute_dual_grid(self):
        """
        compute secondary grid that can be useful for paraview processing
        """
        self.z_grid_dual = np.zeros((self.nstream + 1, self.nspan + 1))
        self.r_grid_dual = np.zeros((self.nstream + 1, self.nspan + 1))

        # internal points
        for istream in range(1, self.nstream):
            for ispan in range(1, self.nspan):
                z_mid_point = 0.25 * (
                        self.z_grid_cg[istream, ispan] + self.z_grid_cg[istream - 1, ispan] + self.z_grid_cg[istream, ispan - 1] +
                        self.z_grid_cg[istream - 1, ispan - 1])

                r_mid_point = 0.25 * (
                        self.r_grid_cg[istream, ispan] + self.r_grid_cg[istream - 1, ispan] + self.r_grid_cg[istream, ispan - 1] +
                        self.r_grid_cg[istream - 1, ispan - 1])

                self.z_grid_dual[istream, ispan] = z_mid_point
                self.r_grid_dual[istream, ispan] = r_mid_point

        # fix the vertices
        self.z_grid_dual[0, 0] = self.z_grid_cg[0, 0]
        self.r_grid_dual[0, 0] = self.r_grid_cg[0, 0]
        self.z_grid_dual[0, -1] = self.z_grid_cg[0, -1]
        self.r_grid_dual[0, -1] = self.r_grid_cg[0, -1]
        self.z_grid_dual[-1, -1] = self.z_grid_cg[-1, -1]
        self.r_grid_dual[-1, -1] = self.r_grid_cg[-1, -1]
        self.z_grid_dual[-1, 0] = self.z_grid_cg[-1, 0]
        self.r_grid_dual[-1, 0] = self.r_grid_cg[-1, 0]

        # istream = 0 border
        for istream in range(0, 1):
            for ispan in range(1, self.nspan):
                z_mid_point = 0.5 * (self.z_grid_points[istream, ispan] + self.z_grid_points[istream, ispan - 1])
                r_mid_point = 0.5 * (self.r_grid_points[istream, ispan] + self.r_grid_points[istream, ispan - 1])
                self.z_grid_dual[istream, ispan] = z_mid_point
                self.r_grid_dual[istream, ispan] = r_mid_point

        # istream = -1 border
        for istream in range(self.nstream, self.nstream + 1):
            for ispan in range(1, self.nspan):
                z_mid_point = 0.5 * (self.z_grid_points[istream - 1, ispan] + self.z_grid_points[istream - 1, ispan - 1])
                r_mid_point = 0.5 * (self.r_grid_points[istream - 1, ispan] + self.r_grid_points[istream - 1, ispan - 1])
                self.z_grid_dual[istream, ispan] = z_mid_point
                self.r_grid_dual[istream, ispan] = r_mid_point

        # ispan = 0 border
        for istream in range(1, self.nstream):
            for ispan in range(0, 1):
                z_mid_point = 0.5 * (self.z_grid_points[istream, ispan] + self.z_grid_points[istream - 1, ispan])
                r_mid_point = 0.5 * (self.r_grid_points[istream, ispan] + self.r_grid_points[istream - 1, ispan])
                self.z_grid_dual[istream, ispan] = z_mid_point
                self.r_grid_dual[istream, ispan] = r_mid_point

        # ispan = -1 border
        for istream in range(1, self.nstream):
            for ispan in range(self.nspan, self.nspan + 1):
                z_mid_point = 0.5 * (self.z_grid_points[istream, ispan - 1] + self.z_grid_points[istream - 1, ispan - 1])
                r_mid_point = 0.5 * (self.r_grid_points[istream, ispan - 1] + self.r_grid_points[istream - 1, ispan - 1])
                self.z_grid_dual[istream, ispan] = z_mid_point
                self.r_grid_dual[istream, ispan] = r_mid_point

    def write_paraview_grid_file(self, filename='meridional_grid.csv', foldername='Grid', border_factor=0.9, enlargeLoops = 3):
        """
        Write the meridional grid file requireed by Paraview Macro to run the Circumferential Average Process.
        The format of the file generated (istream, ispan, x, y, z). The points at hub and shroud are slightly moved towards the passage to avoid
        sampling in Paraview where there is no data.
        
        Parameters
        ----------------------------

        `filename`: name of the grid file to save

        `foldername`: folder name where to save the grid file

        `border_factor`: factor used to shift the border points slightly inwards such that the intersection with volumetric data is guaranteed
        """

        def move_hub_shroud_points(grid):
            grid[:,0] = grid[:,0] + (grid[:,1] - grid[:,0])*border_factor
            grid[:,-1] = grid[:,-1] + (grid[:,-2] - grid[:,-1])*border_factor
            return grid
        zgrid = move_hub_shroud_points(self.z_grid_points.copy())
        rgrid = move_hub_shroud_points(self.r_grid_points.copy())
        
        niTot, njTot = zgrid.shape
        grid_portion = self.config.get_meridional_grid_portion()
        if grid_portion=='full' or grid_portion=='all':
            zportion, rportion = zgrid[:,:], rgrid[:,:]
        else:
            iStart = grid_portion[0]
            iEnd = grid_portion[-1]+1
            if iEnd>niTot:
                iEnd = niTot
            nStations = iEnd - iStart
            zportion, rportion = np.zeros((nStations, njTot)), np.zeros((nStations, njTot))
            zportion[:,:] = zgrid[iStart:iEnd,:]
            rportion[:,:] = rgrid[iStart:iEnd,:]

        x = rportion
        y = np.zeros_like(x)
        z = zportion
        ni, nj = zportion.shape
        os.makedirs(foldername, exist_ok=True)
        with open(foldername + '/' + filename, 'w') as file:
            for istream in range(ni):
                for ispan in range(nj):
                    file.write('%i,%i,%.9f,%.9f,%.9f\n'
                               %(istream, ispan,
                                 x[istream, ispan], 
                                 y[istream, ispan], 
                                 z[istream, ispan]))
        print('Written meridional grid csv file to: %s' %(foldername + '/' + filename))
        
        
        plt.figure()
        for i in range(niTot):
            plt.plot(self.z_grid_points[i, :], self.r_grid_points[i, :], lw=light_line_width, c='black')
        for j in range(njTot):
            plt.plot(self.z_grid_points[:, j], self.r_grid_points[:, j], lw=light_line_width, c='black')
        plt.scatter(zportion, rportion, s=5, c='red')
        plt.xlabel(r'$z \ \mathrm{[m]}$')
        plt.ylabel(r'$r \ \mathrm{[m]}$')
        plt.gca().set_aspect('equal')
        plt.title('Exported meridional grid')
    
    def write_spanwise_splines(self, foldername='Grid'):
        """
        Write the spanwise splines file required by Paraview Macro to run the radial profiles needee upstream and downstream of the blades
        """
        self.streamline_length = compute_meridional_streamwise_coordinates(self.z_grid_cg, self.r_grid_cg)
        
        os.makedirs(foldername, exist_ok=True)
        offsetGridLines = self.config.get_offset_blade_grid_lines()
        
        nBlades = len(self.blades)
        nBlocks = len(self.blocks)
        
        if nBlocks == 1:
            return
        
        nstream = 0
        for iblade in range(nBlades):
            iBlock = iblade * 2 + 1  
            
            nstreamInitial = self.blocks[iBlock-1].nstream
            nstreamFinal = nstreamInitial + self.blocks[iBlock].nstream-1
            
            # zcoordUp = self.blocks[iBlock-1].z_grid_cg[-1-offsetGridLines,:]
            # rcoordUp = self.blocks[iBlock-1].r_grid_cg[-1-offsetGridLines,:]
            
            zcoordUp = self.z_grid_cg[nstreamInitial-1-offsetGridLines,:]
            rcoordUp = self.r_grid_cg[nstreamInitial-1-offsetGridLines,:]
            stwLenUp = self.streamline_length[nstreamInitial-1-offsetGridLines,:]
            
            with open(foldername + '/spanwise_spline_inlet_blade_%i.csv' % iblade, 'w') as f:
                for i in range(len(zcoordUp)):
                    f.write('%.9f,%.9f\n' % (zcoordUp[i], rcoordUp[i]))
            
            # zcoordDown = self.blocks[iBlock+1].z_grid_cg[offsetGridLines,:]
            # rcoordDown = self.blocks[iBlock+1].r_grid_cg[offsetGridLines,:]
            
            zcoordDown = self.z_grid_cg[nstreamFinal-1+offsetGridLines,:]
            rcoordDown = self.r_grid_cg[nstreamFinal-1+offsetGridLines,:]
            stwLenDown = self.streamline_length[nstreamFinal-1+offsetGridLines,:]
            
            ni = 1
            nj = len(zcoordDown)
            
            filename = 'spanwise_spline_inlet_blade_%i.csv' % iblade
            with open(foldername + '/' + filename, 'w') as file:
                file.write('indexStream,indexSpan,coordX,coordY,coordZ,streamlineLength\n')
                for istream in range(ni):
                    for ispan in range(nj):
                        file.write('%i,%i,%.9f,%.9f,%.9f,%.9f\n'
                                %(istream, ispan,
                                    rcoordUp[ispan], 
                                    0, 
                                    zcoordUp[ispan],
                                    stwLenUp[ispan]))
            
            filename = 'spanwise_spline_outlet_blade_%i.csv' % iblade
            with open(foldername + '/' + filename, 'w') as file:
                file.write('indexStream,indexSpan,coordX,coordY,coordZ,streamlineLength\n')
                for istream in range(ni):
                    for ispan in range(nj):
                        file.write('%i,%i,%.9f,%.9f,%.9f,%.9f\n'
                                %(istream, ispan,
                                    rcoordDown[ispan], 
                                    0, 
                                    zcoordDown[ispan],
                                    stwLenDown[ispan]))
            
            

        
    
    
    def write_thetaWrapped_hub_shroud_curves(self, filename='machine', foldername='Grid'):

        rHub = self.r_grid_points[:,0]
        zHub = self.z_grid_points[:,0]

        rShroud = self.r_grid_points[:,-1]
        zShroud = self.z_grid_points[:,-1]
        
        thetaHub = self.theta_camber[:,0]
        thetaShroud = self.theta_camber[:,-1]
        
        xHub = rHub * np.cos(thetaHub)
        yHub = rHub * np.sin(thetaHub)
        xShroud = rShroud * np.cos(thetaShroud)
        yShroud = rShroud * np.sin(thetaShroud)
        
        if self.config.invert_axial_coordinates():
            zHub *= -1
            zShroud *= -1
        
        
        os.makedirs(foldername, exist_ok=True)
        with open(foldername + '/' + filename + '_hub.txt', 'w') as file:
            for iPoint in range(len(xHub)):
                file.write('%.9f\t%.9f\t%.9f\n' %(xHub[iPoint], yHub[iPoint], zHub[iPoint]))
        
        with open(foldername + '/' + filename + '_shroud.txt', 'w') as file:
            for iPoint in range(len(xHub)):
                file.write('%.9f\t%.9f\t%.9f\n' %(xShroud[iPoint], yShroud[iPoint], zShroud[iPoint]))
        
        
    

    def write_turbobfm_grid_file_2D(self):
        """
        Needed by turboBFM. The dictionnary saved must contain a X and Y for 2D, and X,Y,Z for 3D simulations.
        """
        outputFields = self.config.get_turbo_BFM_mesh_output_fields()
        
        X = self.z_grid_points
        Y = self.r_grid_points

        ni,nj = X.shape

        mesh = {'X': X, 'Y': Y}
        if 'blockage' in outputFields:
            print('Blockage grid added to the TurboBFM mesh file')
            mesh['Blockage'] = self.blockage
            
        if 'camber' in outputFields:
            print('Camber normal vector grid added to the TurboBFM mesh file')
            mesh['Normal'] = self.normal_camber
        
        if 'rpm' in outputFields:
            print('RPM grid added to the TurboBFM mesh file')
            mesh['RPM'] = self.rpm
        
        if 'stwl' in outputFields:
            print('Streamwise length added to the TurboBFM mesh file')
            mesh['StreamwiseLength'] = self.streamline_length
        
        if 'blade_present' in outputFields:
            print('Blade presence grid added to the TurboBFM mesh file')
            mesh['BladePresent'] = self.bladePresent
        
        if 'number_blades' in outputFields:
            print('Number of blades grid added to the TurboBFM mesh file')
            mesh['NumberBlades'] = self.nBlades
        
        if 'frozen_force' in outputFields:
            print('Frozen forces added to the TurboBFM mesh file')
            mesh['Force_Axial'] = self.force_axial
            mesh['Force_Radial'] = self.force_radial
            mesh['Force_Tangential'] = self.force_tangential
        
        if 'calibration_coefficients' in outputFields:
            bf_model = self.config.get_body_force_calibration_method()
            print(f"Calibration coefficients for model {bf_model} added to the TurboBFM mesh file")
            mesh['Calibration_Coefficients'] = self.BFCalibrationCoefficients
        

        filepath = self.config.get_output_data_folder() + '/TurboBFM_Mesh_%02i_%02i.pik' % (ni, nj)
        with open(filepath, 'wb') as f:
            pickle.dump(mesh, f)
        print(f"TurboBFM mesh pickle file saved to {filepath}")
    
    
    
    def write_cturbobfm_grid_file(self):
        """
        Needed by CTurboBFM. Save a CSV file with all the specified grids.
        """
        outputFields = self.config.get_turbo_BFM_mesh_output_fields()
        
        X = self.z_grid_points
        Y = self.r_grid_points

        ni,nj = X.shape
        nk = 1

        mesh = {'x': X, 'y': Y}
        mesh['z'] = np.zeros((ni,nj))
        
        for key in self.bfmFields.keys():
            print(f"{key} grid added to the CTurboBFM mesh file")
            mesh[key] = self.bfmFields[key]
        
        outputTopology = self.config.get_mesh_output_topology()
        
        if outputTopology.lower() == 'axisymmetric':
            self.write_axisymmetric_csv_grid_file(mesh)
        elif outputTopology.lower() == 'periodic' or outputTopology.lower() == 'full_annulus':
            self.write_3D_csv_grid_file(mesh)
        
    
    def write_axisymmetric_csv_grid_file(self, mesh):
        ni,nj = mesh['x'].shape
        filepath = self.config.get_output_data_folder() + '/CTurboBFM_Mesh_%02i_%02i.csv' % (ni, nj)
        with open(filepath, 'w') as f:
            f.write('NI=%i\n' % ni)
            f.write('NJ=%i\n' % nj)
            f.write('NK=1\n')
            for key in mesh.keys():
                if key == 'x':
                    f.write('%s' % key)
                else:
                    f.write(',%s' % key)
            f.write('\n')
            for i in range(ni):
                for j in range(nj):
                    for key, values in mesh.items():
                        if key == 'x':
                            f.write('%.15f' % values[i,j])
                        else:
                            f.write(',%.15f' % values[i,j])
                    f.write('\n')
            
        print(f"CTurboBFM axisymmetric mesh pickle file saved to {filepath}")
    
    def write_3D_csv_grid_file(self, mesh):
        ni, nj = mesh['x'].shape
        nk = self.config.get_mesh_periodic_number_points()
        
        if self.config.get_mesh_output_topology().lower() == 'periodic':
            periodicityTheta = self.config.get_mesh_periodicity_angle()
            theta = np.linspace(0, periodicityTheta, nk)
        elif self.config.get_mesh_output_topology().lower() == 'full_annulus':
            theta = np.linspace(0, 2*np.pi, nk)
        else:
            raise ValueError('Unknown mesh output topology: %s' % self.config.get_mesh_output_topology())
        
        xgrid = np.zeros((ni,nj,nk))
        ygrid = np.zeros((ni,nj,nk))
        zgrid = np.zeros((ni,nj,nk))
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    xPoint = mesh['x'][i,j]
                    rPoint = mesh['y'][i,j]
                    yPoint = rPoint * np.cos(theta[k])
                    zPoint = rPoint * np.sin(theta[k])
                    xgrid[i,j,k] = xPoint
                    ygrid[i,j,k] = yPoint
                    zgrid[i,j,k] = zPoint
        
        mesh['x'] = xgrid
        mesh['y'] = ygrid
        mesh['z'] = zgrid
        
        filepath = self.config.get_output_data_folder() + '/CTurboBFM_Mesh_%02i_%02i_%02i.csv' % (ni, nj, nk)
        with open(filepath, 'w') as f:
            f.write('NI=%i\n' % ni)
            f.write('NJ=%i\n' % nj)
            f.write('NK=%i\n' % nk)
            for key in mesh.keys():
                if key == 'x':
                    f.write('%s' % key)
                else:
                    f.write(',%s' % key)
            f.write('\n')
            for i in range(ni):
                for j in range(nj):
                    for k in range(nk):
                        for key, values in mesh.items():
                            if key == 'x':
                                f.write('%.15f' % values[i,j,k])
                            elif key == 'y' or key == 'z':
                                f.write(',%.15f' % values[i,j,k])
                            else:
                                f.write(',%.15f' % values[i,j])
                        f.write('\n')
                    
                    
        
    
    
    def plot_all_relevant_contours(self):
        # plot geometric quantities always contained
        for key in self.bfmFields.keys():
            contour_template(self.z_grid_cg, self.r_grid_cg, self.bfmFields[key], name=key, save_filename='multiblock_%s' %key, folder_name=self.config.get_pictures_folder_path())
        
        
        # # now plot only if really needed
        # if self.config.get_body_force_calibration_method().lower() == 'lift/drag':
        #     for key in self.BFCalibrationCoefficients.keys():
        #         if key.lower()!='model':
        #             contour_template(self.z_grid_cg, self.r_grid_cg, self.BFCalibrationCoefficients[key], 'BF Coefficient %s' %key, save_filename='multiblock_calibration_coefficient_%s' %key, folder_name=self.config.get_pictures_folder_path())
        
        # if self.config.get_body_force_calibration_method().lower() == 'inference':
        #     for key in self.inferenceCoefficients.keys():
        #         contour_template(self.z_grid_cg, self.r_grid_cg, self.inferenceCoefficients[key], 'inferenceCoefficients %s' %key, save_filename='multiblock_inference_coefficient_%s' %key, folder_name=self.config.get_pictures_folder_path())
        
        # bodyForceReconstructed = self.config.perform_body_force_reconstruction()
        # if bodyForceReconstructed:
        #     contour_template(self.z_grid_cg, self.r_grid_cg, self.force_axial, r'$f_{ax} \ \rm{[N/kg]}$', save_filename='multiblock_forceAxial', folder_name=self.config.get_pictures_folder_path())
        #     contour_template(self.z_grid_cg, self.r_grid_cg, self.force_radial, r'$f_{r} \ \rm{[N/kg]}$', save_filename='multiblock_forceRadial', folder_name=self.config.get_pictures_folder_path())
        #     contour_template(self.z_grid_cg, self.r_grid_cg, self.force_tangential, r'$f_{\theta} \ \rm{[N/kg]}$', save_filename='multiblock_forceTangential', folder_name=self.config.get_pictures_folder_path())
        #     contour_template(self.z_grid_cg, self.r_grid_cg, self.angular_momentum_derivative, r'$\partial (r u_{\theta}) / \partial m \ \rm{[m/s]}$', save_filename='multiblock_angularMomentumDerivative', folder_name=self.config.get_pictures_folder_path())
        #     contour_template(self.z_grid_cg, self.r_grid_cg, self.entropy_derivative, r'$\partial (s) / \partial m \ \rm{[J/kgKm]}$', save_filename='multiblock_entropyDerivative', folder_name=self.config.get_pictures_folder_path())
        
        
    def fix_theta_camber_grids(self):
        """For all blocks, try a good compromise with the theta camber. If theta camber is not present, copy the info from the previous block
        """
        if len( self.blocks )== 1:
            self.theta_camber = self.blocks[0].theta_camber
        else:
            
            for iBlock, block in enumerate(self.blocks):
                # contour_template(block.z_grid_points, block.r_grid_points, block.theta_camber, r'$\theta_{c}$ [deg]')
                
                if np.mean(np.abs(block.theta_camber))<1e-3: # this is the case when we don't have a blade camber -> unbladed block
                    ni,nj = block.theta_camber.shape
                    
                    if iBlock == 0: # if the block is the first, simply take the camber from the next one
                        for j in range(nj):
                            block.theta_camber[:,j] = self.blocks[iBlock+1].theta_camber[0,j]
                    elif iBlock == len(self.blocks)-1:  # if the block is the last, simply take the camber from the previous one
                        for j in range(nj):
                            block.theta_camber[:,j] = self.blocks[iBlock-1].theta_camber[-1,j]
                    else: # if the block is in the middle, interpolate linearly between the two bladed-blocks on the side
                        theta_previous = self.blocks[iBlock-1].theta_camber[-1,:]
                        theta_next = self.blocks[iBlock+1].theta_camber[0,:]
                        normStreamLength = compute_meridional_streamwise_coordinates(block.z_grid_points, block.r_grid_points, normalize=True)
                        theta_camber = np.zeros_like(normStreamLength)
                        theta_camber[0,:] = theta_previous
                        theta_camber[-1,:] = theta_next
                        for i in range(1,theta_camber.shape[0]-1):
                            theta_camber[i,:] = theta_camber[0,:] + normStreamLength[i,:] * (theta_camber[-1,:]-theta_camber[0,:])
                        block.theta_camber = theta_camber
            
                # contour_template(block.z_grid_points, block.r_grid_points, block.theta_camber, r'$\theta_{c}$ [deg]')
        
        
        

    def cut_meridional_grid(self, zgrid, rgrid, inlet_cut_type, outlet_cut_type, coord_cutIn, coord_cutOut):
        
        # pointer to reference grids
        if inlet_cut_type.lower()=='axial':
            gridIn = zgrid
        elif inlet_cut_type.lower()=='radial':
            gridIn = rgrid
        
        if outlet_cut_type.lower()=='axial':
            gridOut = zgrid
        elif outlet_cut_type.lower()=='radial':
            gridOut = rgrid
            
        # the cut is given depending on the coordinate of the hub
        idxIN = np.argmin(np.abs(gridIn[:,0] - coord_cutIn))
        idxOUT = np.argmin(np.abs(gridOut[:,-1] - coord_cutOut))
        
        zgridCut = zgrid[idxIN:idxOUT,:]
        rgridCut = rgrid[idxIN:idxOUT,:]
        
        return zgridCut, rgridCut