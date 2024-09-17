#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:29:29 2023
@author: F. Neri, TU Delft
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from Utils.styles import *
from .functions import cluster_sample_u, elliptic_grid_generation, compute_picture_size, transfinite_grid_generation
from .curve import Curve
from Sun.src.general_functions import print_banner_begin, print_banner_end
from .area_element import AreaElement
from scipy.interpolate import CubicSpline
from Grid.src.functions import create_folder
import pickle
import os


class MultiBlock:
    """
    this class contains a multiblock meridional grid
    """

    def __init__(self, *blocks):
        """
        Construct the Block object, storing all the data and methods for the meridional grid. There is no need to provide the
        dimensions and scaling factor of the cordinates since they are already used in the hub and shroud curve objects.
        :param config: configuration file
        :param nstream: number of grid points along the streamwise direction
        :param nspan: number of grid points along the spanwise direction
        """
        self.blocks = blocks

    def assemble_grid(self):
        # self.nstream = self.blocks[0].nstream
        # self.nspan = self.blocks[0].nspan
        self.z_grid_cg = self.blocks[0].z_grid_cg
        self.r_grid_cg = self.blocks[0].r_grid_cg

        for block in self.blocks[1:]:
            self.z_grid_cg = np.concatenate((self.z_grid_cg, block.z_grid_cg[1:, :]), axis=0)
            self.r_grid_cg = np.concatenate((self.r_grid_cg, block.r_grid_cg[1:, :]), axis=0)

        self.z_grid_points = self.z_grid_cg
        self.r_grid_points = self.r_grid_cg

        self.nstream = self.z_grid_points.shape[0]
        self.nspan = self.z_grid_points.shape[1]

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

        self.picture_size_blank, self.picture_size_contour = compute_picture_size(self.z_grid_cg, self.r_grid_cg)

        plt.figure(figsize=self.picture_size_blank)

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
        plt.xlabel(r'$z \ \mathrm{[-]}$')
        plt.ylabel(r'$r \ \mathrm{[-]}$')
        plt.title(r'$(%d \times %d)$' % (self.nstream, self.nspan))

        if not ticks:
            plt.xticks([])
            plt.yticks([])  # plt.xlabel('')  # plt.ylabel('')

        if save_filename is not None and save_foldername is not None:
            plt.savefig(save_foldername + '/' + save_filename + '.pdf', bbox_inches='tight')

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
        create_folder(folder)

        if filename is None:
            filename = 'spline'

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

    def write_paraview_grid_file(self, filename='meridional_grid.csv', foldername='Grid'):
        """
        write the file requireed by Paraview to run the circumferential avg.
        The format of the file generated is:
        istream, ispan, x, y, z
        """
        x = self.r_grid_dual
        y = np.zeros_like(self.r_grid_dual)
        z = self.z_grid_dual

        os.makedirs(foldername, exist_ok=True)
        with open(foldername + '/' + filename, 'w') as file:
            for istream in range(1, self.z_grid_dual.shape[0]-1):
                for ispan in range(1, self.z_grid_dual.shape[1]-1):
                    file.write('%i,%i,%.6f,%.6f,%.6f\n'
                               %(istream, ispan,
                                 x[istream, ispan], y[istream, ispan], z[istream, ispan]))
