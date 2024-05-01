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
import pickle


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
            for istream in range(0, self.nstream + 1):
                plt.plot(self.z_grid_dual[istream, :], self.r_grid_dual[istream, :], '--r', lw=light_line_width)
            for ispan in range(0, self.nspan + 1):
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

    def compute_three_dimensional_mesh(self, N_THETA):
        """
        Compute the Three-dimensional mesh X,Y,Z as 3D arrays, structured
        """
        theta = np.linspace(0, 2 * np.pi, N_THETA)
        self.X_mesh = np.zeros((self.nstream, self.nspan, N_THETA))
        self.Y_mesh = np.zeros((self.nstream, self.nspan, N_THETA))
        self.Z_mesh = np.zeros((self.nstream, self.nspan, N_THETA))

        for i in range(self.nstream):
            for j in range(self.nspan):
                for k in range(N_THETA):
                    self.X_mesh[i, j, k] = self.r_grid_cg[i, j] * np.cos(theta[k])
                    self.Y_mesh[i, j, k] = self.r_grid_cg[i, j] * np.sin(theta[k])
                    self.Z_mesh[i, j, k] = self.z_grid_cg[i, j]

    def save_mesh_pickle(self, filepath=None):
        """
        Save the mesh cordinates in a pickle
        """

        mesh = {'x': self.X_mesh, 'y': self.Y_mesh, 'z': self.Z_mesh}

        if filepath == None:
            filepath = 'mesh_%02i_%02i_%2i.pickle' % (self.nstream, self.nspan, self.X_mesh.shape[2])
        with open(filepath, 'wb') as f:
            pickle.dump(mesh, f)

        print(f"Data saved to '{filepath}'")
