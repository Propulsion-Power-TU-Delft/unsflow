#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:29:29 2023
@author: F. Neri, TU Delft
"""
import numpy as np
import matplotlib.path as mpath
from .styles import *


class Multiblock:
    """
    class that contains multiple block objects, and assemble them together
    """

    def __init__(self, units='in'):
        self.n_blocks = 0
        self.blocks = []
        self.units = units

    def add_block(self, block_grid):
        """
        add a block, follow streamwise order
        """
        self.blocks.append(block_grid)
        self.n_blocks += 1

    def assemble_multiblock(self):
        """
        build global array of grid cordinates
        """
        self.nspan = self.blocks[0].nspan
        self.z_grid = np.zeros((0, self.nspan))
        self.r_grid = np.zeros((0, self.nspan))
        for block in self.blocks:
            # iblock = 0
            self.z_grid = np.concatenate((self.z_grid, block.z_grid_points))
            self.r_grid = np.concatenate((self.r_grid, block.r_grid_points))
        self.nstream = self.z_grid.shape[0]
        self.nAxialNodes = self.nstream
        self.nRadialNodes = self.nspan

        self.find_borders()

    def find_borders(self):
        """
        find the 4 delimiting borders of the block, and the quadrilateral path delimiting it
        """
        inlet_z = self.z_grid[0, :]
        inlet_r = self.r_grid[0, :]
        self.inlet = np.stack((inlet_z, inlet_r), axis=1)

        outlet_z = self.z_grid[-1, :]
        outlet_r = self.r_grid[-1, :]
        self.outlet = np.stack((outlet_z, outlet_r), axis=1)

        hub_z = self.z_grid[:, 0]
        hub_r = self.r_grid[:, 0]
        self.hub = np.stack((hub_z, hub_r), axis=1)

        shroud_z = self.z_grid[:, -1]
        shroud_r = self.r_grid[:, -1]
        self.shroud = np.stack((shroud_z, shroud_r), axis=1)

        path_data = []
        path_data.extend(self.inlet)
        path_data.extend(self.hub)
        path_data.extend(self.outlet)
        path_data.extend(self.shroud)
        self.delimiting_path = mpath.Path(path_data)

    def plot_full_grid(self, save_filename=None, primary_grid=True, primary_grid_points=False, hub_shroud=False):
        """
        plot everything of the grid
        """
        plt.figure(figsize=fig_size)

        # hub and shroud plot
        if hub_shroud:
            for block in self.blocks:
                plt.plot(block.hub.z_spline, block.hub.r_spline, lw=light_line_width, c='black')
                plt.plot(block.shroud.z_spline, block.shroud.r_spline, lw=light_line_width, c='black')

        # primary grid
        if primary_grid:
            for istream in range(0, self.nstream):
                plt.plot(self.z_grid[istream, :], self.r_grid[istream, :], lw=light_line_width, c='black')
            for ispan in range(0, self.nspan):
                plt.plot(self.z_grid[:, ispan], self.r_grid[:, ispan], lw=light_line_width, c='black')

        # primary grid points
        if primary_grid_points:
            plt.scatter(self.z_grid.flatten(), self.r_grid.flatten(),
                        c='black', s=scatter_point_size, label='primary grid nodes')

        if (primary_grid_points):
            plt.legend()
        plt.xlabel(r'$z \ \mathrm{[%s]}$' % (self.units))
        plt.ylabel(r'$r \ \mathrm{[%s]}$' % (self.units))

        if len(self.blocks) > 1:
            plt.title(r'$(%d + %d + %d) \times %d$' % (tuple(self.blocks[i].nstream for i in range(0, self.n_blocks)) +
                                                       (self.blocks[0].nspan,)))
        elif len(self.blocks) == 1:
            plt.title(r'$(%d \times %d)$' % (self.blocks[0].nstream, self.blocks[0].nspan))

        if save_filename is not None:
            plt.savefig(folder_name + save_filename, bbox_inches='tight')

    # def find_minimum_distance(self, istream, ispan):
    #     r_west = sqrt((self.r_grid[istream+1, ispan]-self.r_grid[istream, ispan])**2 + 
    #                      (self.z_grid[istream+1, ispan]-self.z_grid[istream, ispan])**2)
    #     r_east = sqrt((self.r_grid[istream-1, ispan]-self.r_grid[istream, ispan])**2 + 
    #                      (self.z_grid[istream-1, ispan]-self.z_grid[istream, ispan])**2)
    #     r_north = sqrt((self.r_grid[istream, ispan+1]-self.r_grid[istream, ispan])**2 + 
    #                      (self.z_grid[istream, ispan+1]-self.z_grid[istream, ispan])**2)
    #     r_south = sqrt((self.r_grid[istream, ispan-1]-self.r_grid[istream, ispan])**2 + 
    #                      (self.z_grid[istream, ispan-1]-self.z_grid[istream, ispan])**2)
    #     r_min = min(r_west, r_east, r_north, r_south)
    #     return r_min

    # def compute_double_grid(self):
    #     """
    #     compute a secondary grid, using the points that lie in the baricenter of 4 primary grid points
    #     """
    #     self.z_grid_centers = np.zeros((self.nstream+1, self.nspan+1))
    #     self.r_grid_centers = np.zeros((self.nstream+1, self.nspan+1))

    #     #internal points
    #     for istream in range(1,self.nstream):
    #         for ispan in range(1,self.nspan):

    #                 z_mid_point = 0.25*(self.z_grid[istream, ispan] + self.z_grid[istream-1, ispan] + 
    #                                     self.z_grid[istream, ispan-1] + self.z_grid[istream-1, ispan-1])

    #                 r_mid_point = 0.25*(self.r_grid[istream, ispan] + self.r_grid[istream-1, ispan] + 
    #                                     self.r_grid[istream, ispan-1] + self.r_grid[istream-1, ispan-1])

    #                 self.z_grid_centers[istream, ispan] = z_mid_point
    #                 self.r_grid_centers[istream, ispan] = r_mid_point

    #     #vertices
    #     self.z_grid_centers[0,0] = self.z_grid[0,0]
    #     self.r_grid_centers[0,0] = self.r_grid[0,0]
    #     self.z_grid_centers[0,-1] = self.z_grid[0,-1]
    #     self.r_grid_centers[0,-1] = self.r_grid[0,-1]
    #     self.z_grid_centers[-1,-1] = self.z_grid[-1,-1]
    #     self.r_grid_centers[-1,-1] = self.r_grid[-1,-1]
    #     self.z_grid_centers[-1,0] = self.z_grid[-1,0]
    #     self.r_grid_centers[-1,0] = self.r_grid[-1,0]

    #     #istream = 0 border
    #     for istream in range(0,1):
    #         for ispan in range(1,self.nspan):
    #             z_mid_point = 0.5*(self.z_grid[istream, ispan] + self.z_grid[istream, ispan-1])
    #             r_mid_point = 0.5*(self.r_grid[istream, ispan] + self.r_grid[istream, ispan-1])
    #             self.z_grid_centers[istream, ispan] = z_mid_point
    #             self.r_grid_centers[istream, ispan] = r_mid_point

    #     #istream = -1 border
    #     for istream in range(self.nstream,self.nstream+1):
    #         for ispan in range(1,self.nspan):
    #             z_mid_point = 0.5*(self.z_grid[istream-1, ispan] + self.z_grid[istream-1, ispan-1])
    #             r_mid_point = 0.5*(self.r_grid[istream-1, ispan] + self.r_grid[istream-1, ispan-1])
    #             self.z_grid_centers[istream, ispan] = z_mid_point
    #             self.r_grid_centers[istream, ispan] = r_mid_point

    #     #ispan = 0 border
    #     for istream in range(1,self.nstream):
    #         for ispan in range(0,1):
    #             z_mid_point = 0.5*(self.z_grid[istream, ispan] + self.z_grid[istream-1, ispan])
    #             r_mid_point = 0.5*(self.r_grid[istream, ispan] + self.r_grid[istream-1, ispan])
    #             self.z_grid_centers[istream, ispan] = z_mid_point
    #             self.r_grid_centers[istream, ispan] = r_mid_point

    #     #ispan = -1 border
    #     for istream in range(1,self.nstream):
    #         for ispan in range(self.nspan,self.nspan+1):
    #             z_mid_point = 0.5*(self.z_grid[istream, ispan-1] + self.z_grid[istream-1, ispan-1])
    #             r_mid_point = 0.5*(self.r_grid[istream, ispan-1] + self.r_grid[istream-1, ispan-1])
    #             self.z_grid_centers[istream, ispan] = z_mid_point
    #             self.r_grid_centers[istream, ispan] = r_mid_point
