#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:52:09 2023
@author: F. Neri, TU Delft
"""
import copy

import numpy as np
import matplotlib.pyplot as plt
from .node import Node
from .styles import *
from .general_functions import GaussLobattoPoints


class SunGrid():
    """
    Class of Sun Grid. It contains a grid of Node objects, in which every node object contains the properties that we need
    to apply the Sun instability model.
    """

    def __init__(self, meridional_obj, mode='physical'):
        """
        instantiate the Sun Grid object, that contains the grids (physical and spectral), and the arrays of nodes objects
        contaning the fields and the matrices necessary for the instability model
        :param meridional_obj: object storing the 2D meridional flow fields, processed from CFD.
        :param mode: if physical it stores physical cordinates and fields data, if spectral it stores only spectral cordinates.
        """
        self._meridional_obj = meridional_obj  # data contaning the fluid dynamic fields on the meridional plane
        self.n_stream = meridional_obj.nstream
        self.nAxialNodes = self.n_stream
        self.n_span = meridional_obj.nspan
        self.nRadialNodes = self.n_span
        self.nPoints = self.n_stream*self.n_span

        if mode == 'physical':
            self.rGrid, self.zGrid = meridional_obj.r_cg.copy(), meridional_obj.z_cg.copy()
        elif mode == 'spectral':  # construct a gauss-lobatto grid for the spectral dataset
            self.z = GaussLobattoPoints(self.nAxialNodes)
            self.r = GaussLobattoPoints(self.nRadialNodes)
            self.rGrid, self.zGrid = np.meshgrid(self.r, self.z)
        else:
            raise ValueError("Unknown mode")

        if mode == 'physical':
            self.dataSet = np.empty((self.n_stream, self.n_span), dtype=Node)  # an array of Node elements
            counter = 0

            Nz = self.n_stream
            Nr = self.n_span
            for ii in range(0, self.n_stream):
                for jj in range(0, self.n_span):
                    # add first topological quantities
                    if ii == 0:
                        self.dataSet[ii, jj] = Node(meridional_obj.z_grid[ii, jj], meridional_obj.r_grid[ii, jj], 'inlet', counter)
                    elif ii == Nz - 1:
                        self.dataSet[ii, jj] = Node(meridional_obj.z_grid[ii, jj], meridional_obj.r_grid[ii, jj], 'outlet', counter)
                    elif jj == 0 and ii != 0 and ii != Nz - 1:
                        self.dataSet[ii, jj] = Node(meridional_obj.z_grid[ii, jj], meridional_obj.r_grid[ii, jj], 'hub', counter)
                    elif jj == Nr - 1 and ii != 0 and ii != Nz - 1:
                        self.dataSet[ii, jj] = Node(meridional_obj.z_grid[ii, jj], meridional_obj.r_grid[ii, jj], 'shroud', counter)
                    elif ii != 0 and ii != Nz - 1 and jj != 0 and jj != Nr - 1:
                        self.dataSet[ii, jj] = Node(meridional_obj.z_grid[ii, jj], meridional_obj.r_grid[ii, jj], 'internal', counter)
                    else:
                        raise ValueError("The constructor of the grid has some problems")

                    # add the fluid dynamic field if is a physical grid
                    if mode == 'physical':
                        self.dataSet[ii, jj].AppendDensityInfo(meridional_obj.rho[ii, jj],
                                                               meridional_obj.drho_dr[ii, jj],
                                                               meridional_obj.drho_dz[ii, jj])

                        self.dataSet[ii, jj].AppendVelocityInfo(meridional_obj.ur[ii, jj],
                                                                meridional_obj.ut[ii, jj],
                                                                meridional_obj.uz[ii, jj],
                                                                meridional_obj.dur_dr[ii, jj],
                                                                meridional_obj.dur_dz[ii, jj],
                                                                meridional_obj.dut_dr[ii, jj],
                                                                meridional_obj.dut_dz[ii, jj],
                                                                meridional_obj.duz_dr[ii, jj],
                                                                meridional_obj.duz_dz[ii, jj])

                        self.dataSet[ii, jj].AppendPressureInfo(meridional_obj.p[ii, jj],
                                                                meridional_obj.dp_dr[ii, jj],
                                                                meridional_obj.dp_dz[ii, jj])

                    if counter != jj+ii*self.n_span:
                        raise ValueError('Error in the numbering of the nodes')
                    counter += 1
        else:
            pass


    @property
    def meridional_obj(self):
        return self._meridional_obj

    def PrintInfo(self, datafile='terminal'):
        """
        Print information about the nodes.
        :param datafile: set printing destination.
        """
        for ii in range(0, self.nAxialNodes):
            for jj in range(0, self.nRadialNodes):
                self.dataSet[ii, jj].PrintInfo(datafile)


    def ComputeBoundaryNormals(self):
        """
        For every node on the hub and shroud it computes the normal vectors, and store them at the node level.
        The components are in the {r, theta, zeta} reference frame.
        """
        for ii in range(0, self.nAxialNodes):
            for jj in range(0, self.nRadialNodes):
                if (self.dataSet[ii, jj].marker == 'hub' or self.dataSet[ii, jj].marker == 'shroud'):
                    t_vec = np.array([self.dataSet[ii + 1, jj].r - self.dataSet[ii - 1, jj].r,
                             0,
                             self.dataSet[ii + 1, jj].z - self.dataSet[ii - 1, jj].z])  # tangent vector
                    t_vers = t_vec / np.linalg.norm(t_vec)  # tangent versor
                    n_vers = np.array([-t_vers[2],
                                       0,
                                       t_vers[0]])  # normal versor, perpendicular to t_vers in the r,z plane
                elif self.dataSet[ii, jj].marker == 'inlet':
                    n_vec = np.array([self.dataSet[ii + 1, jj].r - self.dataSet[ii, jj].r,
                                      0,
                                      self.dataSet[ii + 1, jj].z - self.dataSet[ii, jj].z])  # normal vector
                    n_vers = n_vec / np.linalg.norm(n_vec)
                elif self.dataSet[ii, jj].marker == 'outlet':
                    n_vec = np.array([self.dataSet[ii, jj].r - self.dataSet[ii-1, jj].r,
                                      0,
                                      self.dataSet[ii, jj].z - self.dataSet[ii-1, jj].z])  # normal vector
                    n_vers = n_vec / np.linalg.norm(n_vec)
                elif self.dataSet[ii, jj].marker != 'internal':
                    raise ValueError('Normal to the walls failed')

                self.dataSet[ii, jj].AddNormalVersor(n_vers)

    def ShowNormals(self):
        """
        Print information about the normals to the walls. The reference frame is {r,theta,z}.
        """
        self.ShowGrid(vector='wall normals')  # show only the boundaries with normal vector superposed



    def PhysicalToSpectralData(self):
        """
        It returns a new Grid object with the same data of the original one, but with spectral cordinates
        located on the gauss-lobatto points between 1 and -1 in both the directions.
        It conserves the same amount of grid nodes of the physical grid.
        """
        newGridObj = SunGrid(self.meridional_obj, mode='spectral')
        return newGridObj



    def ShowGrid(self, formatFig=(10, 6), save_filename=None, mode=None, vector=None):
        """
        Show a scatter plots of the grid, with different colors for the different zones.
        if mode is set to boundaries it plots only them with normal vectors superposed.
        :param formatFig: format of figure
        :param save_filename: specify name of the figs to save
        :param mode: type of visualization
        :param vector: if True plots also the boundary normals.
        """
        mark = np.empty((self.nAxialNodes, self.nRadialNodes), dtype=str)
        for ii in range(0, self.nAxialNodes):
            for jj in range(0, self.nRadialNodes):
                if self.dataSet[ii, jj].marker == 'inlet':
                    mark[ii, jj] = "i"
                elif self.dataSet[ii, jj].marker == 'outlet':
                    mark[ii, jj] = "o"
                elif self.dataSet[ii, jj].marker == 'hub':
                    mark[ii, jj] = "h"
                elif self.dataSet[ii, jj].marker == 'shroud':
                    mark[ii, jj] = "s"
                else:
                    mark[ii, jj] = ''

        plt.figure(figsize=formatFig)

        if mode is None:
            condition = mark == 'i'  # plot only the inlet points
            plt.scatter(self.zGrid[condition], self.rGrid[condition], label='inlet')
            condition = mark == 'o'  # plot only the outlet points
            plt.scatter(self.zGrid[condition], self.rGrid[condition], label='outlet')
            condition = mark == 'h'  # plot only the hub
            plt.scatter(self.zGrid[condition], self.rGrid[condition], label='hub')
            condition = mark == 's'  # plot only shroud
            plt.scatter(self.zGrid[condition], self.rGrid[condition], label='shroud')
            condition = mark == ''  # plot all the remaining internal points
            if vector is None:
                plt.scatter(self.zGrid[condition], self.rGrid[condition], c='black')

            if vector == 'wall normals':
                for ii in range(0, self.nAxialNodes):
                    for jj in range(0, self.nRadialNodes):
                        if (self.dataSet[ii, jj].marker != 'internal'):
                            plt.quiver(self.dataSet[ii, jj].z, self.dataSet[ii, jj].r,
                                       self.dataSet[ii, jj].n_wall[2], self.dataSet[ii, jj].n_wall[0], scale=30)
            plt.legend()

        elif mode == 'lines':
            for ii in range(0, self.nAxialNodes):
                plt.plot(self.zGrid[ii, :], self.rGrid[ii, :], 'black', lw = light_line_width)
            for jj in range(0, self.nRadialNodes):
                plt.plot(self.zGrid[:, jj], self.rGrid[:, jj], 'black', lw = light_line_width)

        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')
            plt.close()



    def Normalize(self, rho_ref, u_ref, x_ref):
        """
        Taking reference magnitudes for density, velocity and length, it normalizes all the data
        contained in the grid, and in the nodes.
        :param rho_ref: reference density [kg/m3]
        :param u_ref: reference velocity [m/s]
        :param x_ref: reference length [m]
        """
        self.zGrid /= x_ref
        self.rGrid /= x_ref

        # normalize every node quantity with the reference ones
        for ii in range(0, self.n_stream):
            for jj in range(0, self.n_span):
                self.dataSet[ii, jj].Normalize(rho_ref, u_ref, x_ref)

    def return_2d_field(self, field):
        """
        Given the name of the field, return the 2D array
        """
        size = self.dataSet.shape
        array = np.zeros(size)
        for i in range(size[0]):
            for j in range(size[1]):
                if field == 'z':
                    array[i, j] = self.dataSet[i, j].z
                elif field == 'r':
                    array[i, j] = self.dataSet[i, j].r
                elif field == 'rho':
                    array[i, j] = self.dataSet[i, j].rho
                elif field == 'ur':
                    array[i, j] = self.dataSet[i, j].ur
                elif field == 'ut':
                    array[i, j] = self.dataSet[i, j].ut
                elif field == 'uz':
                    array[i, j] = self.dataSet[i, j].uz
                elif field == 'p':
                    array[i, j] = self.dataSet[i, j].p

                elif field == 'drho_dr':
                    array[i, j] = self.dataSet[i, j].drho_dr
                elif field == 'drho_dz':
                    array[i, j] = self.dataSet[i, j].drho_dz
                elif field == 'dur_dr':
                    array[i, j] = self.dataSet[i, j].dur_dr
                elif field == 'dur_dz':
                    array[i, j] = self.dataSet[i, j].dur_dz
                elif field == 'dut_dr':
                    array[i, j] = self.dataSet[i, j].dut_dr
                elif field == 'dut_dz':
                    array[i, j] = self.dataSet[i, j].dut_dz
                elif field == 'duz_dr':
                    array[i, j] = self.dataSet[i, j].duz_dr
                elif field == 'duz_dz':
                    array[i, j] = self.dataSet[i, j].duz_dz
                elif field == 'dp_dr':
                    array[i, j] = self.dataSet[i, j].dp_dr
                elif field == 'dp_dz':
                    array[i, j] = self.dataSet[i, j].dp_dz
        return array



    def check_fields(self):
        """
        Check the fluid dynamic fields associated to the nodes dataset.
        """
        z = self.return_2d_field('z')
        r = self.return_2d_field('r')

        fields = ['rho', 'ur', 'ut', 'uz', 'p', 'drho_dr', 'drho_dz', 'dur_dr', 'dur_dz', 'dut_dr', 'dut_dz', 'duz_dr', 'duz_dz',
                  'dp_dr', 'dp_dz']

        for field in fields:
            plt.figure()
            plt.contourf(z, r, self.return_2d_field(field), levels=N_levels, cmap=color_map)
            plt.title(field)
            plt.colorbar()

