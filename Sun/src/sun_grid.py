#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:52:09 2023
@author: F. Neri, TU Delft
"""
import copy

import numpy as np
import matplotlib.pyplot as plt
from Sun.src.node import Node
import Utils
from Sun.src.general_functions import GaussLobattoPoints
from Utils.styles import *
import os


class SunGrid():
    """
    Class of Sun Grid. It contains a grid of Node objects, in which every node object contains the properties that we need
    to apply the Sun instability model.
    """

    def __init__(self, inputData, gridType='physical'):
        """
        instantiate the Sun Grid object, that contains the grids (physical and spectral), and the arrays of nodes objects
        contaning the fields and the matrices necessary for the instability model
        :param meridional_obj: object storing the 2D meridional flow fields, processed from CFD.
        :param mode: if physical it stores physical cordinates and fields data, if spectral it stores only spectral cordinates.
        """
        self.nStream = inputData['AxialCoord'].shape[0]
        self.nSpan = inputData['AxialCoord'].shape[1]
        self.nPoints = self.nStream*self.nSpan
        self.inputData = inputData

        if gridType == 'physical':
            self.rGrid, self.zGrid = inputData['RadialCoord'].copy(), inputData['AxialCoord'].copy()
            
            self.dataSet = np.empty((self.nStream, self.nSpan), dtype=Node)  # an array of Node elements
            counter = 0

            Nz = self.nStream
            Nr = self.nSpan
            for ii in range(0, self.nStream):
                for jj in range(0, self.nSpan):
                    if ii == 0:
                        self.dataSet[ii, jj] = Node(self.zGrid[ii, jj], self.rGrid[ii, jj],
                                                    'inlet', counter)
                    elif ii == Nz - 1:
                        self.dataSet[ii, jj] = Node(self.zGrid[ii, jj], self.rGrid[ii, jj],
                                                    'outlet', counter)
                    elif jj == 0 and ii != 0 and ii != Nz - 1:
                        self.dataSet[ii, jj] = Node(self.zGrid[ii, jj], self.rGrid[ii, jj],
                                                    'hub', counter)
                    elif jj == Nr - 1 and ii != 0 and ii != Nz - 1:
                        self.dataSet[ii, jj] = Node(self.zGrid[ii, jj], self.rGrid[ii, jj],
                                                    'shroud', counter)
                    elif ii != 0 and ii != Nz - 1 and jj != 0 and jj != Nr - 1:
                        self.dataSet[ii, jj] = Node(self.zGrid[ii, jj], self.rGrid[ii, jj],
                                                    'internal', counter)
                    else:
                        raise ValueError("The constructor of the grid has some problems")

                    if counter != jj+ii*self.nSpan:
                        raise ValueError('Error in the numbering of the nodes')
                    counter += 1
        
        elif gridType == 'spectral':  # construct a gauss-lobatto grid for the spectral dataset
            self.z = GaussLobattoPoints(self.nStream)
            self.r = GaussLobattoPoints(self.nSpan)
            self.rGrid, self.zGrid = np.meshgrid(self.r, self.z)
        else:
            raise ValueError("Unknown grid type")


    @property
    def meridional_obj(self):
        return self._meridional_obj

    def PrintInfo(self, datafile='terminal'):
        """
        Print information about the nodes.
        :param datafile: set printing destination.
        """
        for ii in range(0, self.nStream):
            for jj in range(0, self.nSpan):
                self.dataSet[ii, jj].PrintInfo(datafile)

    def ComputeBoundaryNormals(self):
        """
        For every node on the hub and shroud it computes the normal vectors, and store them at the node level.
        The components are in the {r, theta, zeta} reference frame.
        """
        for ii in range(0, self.nStream):
            for jj in range(0, self.nSpan):
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
        newGridObj = SunGrid(self.inputData, gridType='spectral')
        return newGridObj



    def ShowGrid(self, save_filename=None, mode=None, vector=None):
        """
        Show a scatter plots of the grid, with different colors for the different zones.
        if mode is set to boundaries it plots only them with normal vectors superposed.
        :param formatFig: format of figure
        :param save_filename: specify name of the figs to save
        :param mode: type of visualization
        :param vector: if True plots also the boundary normals.
        """
        mark = np.empty((self.nStream, self.nSpan), dtype=str)
        for ii in range(0, self.nStream):
            for jj in range(0, self.nSpan):
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

        plt.figure()

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
                for ii in range(0, self.nStream):
                    for jj in range(0, self.nSpan):
                        if self.dataSet[ii, jj].marker == 'hub' or self.dataSet[ii, jj].marker == 'shroud':
                            plt.quiver(self.dataSet[ii, jj].z, self.dataSet[ii, jj].r,
                                       self.dataSet[ii, jj].n_wall[2], self.dataSet[ii, jj].n_wall[0])
            plt.legend()
            plt.gca().set_aspect('equal', adjustable='box')


        elif mode == 'lines':
            for ii in range(0, self.nStream):
                plt.plot(self.zGrid[ii, :], self.rGrid[ii, :], 'black', lw = light_line_width)
            for jj in range(0, self.nSpan):
                plt.plot(self.zGrid[:, jj], self.rGrid[:, jj], 'black', lw = light_line_width)

        if save_filename is not None:
            os.makedirs('pictures', exist_ok=True)
            plt.savefig('pictures/' + save_filename + '.pdf', bbox_inches='tight')



