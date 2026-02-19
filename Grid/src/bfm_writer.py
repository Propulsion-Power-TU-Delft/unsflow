#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:29:29 2023
@author: F. Neri, TU Delft
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

import grid.src
from utils.styles import *
from .functions import cluster_sample_u, elliptic_grid_generation, transfinite_grid_generation
from .curve import Curve
from sun.src.general_functions import print_banner_begin, print_banner_end
from .area_element import AreaElement
from scipy.interpolate import CubicSpline
from grid.src.functions import create_folder
import pickle


class BFM_Writer:
    """
    Class needed to write the body force input file necessary to SU2
    """

    def __init__(self, blades, config):
        """
        :param blades: the blades objects retaining the information
        :param config: config file object
        """
        self.blades = blades
        self.config = config

    def getDefaultOutFields(self, iblade):
        stream_LE = np.zeros_like(self.blades[iblade].streamline_length)
        for i in range(stream_LE.shape[0]):
            stream_LE[i,:] = self.blades[iblade].streamline_length[0, :]

        data = {'1:axial_coordinate': self.blades[iblade].z_camber,
                '2:radial_coordinate': self.blades[iblade].r_camber,
                '3:n_ax': self.blades[iblade].n_camber_z,
                '4:n_tang': self.blades[iblade].n_camber_t,
                '5:n_rad': self.blades[iblade].n_camber_r,
                '6:stw_LE': stream_LE,
                '7:stw': self.blades[iblade].streamline_length,
                '8:blockage_factor': self.blades[iblade].blockage}
        return data

    @staticmethod
    def getSingleStringOfKeys(myDict):
        """
        Concatenate all the keys in a single string
        """
        variables = ''
        for key, values in myDict.items():
            variables += key + ' '
        return variables

    @staticmethod
    def getSingleStringOfValues(myDict, i, j):
        """
        Concatenate all the values in a single string
        """
        variables = ''
        for key, values in myDict.items():
            variables += '%.10e\t' % values[i, j]
        variables += '\n'
        return variables


    def write_bfm_input_file(self, filename='BFM_input.drg', addFields=None):
        """
        Write the BFM configuration file according to SU2 structure.
        :param filename: filename of the bfm input file
        :param addFields: Fields added to the default values
        """
        # Build the dataset with the default values. List of blade files
        outFieldList = []
        for ii, blade in enumerate(self.blades):
            outFieldList.append(self.getDefaultOutFields(ii))

        # Add the additional requested values to each blade in the list
        for ii, OutFieldDict in enumerate(outFieldList):
            keyDum = len(OutFieldDict)
            if addFields is not None:
                for addField in addFields:
                    if addField == 'blockage_gradient':
                        OutFieldDict['%i:dblockage_daxial' %(keyDum)] = self.blades[ii].db_dz
                        OutFieldDict['%i:dblockage_dradial' %(keyDum+1)] = self.blades[ii].db_dr
                        keyDum += 2
                    elif addField == 'frozen_forces':
                        OutFieldDict['%i:force_turning' %(keyDum)] = self.blades[ii].meridional_fields['Force_Turning']
                        OutFieldDict['%i:force_loss' %(keyDum+1)] = self.blades[ii].meridional_fields['Force_Loss']
                        keyDum += 2
                    else:
                        raise ValueError('Not valid field')

        # Select the fields needed for the output, depending on the bfm-file model chosen
        if filename is None:
            filename = 'BFM_Input.drg'

        with (open(filename, 'w') as file):
            file.write('<header>\n')
            file.write('\n')

            file.write('[version inputfile]\n')
            file.write('1.0.0\n')
            file.write('\n')

            file.write('[number of blade rows]\n')
            rows = len(self.blades)
            file.write('%i\n' % rows)
            file.write('\n')

            file.write('[row blade count]\n')
            blade_count = self.config.get_blades_number()
            if isinstance(blade_count, int):
                file.write('%i\n' % blade_count)
            elif isinstance(blade_count, list):
                for dum in blade_count:
                    file.write('%i\t' % dum)
                file.write('\n')
            else:
                raise ValueError('Unknown blade count')
            file.write('\n')

            file.write('[rotation factor]\n')
            rotation_factor = self.config.get_rotation_factors()
            if isinstance(rotation_factor, int):
                file.write('%i\n' % rotation_factor)
            elif isinstance(rotation_factor, list):
                for dum in rotation_factor:
                    file.write('%i\t' % dum)
                file.write('\n')
            else:
                raise ValueError('Unknown rotation factor')
            file.write('\n')

            file.write('[number of tangential locations]\n')
            if isinstance(blade_count, int):
                file.write('1\n')
            elif isinstance(blade_count, list):
                for _ in blade_count:
                    file.write('1\t')
                file.write('\n')
            else:
                raise ValueError('Unknown number of tangential locations')
            file.write('\n')

            file.write('[number of data entries in chordwise direction]\n')
            if isinstance(self.blades, grid.src.Blade):
                file.write('%i\n' % self.blades.z_camber.shape[0])
            elif isinstance(self.blades, list):
                for blade in self.blades:
                    file.write('%i\t' % blade.z_camber.shape[0])
                file.write('\n')
            else:
                raise ValueError('Unknown number of chordwise direction points')
            file.write('\n')

            file.write('[number of data entries in spanwise direction]\n')
            if isinstance(self.blades, grid.src.Blade):
                file.write('%i\n' % self.blades.z_camber.shape[1])
            elif isinstance(self.blades, list):
                for blade in self.blades:
                    file.write('%i\t' % blade.z_camber.shape[1])
                file.write('\n')
            else:
                raise ValueError('Unknown number of chordwise direction points')
            file.write('\n')

            file.write('[variable names]\n')

            file.write(self.getSingleStringOfKeys(outFieldList[0]))
            file.write('\n')

            file.write('</header>\n')
            file.write('\n')

            file.write('<data>\n')

            for iblade, blade in enumerate(self.blades):
                file.write('<blade row>\n')
                file.write('<tang section>\n')

                for j in range(blade.z_camber.shape[1]):
                    file.write('<radial section>\n')
                    for i in range(blade.z_camber.shape[0]):
                        file.write(self.getSingleStringOfValues(outFieldList[iblade], i, j))
                    file.write('</radial section>\n')
                file.write('</tang section>\n')
                file.write('</blade row>\n')
            file.write('</data>')






