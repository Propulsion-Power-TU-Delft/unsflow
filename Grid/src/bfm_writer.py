#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:29:29 2023
@author: F. Neri, TU Delft
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

import Grid.src
from Utils.styles import *
from .functions import cluster_sample_u, elliptic_grid_generation, compute_picture_size, transfinite_grid_generation
from .curve import Curve
from Sun.src.general_functions import print_banner_begin, print_banner_end
from .area_element import AreaElement
from scipy.interpolate import CubicSpline
from Grid.src.functions import create_folder
import pickle


class BFM_Writer:
    """
    Class needed to write the body force input file necessary to SU2
    """

    def __init__(self, blades, config):
        """
        :param blades: the blades objects retaining the information
        """
        self.blades = blades
        self.config = config


    def write_bfm_input_file(self, filename=None):
        """
        Write the BFM configuration file according to SU2 structure
        """
        if filename is None:
            filename = 'BFM_Input.drg'
        with open(filename, 'w') as file:
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
            if isinstance(self.blades, Grid.src.Blade):
                file.write('%i\n' % self.blades.z_camber.shape[0])
            elif isinstance(self.blades, list):
                for blade in self.blades:
                    file.write('%i\t' % blade.z_camber.shape[0])
                file.write('\n')
            else:
                raise ValueError('Unknown number of chordwise direction points')
            file.write('\n')

            file.write('[number of data entries in spanwise direction]\n')
            if isinstance(self.blades, Grid.src.Blade):
                file.write('%i\n' % self.blades.z_camber.shape[1])
            elif isinstance(self.blades, list):
                for blade in self.blades:
                    file.write('%i\t' % blade.z_camber.shape[1])
                file.write('\n')
            else:
                raise ValueError('Unknown number of chordwise direction points')
            file.write('\n')

            file.write('[variable names]\n')
            file.write('1:axial_coordinate 2:radial_coordinate 3:n_ax 4:n_tang 5:n_rad 6:blockage_factor 7:x_LE '
                       '8:axial_chord\n')
            file.write('\n')

            file.write('</header>\n')
            file.write('\n')

            file.write('<data>\n')

            for blade in self.blades:
                file.write('<blade row>\n')
                file.write('<tang section>\n')

                for j in range(blade.z_camber.shape[1]):
                    file.write('<radial section>\n')
                    for i in range(blade.z_camber.shape[0]):
                        file.write('%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\n' % (
                        blade.z_camber[i, j], blade.r_camber[i, j], blade.n_camber_z[i, j], blade.n_camber_t[i, j],
                        blade.n_camber_r[i, j], blade.blockage[i, j], blade.z_camber[0, j],
                        blade.z_camber[-1, j] - blade.z_camber[0, j]))
                    file.write('</radial section>\n')
                file.write('</tang section>\n')
                file.write('</blade row>\n')
            file.write('</data>')






