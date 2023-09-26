#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:41:53 2023
@author: F. Neri, TU Delft

"""
import time
import Grid
import matplotlib.pyplot as plt

start_time = time.time()
print('Start execution:')

# compute the bladed domain block object
data_folder_path = 'data/geo/'
units = '[m]'
nstream = 30
nspan = 20
grid_sampling = 'default'
hub = Grid.src.Curve(curve_filepath=data_folder_path + 'iris_hub.curve', units=units, degree_spline=3,
                     rescale_factor=1, x_ref=0.0228)
shroud = Grid.src.Curve(curve_filepath=data_folder_path + 'iris_shroud.curve', units=units, degree_spline=3,
                        rescale_factor=1, x_ref=0.0228)
bladed_block = Grid.src.Block(hub, shroud, nstream=nstream, nspan=nspan)

# compute the blade object info, in order to cut the block appropriately
blade = Grid.src.Blade(data_folder_path + 'iris_blade.curve', rescale_factor=1, x_ref=0.0228)
blade.find_inlet_points('axial')
blade.find_outlet_points('radial')

# cut the bladed block properly, and compute the meridional structured mesh
bladed_block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
bladed_block.extend_inlet_outlet_curves()
bladed_block.find_intersections(tol=1e-2, visual_check=True)
bladed_block.bladed_zone_trim(machine_type='radial')
bladed_block.spline_of_hub_shroud()
bladed_block.spline_of_leading_trailing_edge()
bladed_block.sample_hub_shroud(sampling_mode=grid_sampling)
bladed_block.sample_leading_trailing_edges(sampling_mode=grid_sampling)
bladed_block.show_outline_grid()
bladed_block.compute_grid_points(sampling_mode=grid_sampling, grid_mode='spanwise', curved_border='both', smoothing='elliptic',
                                 orthogonality=False, x_stretching=False, y_stretching=False)
plt.show()