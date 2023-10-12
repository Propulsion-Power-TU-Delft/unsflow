#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:41:53 2023
@author: F. Neri, TU Delft

"""
import time
import matplotlib.pyplot as plt
import sys

sys.path.append('../../Grid')
import Grid
import pickle
import numpy as np

start_time = time.time()
print('Start execution:')

# compute the bladed domain block object
data_folder_path = 'data/geo/'
units = '[m]'
nstream = 20
nspan = 20
stream_grid_sampling = 'default'
span_grid_sampling = 'default'

hub = Grid.src.Curve(curve_filepath=data_folder_path + 'iris_hub.curve', units=units, degree_spline=1, rescale_factor=1, x_ref=0.0228)
shroud = Grid.src.Curve(curve_filepath=data_folder_path + 'iris_shroud.curve', units=units, degree_spline=1, rescale_factor=1, x_ref=0.0228)
block = Grid.src.Block(hub, shroud, nstream=nstream, nspan=nspan)

# compute the blade object info, in order to cut the block appropriately
blade = Grid.src.Blade(data_folder_path + 'iris_blade.curve', rescale_factor=1, x_ref=0.0228)
blade.find_inlet_points(geometry_type='axial')
blade.find_outlet_points(geometry_type='axial')

# cut the bladed block properly, and compute the meridional structured mesh
block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
block.extend_inlet_outlet_curves()
block.find_intersections(tol=1e-2)
block.inlet_zone_trim()
block.spline_of_hub_shroud()
block.spline_of_outlet()
block.sample_hub_shroud(sampling_mode=stream_grid_sampling)
block.sample_outlet(sampling_mode=span_grid_sampling)
block.compute_grid_points(sampling_mode=span_grid_sampling, grid_mode='spanwise', curved_border='right', smoothing='elliptic',
                          orthogonality=False, x_stretching=False, y_stretching=False,
                          sigmoid_coeff_x=7, sigmoid_coeff_y=7, method='minimize')
block.compute_grid_centers()
block.plot_full_grid(save_filename='inlet_grid_%2d_%2d' % (nstream, nspan), primary_grid=True)

# instantiate cfd data object and perform processing removing the outliers
file_name = 'data/meta/iris_85krpm_0.11kgs.csv'
data = Grid.src.CfdData(file_name, rpm_drag=0, blade=blade, cut_block=block, verbose=True, normalize=True,
                        rho_ref=1.014, x_ref=0.0228, rpm_ref=85e3, T_ref=288.15)
data.process_from_ansys_csv()

# instantiate meridional process object and avg
data_process = Grid.src.MeridionalProcess(data, block=block, blade=blade, verbose=True)
data_process.compute_streamline_length()
data_process.circumferential_average(mode='cell centered', fix_borders=False, gauss_filter=False)
data_process.compute_regressed_fields(order=4)
data_process.compute_derived_quantities()
data_process.compute_averaged_fluxes()

# data_process.contour_plot(field='rho', save_filename='rho_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='ur', save_filename='ur_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='ut', save_filename='ut_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='ut_rel', save_filename='ut_rel_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='ut_drag', save_filename='ut_drag_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='uz', save_filename='uz_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='p', save_filename='p_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='s', save_filename='s_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='T', save_filename='T_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='drho_dr', save_filename='drho_dr_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='drho_dz', save_filename='drho_dz_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='dur_dr', save_filename='dur_dr_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='dur_dz', save_filename='dur_dz_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='dut_dr', save_filename='dut_dr_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='dut_dz', save_filename='dut_dz_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='duz_dr', save_filename='duz_dr_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='duz_dz', save_filename='duz_dz_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='dp_dr', save_filename='dp_dr_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='dp_dz', save_filename='dp_dz_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='dT_dr', save_filename='dT_dr_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='dT_dz', save_filename='dT_dz_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='ds_dr', save_filename='ds_dr_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='ds_dz', save_filename='ds_dz_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='M', save_filename='M_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='p_tot', save_filename='p_tot_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='p_tot_bar', save_filename='p_tot_bar_%2d_%2d_interp' % (nstream, nspan))
# data_process.quiver_plot(field='p', save_filename='quiver_p_%2d_%2d' % (nstream, nspan))
# data_process.quiver_plot(save_filename='quiver_%2d_%2d' % (nstream, nspan))

delattr(data_process, 'data')
data_process.store_pickle(file_name='iris_inlet_%d_%d' %(nstream, nspan))
end_time = time.time()
delta_time = end_time-start_time
print('Total time: %d sec' % (delta_time))
# plt.show()
