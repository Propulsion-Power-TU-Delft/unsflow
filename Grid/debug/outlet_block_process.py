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

start_time = time.time()
print('Start execution:')

# compute the bladed domain block object
data_folder_path = '../../Grid/geo/'
units = '[m]'
nstream = 15
nspan = 10
stream_grid_sampling = 'clustering_left'
span_grid_sampling = 'clustering'

hub = Grid.src.Curve(curve_filepath=data_folder_path + 'iris_hub.curve', units=units, degree_spline=1)
shroud = Grid.src.Curve(curve_filepath=data_folder_path + 'iris_shroud.curve', units=units, degree_spline=3)
block = Grid.src.Block(hub, shroud, blade_file=data_folder_path + 'iris_blade.curve', block_type='inlet',
                     nstream=nstream, nspan=nspan)

# compute the blade object info, in order to cut the block appropriately
blade = Grid.src.Blade(data_folder_path + 'iris_blade.curve')
blade.find_inlet_points()
blade.find_outlet_points()

# cut the bladed block properly, and compute the meridional structured mesh
block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
block.extend_inlet_outlet_curves()
block.find_intersections()
block.outlet_zone_trim()
block.spline_of_trim()
block.sample(sampling_mode = stream_grid_sampling)
block.compute_span_points(sampling_mode = span_grid_sampling)
block.compute_double_grid()
block.find_border()
block.plot_full_grid(save_filename='grid_%2d_%2d' % (nstream, nspan))

# instantiate cfd data object and perform processing removing the outliers
file_name = '../data/iris_85krpm_0.11kgs.csv'
data = Grid.src.CfdData(file_name, cut_block=block, blade=blade, verbose=True)
data.process_from_ansys_csv()
# data.scatter_plot(field='rho')
# data.scatter_plot(field='ur')
# data.scatter_plot(field='ut')
# data.scatter_plot(field='uz')
# data.scatter_plot(field='p')
# data.compute_flow_ideal_vectors()
# data.compute_bfm_radial_fields()

# instantiate meridional process object
data_process = Grid.src.MeridionalProcess(data, block=block, blade=blade, verbose=True)
data_process.circumferential_average(mode='circular', fix_borders=False, gauss_filter=True)
data_process.compute_rbf_gradients()
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
# data_process.contour_plot(field='mu', save_filename='mu_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='k', save_filename='k_%2d_%2d_interp' % (nstream, nspan))
# data_process.contour_plot(field='a1', save_filename='a1_%2d_%2d' % (nstream, nspan))
# data_process.contour_plot(field='a2', save_filename='a2_%2d_%2d' % (nstream, nspan))
# data_process.contour_plot(field='a3', save_filename='a3_%2d_%2d' % (nstream, nspan))
# data_process.contour_plot(field='F_ntheta', save_filename='F_ntheta_%2d_%2d' % (nstream, nspan))
# data_process.contour_plot(field='F_nr', save_filename='F_nr_%2d_%2d' % (nstream, nspan))
# data_process.contour_plot(field='F_nz', save_filename='F_nz_%2d_%2d' % (nstream, nspan))
# data_process.contour_plot(field='F_t', save_filename='F_t_%2d_%2d' % (nstream, nspan))
data_process.quiver_plot(field='p', save_filename='quiver_p_%2d_%2d' % (nstream, nspan))
data_process.quiver_plot(save_filename='quiver_%2d_%2d' % (nstream, nspan))


data_process.store_pickle(folder='data/meta/', file_name='iris_outlet_data_%2d_%2d' % (nstream, nspan))
end_time = time.time()
delta_time = end_time-start_time
print('Total time: %d sec' % (delta_time))
# plt.show()

