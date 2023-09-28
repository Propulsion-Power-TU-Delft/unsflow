#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:41:53 2023
@author: F. Neri, TU Delft

"""
import time
import Grid
import matplotlib.pyplot as plt
import pickle
import numpy as np

start_time = time.time()
print('Start execution:')

# compute the bladed domain block object
data_folder_path = 'nasa_rotor_37/cordinates/'
units = '[m]'
nstream = 50
nspan = 30
grid_sampling = 'default'
hub = Grid.src.Curve(curve_filepath=data_folder_path + 'hub.curve', units=units, degree_spline=3, rescale_factor=0.01, x_ref=0.252)
shroud = Grid.src.Curve(curve_filepath=data_folder_path + 'shroud.curve', units=units, degree_spline=3, rescale_factor=0.01, x_ref=0.252)
bladed_block = Grid.src.Block(hub, shroud, nstream=nstream, nspan=nspan)

# compute the blade object info, in order to cut the block appropriately
blade = Grid.src.Blade(data_folder_path + 'profile.curve', rescale_factor=0.01, x_ref=0.252)
blade.find_inlet_points('axial')
blade.find_outlet_points('axial')

# cut the bladed block properly, and compute the meridional structured mesh
bladed_block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
bladed_block.extend_inlet_outlet_curves()
bladed_block.find_intersections(tol=1e-3, visual_check=False)
bladed_block.bladed_zone_trim(machine_type='axial')
bladed_block.spline_of_hub_shroud()
bladed_block.spline_of_leading_trailing_edge()
bladed_block.sample_hub_shroud(sampling_mode=grid_sampling)
bladed_block.sample_leading_trailing_edges(sampling_mode=grid_sampling)
bladed_block.compute_grid_points(sampling_mode=grid_sampling, grid_mode='spanwise', curved_border='both', smoothing='elliptic',
                                 orthogonality=False, x_stretching=False, y_stretching=False)
bladed_block.compute_double_grid()
bladed_block.find_border()
bladed_block.plot_full_grid(save_filename='grid_%2d_%2d' % (nstream, nspan), primary_grid=True)

# find the camber surface, using the (z,r) grid found in the bladed block
blade.find_camber_surface(bladed_block)
# blade.plot_camber_surface(save_filename='camber_surface')
blade.compute_camber_vectors()
# blade.show_normal_vectors(save_filename='normal_vectors')
# blade.show_streamline_vectors(save_filename='streamline_vectors')
# blade.show_spanline_vectors(save_filename='spanline_vectors')
blade.compute_blade_camber_angles(convention='rotation-wise')
blade.show_blade_angles_contour(save_filename='geometry_%2d_%2d' % (nstream, nspan))

# instantiate cfd data object and perform processing removing the outliers
file_name = 'data/meta/config_01.csv'
data = Grid.src.CfdData(file_name, blade=blade, rpm_drag=-17189, cut_block=bladed_block, verbose=True, normalize=True,
                        rho_ref=1.014, x_ref=0.252, rpm_ref=-17189, T_ref=288.15)

data.process_from_ansys_csv()
data.compute_flow_ideal_vectors()
data.compute_bfm_radial_fields()

# instantiate meridional process object and avg
data_process = Grid.src.MeridionalProcess(data, block=bladed_block, blade=blade, verbose=True)
data_process.compute_streamline_length()
data_process.circumferential_average(mode='circular', bfm='radial', fix_borders=False, gauss_filter=True)
# data_process.compute_rbf_fields()
data_process.compute_rbf_gradients()
data_process.compute_bfm_axial(mode='global')

# final meridional plots
save_plots = True
if save_plots:
    data_process.contour_plot(field='streamline length', save_filename='sl_length_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='rho', save_filename='rho_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='ur', save_filename='ur_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='ut', save_filename='ut_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='ut_rel', save_filename='ut_rel_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='ut_drag', save_filename='ut_drag_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='uz', save_filename='uz_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='p', save_filename='p_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='s', save_filename='s_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='T', save_filename='T_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='drho_dr', save_filename='drho_dr_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='drho_dz', save_filename='drho_dz_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='dur_dr', save_filename='dur_dr_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='dur_dz', save_filename='dur_dz_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='dut_dr', save_filename='dut_dr_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='dut_dz', save_filename='dut_dz_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='duz_dr', save_filename='duz_dr_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='duz_dz', save_filename='duz_dz_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='dp_dr', save_filename='dp_dr_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='dp_dz', save_filename='dp_dz_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='dT_dr', save_filename='dT_dr_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='dT_dz', save_filename='dT_dz_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='ds_dr', save_filename='ds_dr_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='ds_dz', save_filename='ds_dz_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='M', save_filename='M_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='p_tot', save_filename='p_tot_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='p_tot_bar', save_filename='p_tot_bar_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='mu', save_filename='mu_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='k', save_filename='k_%2d_%2d_interp' % (nstream, nspan))
    data_process.contour_plot(field='a1', save_filename='a1_%2d_%2d' % (nstream, nspan))
    data_process.contour_plot(field='a2', save_filename='a2_%2d_%2d' % (nstream, nspan))
    data_process.contour_plot(field='a3', save_filename='a3_%2d_%2d' % (nstream, nspan))
    data_process.contour_plot(field='F_ntheta', save_filename='F_ntheta_%2d_%2d' % (nstream, nspan))
    data_process.contour_plot(field='F_nr', save_filename='F_nr_%2d_%2d' % (nstream, nspan))
    data_process.contour_plot(field='F_nz', save_filename='F_nz_%2d_%2d' % (nstream, nspan))
    data_process.contour_plot(field='F_t', save_filename='F_t_%2d_%2d' % (nstream, nspan))
    data_process.contour_plot(field='F_n', save_filename='F_n_%2d_%2d' % (nstream, nspan))
    data_process.quiver_plot(field='p', save_filename='quiver_p_%2d_%2d' % (nstream, nspan))

delattr(data_process, 'data')
data_process.store_pickle(file_name='nasa_rotor_config_01_blade_%d_%d' %(nstream, nspan))
end_time = time.time()
delta_time = end_time - start_time
print('Total time: %d sec' % (delta_time))
# plt.show()
