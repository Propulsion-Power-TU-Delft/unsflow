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






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GLOBAL DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_folder_path = 'nasa_rotor_37/cordinates/'
units = '[m]'
rho_ref = 1.014  # reference density [kg/m3]
x_ref = 0.252  # reference length, tip radius [m]
rpm_ref = -17189  # shaft rpm with sign
T_ref = 288.15  # reference temperature [K]
rescale_factor = 0.01  # cordinates of data files are in [cm]
sigmoid_coeff_stream = 8
sigmoid_coeff_span = 10






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INLET PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("\nINLET BLOCK PROCESSING...")
nstream = 15
nspan = 15
stream_grid_sampling = 'default'
span_grid_sampling = 'default'

hub = Grid.src.Curve(curve_filepath=data_folder_path + 'hub.curve', units=units, degree_spline=3,
                     rescale_factor=rescale_factor, x_ref=x_ref)
shroud = Grid.src.Curve(curve_filepath=data_folder_path + 'shroud.curve', units=units, degree_spline=3,
                        rescale_factor=rescale_factor, x_ref=x_ref)
block = Grid.src.Block(hub, shroud, nstream=nstream, nspan=nspan)

# compute the blade object info, in order to cut the block appropriately
blade = Grid.src.Blade(data_folder_path + 'profile.curve', rescale_factor=rescale_factor, x_ref=x_ref)
blade.find_inlet_points(geometry_type='axial')
blade.find_outlet_points(geometry_type='axial')

# cut the bladed block properly, and compute the meridional structured mesh
block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
block.extend_inlet_outlet_curves()
block.find_intersections()
block.inlet_zone_trim()
block.spline_of_hub_shroud()
block.spline_of_outlet()
block.sample_hub_shroud()
block.sample_outlet()
block.compute_grid_points(grid_mode='elliptic', orthogonality=True,
                          x_stretching='sigmoid_right', y_stretching='sigmoid',
                          sigmoid_coeff_x=sigmoid_coeff_stream, sigmoid_coeff_y=sigmoid_coeff_span)
block.compute_grid_centers()

# instantiate cfd data object and perform processing removing the outliers
file_name = 'data/meta/config_01.csv'
data = Grid.src.CfdData(file_name, rpm_drag=rpm_ref, blade=blade, cut_block=block, verbose=True, normalize=True,
                        rho_ref=rho_ref, x_ref=x_ref, rpm_ref=rpm_ref, T_ref=T_ref)
data.process_from_ansys_csv()

# instantiate meridional process object and avg
inlet_process = Grid.src.MeridionalProcess(data, block=block, verbose=True)
inlet_process.compute_streamline_length()
inlet_process.circumferential_average(gauss_filter=False)
inlet_process.compute_regressed_fields()
inlet_process.compute_derived_quantities()
inlet_process.compute_averaged_fluxes()
# inlet_process.contour_plot(field='M', save_filename='M_%2d_%2d_interp' % (nstream, nspan))
# inlet_process.quiver_plot(field='p', save_filename='quiver_p_%2d_%2d' % (nstream, nspan))
# inlet_process.plot_averaged_fluxes(field='M', save_filename='flux_M_%d_%d' %(nstream, nspan))
delattr(inlet_process, 'data')  # release useless memory






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BLADE PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("\nBLADE BLOCK PROCESSING...")
nstream = 15
nspan = 15
hub = Grid.src.Curve(curve_filepath=data_folder_path + 'hub.curve', units=units, degree_spline=3,
                     rescale_factor=rescale_factor, x_ref=x_ref)
shroud = Grid.src.Curve(curve_filepath=data_folder_path + 'shroud.curve', units=units, degree_spline=3,
                        rescale_factor=rescale_factor, x_ref=x_ref)
bladed_block = Grid.src.Block(hub, shroud, nstream=nstream, nspan=nspan)

# cut the bladed block properly, and compute the meridional structured mesh
bladed_block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
bladed_block.extend_inlet_outlet_curves()
bladed_block.find_intersections()
bladed_block.bladed_zone_trim(machine_type='axial')
bladed_block.spline_of_hub_shroud()
bladed_block.spline_of_leading_trailing_edge()
bladed_block.sample_hub_shroud()
bladed_block.sample_leading_trailing_edges()
bladed_block.compute_grid_points(grid_mode='elliptic', orthogonality=True, x_stretching='sigmoid', y_stretching='sigmoid',
                                 sigmoid_coeff_x=sigmoid_coeff_stream, sigmoid_coeff_y=sigmoid_coeff_span,
                                 inlet_meridional_obj=inlet_process)
bladed_block.compute_grid_centers()

blade.find_camber_surface(bladed_block)
blade.compute_camber_vectors()
blade.compute_blade_camber_angles(convention='rotation-wise')
blade.show_blade_angles_contour(save_filename='geometry_%2d_%2d' % (nstream, nspan))

# instantiate meridional process object and avg
blade_process = Grid.src.MeridionalProcess(data, block=bladed_block, blade=blade, verbose=True)
blade_process.compute_camber_angles()
blade_process.compute_streamline_length()
blade_process.circumferential_average(gauss_filter=False)
blade_process.compute_regressed_fields()
blade_process.compute_derived_quantities()
blade_process.compute_bfm_axial(mode='global', save_fig=True)
blade_process.compute_averaged_fluxes()

# blade_process.contour_plot(field='rho', save_filename='rho_%2d_%2d' % (nstream, nspan), quiver=True)
# blade_process.contour_plot(field='ur', save_filename='ur_%2d_%2d' % (nstream, nspan))
# blade_process.plot_averaged_fluxes(field='p', save_filename='flux_p_%d_%d' %(nstream, nspan))
# blade_process.plot_averaged_fluxes(field='s', save_filename='flux_s_%d_%d' %(nstream, nspan))
# blade_process.plot_averaged_fluxes(field='T', save_filename='flux_T_%d_%d' %(nstream, nspan))
delattr(blade_process, 'data')






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTLET PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("\nOUTLET BLOCK PROCESSING...")
nstream = 25
nspan = 15
hub = Grid.src.Curve(curve_filepath=data_folder_path + 'hub.curve', units=units, degree_spline=3,
                     rescale_factor=rescale_factor, x_ref=x_ref)
shroud = Grid.src.Curve(curve_filepath=data_folder_path + 'shroud.curve', units=units, degree_spline=3,
                        rescale_factor=rescale_factor, x_ref=x_ref)
block = Grid.src.Block(hub, shroud, nstream=nstream, nspan=nspan)
block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
block.extend_inlet_outlet_curves()
block.find_intersections()
block.outlet_zone_trim(mode='axial')
block.spline_of_hub_shroud()
block.spline_of_inlet()
block.sample_hub_shroud()
block.sample_inlet()
block.compute_grid_points(grid_mode='elliptic', orthogonality=True, x_stretching='sigmoid_left', y_stretching='sigmoid',
                          sigmoid_coeff_x=sigmoid_coeff_stream, sigmoid_coeff_y=sigmoid_coeff_span,
                          inlet_meridional_obj=blade_process)
block.compute_grid_centers()

outlet_process = Grid.src.MeridionalProcess(data, block=block, blade=blade, verbose=True)
outlet_process.compute_streamline_length()
outlet_process.circumferential_average(gauss_filter=False)
outlet_process.compute_regressed_fields()
outlet_process.compute_derived_quantities()
outlet_process.compute_averaged_fluxes()

# outlet_process.contour_plot(field='rho')
# outlet_process.contour_plot(field='ur')
# outlet_process.plot_averaged_fluxes(field='p')
# outlet_process.plot_averaged_fluxes(field='s')
# outlet_process.plot_averaged_fluxes(field='T')
delattr(outlet_process, 'data')







#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ASSEMBLY PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("\nASSEMBLY PROCESSING...")
obj = Grid.src.meridional_process_group.MeridionalProcessGroup()
obj.add_to_group(inlet_process)
obj.add_to_group(blade_process)
obj.add_to_group(outlet_process)
obj.assemble_fields()
obj.show_grid()
obj.gauss_filtering()
obj.gauss_filtering()
obj.assemble_field_gradients()
obj.gauss_filtering_gradients()
obj.contour_field_gradients()
obj.plot_averaged_fluxes(field='rho')
obj.plot_averaged_fluxes(field='ur')
obj.plot_averaged_fluxes(field='ut')
obj.plot_averaged_fluxes(field='uz')
obj.plot_averaged_fluxes(field='p')
obj.plot_averaged_fluxes(field='T')
obj.plot_averaged_fluxes(field='s')
obj.plot_averaged_fluxes(field='p_tot')
obj.plot_averaged_fluxes(field='T_tot')
obj.plot_averaged_fluxes(field='M')
obj.plot_averaged_fluxes(field='M_rel')

obj.compute_performance()
obj.print_performance()







end_time = time.time()
delta_time = end_time-start_time
print('Total time: %d sec' % (delta_time))
plt.show()
