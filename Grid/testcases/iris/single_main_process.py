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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MESH_TYPE = 'default'
REGRESSION = True
INLET_NZ = 20
BLADE_NZ = 40
OUTLET_NZ = 30
NR = 30
AVG_MODE = 'cell centered'
file_name = 'data/meta/85krpm_0.11kgs_meridional.csv'
MULTIBLOCK_FILTERING = False
SHOCK_SMOOTHING = False
INTERP_METHOD = 'nearest'
INLET_BLOCK = True
BLADE_BLOCK = True
OUTLET_BLOCK = True





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GLOBAL DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_folder_path = 'data/geo/'
units = '[m]'
rho_ref = 1.014  # reference density [kg/m3]
x_ref = 0.0228  # reference length, tip radius [m]
rpm_ref = +94e3  # shaft rpm with sign
T_ref = 288.15  # reference temperature [K]
rescale_factor = 1  # cordinates of data files are in [m]
sigmoid_coeff_stream = 8
sigmoid_coeff_span = 8



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BLADE INFO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# compute the blade object info, in order to cut the block appropriately
blade = Grid.src.Blade(data_folder_path + 'iris_blade.curve', rescale_factor=rescale_factor, x_ref=x_ref)
blade.find_inlet_points(geometry_type='axial')
blade.find_outlet_points(geometry_type='axial')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CFD INFO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# instantiate cfd data object and perform processing removing the outliers
data = Grid.src.CfdData(file_name, rpm_drag=rpm_ref, blade=blade, verbose=True, normalize=True,
                        rho_ref=rho_ref, x_ref=x_ref, T_ref=T_ref, file_type='Ansys2D')
data.process_from_ansys_csv()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INLET PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if INLET_BLOCK:
    print("\nINLET BLOCK PROCESSING...")
    nstream = INLET_NZ
    nspan = NR

    hub = Grid.src.Curve(curve_filepath=data_folder_path + 'iris_hub.curve', units=units, degree_spline=1,
                         rescale_factor=rescale_factor, x_ref=x_ref)
    shroud = Grid.src.Curve(curve_filepath=data_folder_path + 'iris_shroud.curve', units=units, degree_spline=1,
                            rescale_factor=rescale_factor, x_ref=x_ref)
    block = Grid.src.Block(hub, shroud, nstream=nstream, nspan=nspan)

    # compute the blade object info, in order to cut the block appropriately


    # cut the bladed block properly, and compute the meridional structured mesh
    block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
    block.extend_inlet_outlet_curves()
    block.find_intersections()
    block.inlet_zone_trim(mode='axial')
    block.spline_of_hub_shroud()
    block.spline_of_outlet()
    block.sample_hub_shroud()
    block.sample_inlet_outlet()
    if MESH_TYPE == 'default':
        block.compute_grid_points(grid_mode='elliptic', orthogonality=False,
                                  x_stretching=False, y_stretching=False,
                                  sigmoid_coeff_x=sigmoid_coeff_stream, sigmoid_coeff_y=sigmoid_coeff_span)
    else:
        block.compute_grid_points(grid_mode='elliptic', orthogonality=True,
                                  x_stretching='sigmoid_right', y_stretching='sigmoid',
                                  sigmoid_coeff_x=sigmoid_coeff_stream, sigmoid_coeff_y=sigmoid_coeff_span)
    block.compute_grid_centers()

    # instantiate meridional process object and avg
    inlet_process = Grid.src.MeridionalProcess(data, block=block, verbose=True, GAMMA=1.4)
    inlet_process.compute_streamline_length()
    inlet_process.interpolate_on_working_grid(method=INTERP_METHOD)
    if REGRESSION:
        inlet_process.compute_regressed_fields()
    else:
        inlet_process.compute_field_gradients()
    inlet_process.compute_derived_quantities()  # to recompute the derived quantities based on the regressed values
    inlet_process.compute_averaged_fluxes()
    inlet_process.compute_body_fource_S('unbladed')
    delattr(inlet_process, 'data')  # release useless memory






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BLADE PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if BLADE_BLOCK:
    print("\nBLADE BLOCK PROCESSING...")
    nstream = BLADE_NZ
    nspan = NR
    hub = Grid.src.Curve(curve_filepath=data_folder_path + 'iris_hub.curve', units=units, degree_spline=3,
                         rescale_factor=rescale_factor, x_ref=x_ref)
    shroud = Grid.src.Curve(curve_filepath=data_folder_path + 'iris_shroud.curve', units=units, degree_spline=3,
                            rescale_factor=rescale_factor, x_ref=x_ref)
    bladed_block = Grid.src.Block(hub, shroud, nstream=nstream, nspan=nspan)

    # cut the bladed block properly, and compute the meridional structured mesh
    bladed_block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
    bladed_block.extend_inlet_outlet_curves()
    bladed_block.find_intersections()
    bladed_block.bladed_zone_trim(machine_type='radial')
    bladed_block.spline_of_hub_shroud()
    bladed_block.spline_of_leading_trailing_edge()
    bladed_block.sample_hub_shroud()
    bladed_block.sample_inlet_outlet()
    if MESH_TYPE == 'default':
        try:
            bladed_block.compute_grid_points(grid_mode='elliptic', orthogonality=False, x_stretching=False, y_stretching=False,
                                             sigmoid_coeff_x=sigmoid_coeff_stream, sigmoid_coeff_y=sigmoid_coeff_span,
                                             inlet_meridional_obj=inlet_process)
        except:
            bladed_block.compute_grid_points(grid_mode='elliptic', orthogonality=False, x_stretching=False, y_stretching=False,
                                             sigmoid_coeff_x=sigmoid_coeff_stream, sigmoid_coeff_y=sigmoid_coeff_span)
    else:
        try:
            bladed_block.compute_grid_points(grid_mode='elliptic', orthogonality=True, x_stretching='sigmoid',
                                             y_stretching='sigmoid',
                                             sigmoid_coeff_x=sigmoid_coeff_stream, sigmoid_coeff_y=sigmoid_coeff_span,
                                             inlet_meridional_obj=inlet_process)
        except:
            bladed_block.compute_grid_points(grid_mode='elliptic', orthogonality=True, x_stretching='sigmoid',
                                             y_stretching='sigmoid',
                                             sigmoid_coeff_x=sigmoid_coeff_stream, sigmoid_coeff_y=sigmoid_coeff_span)
    bladed_block.compute_grid_centers()

    blade.find_camber_surface(bladed_block)
    blade.compute_camber_vectors()
    blade.compute_blade_camber_angles(convention='rotation-wise')

    # instantiate meridional process object and avg
    blade_process = Grid.src.MeridionalProcess(data, block=bladed_block, blade=blade, verbose=True)
    blade_process.compute_camber_angles()
    blade_process.compute_streamline_length()
    blade_process.interpolate_on_working_grid(method=INTERP_METHOD)
    if REGRESSION:
        blade_process.compute_regressed_fields()
    else:
        blade_process.compute_field_gradients()
    blade_process.compute_derived_quantities()
    blade_process.compute_bfm_axial(mode='global', save_fig=True)
    blade_process.compute_body_fource_S('rotor')
    blade_process.compute_averaged_fluxes()
    blade_process.contour_plot('s')
    delattr(blade_process, 'data')







#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTLET PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if OUTLET_BLOCK:
    print("\nOUTLET BLOCK PROCESSING...")
    nstream = OUTLET_NZ
    nspan = NR
    hub = Grid.src.Curve(curve_filepath=data_folder_path + 'iris_hub.curve', units=units, degree_spline=3,
                         rescale_factor=rescale_factor, x_ref=x_ref)
    shroud = Grid.src.Curve(curve_filepath=data_folder_path + 'iris_shroud.curve', units=units, degree_spline=1,
                            rescale_factor=rescale_factor, x_ref=x_ref)
    block = Grid.src.Block(hub, shroud, nstream=nstream, nspan=nspan)
    block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
    block.extend_inlet_outlet_curves()
    block.find_intersections()
    block.outlet_zone_trim(mode='radial')
    block.spline_of_hub_shroud()
    block.spline_of_inlet()
    block.sample_hub_shroud()
    block.sample_inlet_outlet()
    if MESH_TYPE == 'default':
        try:
            block.compute_grid_points(grid_mode='elliptic', orthogonality=False, x_stretching=False, y_stretching=False,
                                      sigmoid_coeff_x=sigmoid_coeff_stream, sigmoid_coeff_y=sigmoid_coeff_span,
                                      inlet_meridional_obj=blade_process)
        except:
            block.compute_grid_points(grid_mode='elliptic', orthogonality=False, x_stretching=False, y_stretching=False,
                                      sigmoid_coeff_x=sigmoid_coeff_stream, sigmoid_coeff_y=sigmoid_coeff_span)
    else:
        try:
            block.compute_grid_points(grid_mode='elliptic', orthogonality=True, x_stretching='sigmoid_left',
                                      y_stretching='sigmoid',
                                      sigmoid_coeff_x=sigmoid_coeff_stream, sigmoid_coeff_y=sigmoid_coeff_span,
                                      inlet_meridional_obj=blade_process)
        except:
            block.compute_grid_points(grid_mode='elliptic', orthogonality=True, x_stretching='sigmoid_left',
                                      y_stretching='sigmoid',
                                      sigmoid_coeff_x=sigmoid_coeff_stream, sigmoid_coeff_y=sigmoid_coeff_span)
    block.compute_grid_centers()

    outlet_process = Grid.src.MeridionalProcess(data, block=block, blade=blade, verbose=True)
    outlet_process.compute_streamline_length()
    outlet_process.interpolate_on_working_grid(method=INTERP_METHOD)
    # outlet_process.c_plot('s')
    if REGRESSION:
        outlet_process.compute_regressed_fields()
    else:
        outlet_process.compute_field_gradients()
    outlet_process.compute_derived_quantities()
    outlet_process.compute_averaged_fluxes()
    outlet_process.compute_body_fource_S('unbladed')
    delattr(outlet_process, 'data')







#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ASSEMBLY PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if INLET_BLOCK and BLADE_BLOCK and OUTLET_BLOCK:
    print("\nASSEMBLY PROCESSING...")
    obj = Grid.src.meridional_process_group.MeridionalProcessGroup()
    obj.add_to_group(inlet_process)
    obj.add_to_group(blade_process)
    obj.add_to_group(outlet_process)
    obj.assemble_fields()
    obj.assemble_field_gradients()
    obj.assemble_body_force_fields()
    if SHOCK_SMOOTHING:
        obj.shock_smoothing(INLET_NZ-1)
    if MULTIBLOCK_FILTERING:
        obj.gauss_filtering()
        obj.gauss_filtering_gradients()
    obj.compute_streamline_length()
    obj.show_grid()

    obj.contour_fields(save_filename='filt_%s_%i_%i_%i_%i'
                                     % (MULTIBLOCK_FILTERING, INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))
    obj.contour_field_gradients(save_filename='filt_%s_%i_%i_%i_%i'
                                              % (MULTIBLOCK_FILTERING, INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))
    obj.plot_averaged_fluxes(field='rho', save_filename='flux_rho_filt_%s_%i_%i_%i_%i'
                                                        % (MULTIBLOCK_FILTERING, INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))
    obj.plot_averaged_fluxes(field='ur', save_filename='flux_ur_filt_%s_%i_%i_%i_%i'
                                                       % (MULTIBLOCK_FILTERING, INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))
    obj.plot_averaged_fluxes(field='ut', save_filename='flux_ut_filt_%s_%i_%i_%i_%i'
                                                       % (MULTIBLOCK_FILTERING, INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))
    obj.plot_averaged_fluxes(field='uz', save_filename='flux_uz_filt_%s_%i_%i_%i_%i'
                                                       % (MULTIBLOCK_FILTERING, INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))
    obj.plot_averaged_fluxes(field='p', save_filename='flux_p_filt_%s_%i_%i_%i_%i'
                                                      % (MULTIBLOCK_FILTERING, INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))
    obj.plot_averaged_fluxes(field='T', save_filename='flux_T_filt_%s_%i_%i_%i_%i'
                                                      % (MULTIBLOCK_FILTERING, INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))
    obj.plot_averaged_fluxes(field='s', save_filename='flux_s_filt_%s_%i_%i_%i_%i'
                                                      % (MULTIBLOCK_FILTERING, INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))
    obj.plot_averaged_fluxes(field='p_tot', save_filename='flux_p_tot_filt_%s_%i_%i_%i_%i'
                                                          % (MULTIBLOCK_FILTERING, INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))
    obj.plot_averaged_fluxes(field='T_tot', save_filename='flux_T_tot_filt_%s_%i_%i_%i_%i'
                                                          % (MULTIBLOCK_FILTERING, INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))
    obj.plot_averaged_fluxes(field='M', save_filename='flux_M_filt_%s_%i_%i_%i_%i'
                                                      % (MULTIBLOCK_FILTERING, INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))
    obj.plot_averaged_fluxes(field='M_rel', save_filename='flux_M_rel_filt_%s_%i_%i_%i_%i'
                                                          % (MULTIBLOCK_FILTERING, INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))
    obj.compute_performance()
    obj.print_performance()
    obj.compose_global_sun_Omega_tau()
    delattr(obj, 'group')
    obj.store_pickle(file_name='inlet_%i_blade_%i_outlet_%i_nspan_%i' % (INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))

end_time = time.time()
delta_time = end_time - start_time
print('Total time: %d sec' % (delta_time))
