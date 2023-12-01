import time
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import Grid
from pympler import asizeof
from Grid.src.config import Config

start_time = time.time()
print('Start execution:')
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
configuration_file = 'nasa_rotor_37.ini'
config = Config(configuration_file)
INLET_BLOCK = False
BLADE_BLOCK = True
OUTLET_BLOCK = False









# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BLADE GEO AND CFD DATA READING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
blade = Grid.src.Blade(config)
blade.find_inlet_points()
blade.find_outlet_points()

data = Grid.src.CfdData(config, blade)
data.process_from_ansys_csv()








# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INLET BLOCKPROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if INLET_BLOCK:
    print("\nINLET BLOCK PROCESSING...")
    block = Grid.src.Block(config, nstream=config.get_streamwise_points()[0], nspan=config.get_spanwise_points())
    block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
    block.extend_inlet_outlet_curves()
    block.find_intersections()
    block.inlet_zone_trim(mode=config.get_blade_inlet_type())
    block.spline_of_hub_shroud()
    block.spline_of_outlet()
    block.sample_hub_shroud()
    block.sample_inlet_outlet()
    block.compute_grid_points()

    inlet_process = Grid.src.MeridionalProcess(config, data, block)
    inlet_process.compute_streamline_length()
    inlet_process.interpolate_on_working_grid()
    if config.get_standard_regression():
        inlet_process.compute_regressed_fields()
    else:
        inlet_process.compute_field_gradients()
    inlet_process.compute_derived_quantities()
    inlet_process.compute_averaged_fluxes()
    inlet_process.compute_body_fource_S(config.get_blocks_type()[0])
    inlet_process.contour_all_plots()
    delattr(inlet_process, 'data')  # release useless memory





# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BLADE BLOCK PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if BLADE_BLOCK:
    print("\nBLADE BLOCK PROCESSING...")
    bladed_block = Grid.src.Block(config, nstream=config.get_streamwise_points()[1], nspan=config.get_spanwise_points())
    bladed_block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
    bladed_block.extend_inlet_outlet_curves()
    bladed_block.find_intersections()
    bladed_block.bladed_zone_trim(machine_type='axial')
    bladed_block.spline_of_hub_shroud()
    bladed_block.spline_of_leading_trailing_edge()
    bladed_block.sample_hub_shroud()
    bladed_block.sample_inlet_outlet()
    bladed_block.compute_grid_points()

    blade.find_camber_surface(bladed_block)
    blade.compute_camber_vectors()
    blade.compute_blade_camber_angles()
    blade.show_blade_angles_contour()

    # instantiate meridional process object and avg
    blade_process = Grid.src.MeridionalProcess(config, data, bladed_block, blade=blade)
    blade_process.compute_camber_angles()
    blade_process.compute_streamline_length()
    blade_process.interpolate_on_working_grid()
    # blade_process.compute_field_gradients(method=GRAD_METHOD)

    if config.get_standard_regression():
        blade_process.compute_regressed_fields()
    else:
        blade_process.compute_field_gradients()
    blade_process.compute_derived_quantities()
    blade_process.contour_entropy_generation()
    blade_process.compute_bfm_axial()
    blade_process.compute_body_fource_S('rotor')
    blade_process.compute_averaged_fluxes()
    blade_process.contour_all_plots()
    delattr(blade_process, 'data')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTLET BLOCK PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if OUTLET_BLOCK:
    print("\nOUTLET BLOCK PROCESSING...")
    nstream = OUTLET_NZ
    nspan = NR
    hub = Grid.src.Curve(curve_filepath=geo_folder + 'hub.curve', units=units, degree_spline=3,
                         rescale_factor=rescale_factor, x_ref=x_ref)
    shroud = Grid.src.Curve(curve_filepath=geo_folder + 'shroud.curve', units=units, degree_spline=3,
                            rescale_factor=rescale_factor, x_ref=x_ref)
    block = Grid.src.Block(hub, shroud, nstream=nstream, nspan=nspan)
    block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
    block.extend_inlet_outlet_curves()
    block.find_intersections()
    block.outlet_zone_trim(mode='axial')
    block.spline_of_hub_shroud()
    block.spline_of_inlet()
    block.sample_hub_shroud()
    block.sample_inlet_outlet()
    block.compute_grid_points(grid_mode='elliptic', orthogonality=True, x_stretching=MESH_TYPE, y_stretching=MESH_TYPE,
                              sigmoid_coeff_x=sigmoid_coeff_stream, sigmoid_coeff_y=sigmoid_coeff_span)

    outlet_process = Grid.src.MeridionalProcess(data, block=block, blade=blade, verbose=True)
    outlet_process.compute_streamline_length()
    outlet_process.interpolate_on_working_grid(method=INTERP_METHOD)
    # outlet_process.compute_field_gradients(method=GRAD_METHOD)
    if REGRESSION:
        outlet_process.compute_regressed_fields()
    # else:
    #     outlet_process.compute_field_gradients()
    outlet_process.compute_derived_quantities()
    outlet_process.compute_averaged_fluxes()
    outlet_process.compute_body_fource_S('unbladed')
    outlet_process.contour_all_plots()
    delattr(outlet_process, 'data')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ASSEMBLY PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        obj.shock_smoothing(INLET_NZ - 1)
    if MULTIBLOCK_FILTERING:
        # obj.gauss_filtering()
        obj.gauss_filtering_gradients()
    obj.compute_streamline_length()
    obj.show_grid(save_filename='%i_%i_%i_%i' % (INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))

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
    # obj.compose_global_sun_Omega_tau()
    obj.contour_fields()
    obj.contour_field_gradients()
    delattr(obj, 'group')
    obj.store_pickle(file_name='inlet_%i_blade_%i_outlet_%i_nspan_%i' % (INLET_NZ, BLADE_NZ, OUTLET_NZ, NR))


    def print_attribute_sizes(Object):
        tot_size = 0
        for attribute_name in dir(Object):
            attribute = getattr(Object, attribute_name)
            size_in_bytes = sys.getsizeof(attribute)
            tot_size += size_in_bytes
            print(f"Size of {attribute_name}: {size_in_bytes} bytes")
        print(f"Total size: {tot_size} bytes")

    print_attribute_sizes(obj)

end_time = time.time()
delta_time = end_time - start_time
print('Total time: %d sec' % (delta_time))

plt.show()
