import time
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import Grid
from Grid.src.config import Config
from Grid.src.functions import create_folder

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% USER INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
configuration_file = 'nasa_lscc.ini'
folder_out = 'pictures'

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
start_time = time.time()
print('Start execution:')
create_folder(folder_out)
picture_prefix_names = configuration_file.split('.')[0]
config = Config(configuration_file)
INLET_BLOCK = True
BLADE_BLOCK = True
OUTLET_BLOCK = True


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BLADE GEO AND CFD DATA READING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
blade = Grid.src.Blade(config)
blade.find_inlet_points()
blade.find_outlet_points()

data = Grid.src.CfdData(config, blade)
data.compute_derived_quantities()

strwise_pts = config.get_streamwise_points()
spwise_pts = config.get_spanwise_points()

block_counter = 0
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INLET BLOCKPROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if INLET_BLOCK:
    print("\nINLET BLOCK PROCESSING...")
    block = Grid.src.Block(config, nstream=config.get_streamwise_points()[0], nspan=config.get_spanwise_points())
    block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
    block.extend_inlet_outlet_curves()
    block.find_intersections()
    block.trim_inlet(z_trim=-0.20373/config.get_reference_length())
    block.inlet_zone_trim(mode=config.get_blade_inlet_type())
    block.spline_of_hub_shroud()
    block.spline_of_outlet()
    block.sample_hub_shroud()
    block.sample_inlet_outlet()
    block.compute_grid_points(block_counter)
    inlet_process = Grid.src.MeridionalProcess(config, data, block)
    inlet_process.compute_streamline_length()
    inlet_process.interpolate_on_working_grid()
    inlet_process.compute_regressed_fields_chebyshev()
    inlet_process.compute_derived_quantities()
    inlet_process.compute_averaged_fluxes()
    inlet_process.compute_body_fource_S(config.get_blocks_type()[0])


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BLADE BLOCK PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
block_counter+=1
if BLADE_BLOCK:
    print("\nBLADE BLOCK PROCESSING...")
    bladed_block = Grid.src.Block(config, nstream=config.get_streamwise_points()[1], nspan=config.get_spanwise_points())
    bladed_block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
    bladed_block.extend_inlet_outlet_curves()
    bladed_block.find_intersections()
    bladed_block.bladed_zone_trim(machine_type='radial')
    bladed_block.spline_of_hub_shroud()
    bladed_block.spline_of_leading_trailing_edge()
    bladed_block.sample_hub_shroud()
    bladed_block.sample_inlet_outlet()
    bladed_block.compute_grid_points(block_counter)
    bladed_block.plot_full_grid(save_filename='blade_grid', save_foldername=folder_out)

    blade.find_camber_surface(bladed_block)
    blade.find_ss_surface(bladed_block)
    blade.find_ps_surface(bladed_block)
    blade.plot_camber_surface()
    blade.compute_camber_vectors()
    blade.show_normal_vectors()
    blade.show_streamline_vectors()
    blade.show_spanline_vectors()
    blade.compute_blade_camber_angles()
    blade.compute_blade_thickness(save_filename='nasa_lscc', folder_name=folder_out)
    blade.compute_blade_blockage(36, save_filename='nasa_lscc', folder_name=folder_out)
    blade.show_blade_angles_contour(save_filename='nasa_lscc', folder_name=folder_out)

    blade_process = Grid.src.MeridionalProcess(config, data, bladed_block, blade=blade)
    blade_process.compute_camber_angles()
    blade_process.compute_streamline_length()
    blade_process.compute_spanwise_length()
    blade_process.interpolate_on_working_grid()
    blade_process.compute_regressed_fields_chebyshev()
    blade_process.compute_derived_quantities()
    blade_process.contour_entropy_generation()
    blade_process.compute_bfm_axial(save_fig=True)
    blade_process.compute_body_fource_S('rotor')
    blade_process.compute_averaged_fluxes()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTLET BLOCK PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
block_counter+=1
if OUTLET_BLOCK:
    print("\nOUTLET BLOCK PROCESSING...")
    block = Grid.src.Block(config, nstream=config.get_streamwise_points()[2], nspan=config.get_spanwise_points())
    block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
    block.extend_inlet_outlet_curves()
    block.find_intersections()
    block.outlet_zone_trim(mode=config.get_blade_outlet_type())
    block.trim_outlet(r_trim=1.67 / config.get_reference_length())
    block.spline_of_hub_shroud()
    block.spline_of_inlet()
    block.sample_hub_shroud()
    block.sample_inlet_outlet()
    block.compute_grid_points(block_counter)

    outlet_process = Grid.src.MeridionalProcess(config, data, block, blade=blade)
    outlet_process.compute_streamline_length()
    outlet_process.compute_spanwise_length()
    outlet_process.interpolate_on_working_grid()
    outlet_process.compute_regressed_fields_chebyshev()
    outlet_process.compute_derived_quantities()
    outlet_process.compute_averaged_fluxes()
    outlet_process.compute_body_fource_S('unbladed')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ASSEMBLY PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if INLET_BLOCK and BLADE_BLOCK and OUTLET_BLOCK:
    print("\nASSEMBLY PROCESSING...")
    obj = Grid.src.meridional_process_group.MeridionalProcessGroup(config)
    obj.add_to_group(inlet_process)
    obj.add_to_group(blade_process)
    obj.add_to_group(outlet_process)
    obj.assemble_fields()
    obj.assemble_field_gradients()
    obj.assemble_body_force_fields()
    # if config.get_shock_smoothing:
    #     obj.shock_smoothing(INLET_NZ - 1)
    obj.compute_streamline_length()
    obj.show_grid(save_filename=config.picture_name_template, folder_name=folder_out)
    obj.contour_fields(save_filename=config.picture_name_template, folder_name=folder_out)
    obj.contour_field_gradients(save_filename=config.picture_name_template, folder_name=folder_out)
    obj.compute_performance()
    obj.print_performance()
    obj.store_pickle(file_name=config.picture_name_template+'_design')
    obj.print_memory_info()

end_time = time.time()
delta_time = end_time - start_time
print('Total time: %d sec' % (delta_time))

# plt.show()
