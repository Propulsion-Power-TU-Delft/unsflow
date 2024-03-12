import time
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import Grid
from Grid.src.config import Config
from Grid.src.functions import create_folder

start_time = time.time()
print('Start execution:')
folder_out = 'pictures'
create_folder(folder_out)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
configuration_file = 'nasa_rotor_37.ini'
picture_prefix_names = configuration_file.split('.')[0]
config = Config(configuration_file)
INLET_BLOCK = True
BLADE_BLOCK = True
OUTLET_BLOCK = True

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BLADE GEO AND CFD DATA READING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
blade = Grid.src.Blade(config)
blade.find_inlet_points()
blade.find_outlet_points()

data = Grid.src.CfdData(config, blade)
data.process_from_ansys_csv()

strwise_pts = config.get_streamwise_points()
spwise_pts = config.get_spanwise_points()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INLET BLOCK PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    block.compute_double_grid()
    block.compute_total_area()

    inlet_process = Grid.src.MeridionalProcess(config, data, block)
    inlet_process.compute_streamline_length()
    inlet_process.interpolate_on_working_grid()
    inlet_process.compute_regressed_fields()
    inlet_process.compute_derived_quantities()
    inlet_process.compute_averaged_fluxes()
    inlet_process.compute_body_fource_S(config.get_blocks_type()[0])
    inlet_process.compute_body_force_residuals()
    inlet_process.override_baseflow()
    inlet_process.contour_plot(field='uz', quiver=True)

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
    bladed_block.compute_double_grid()
    bladed_block.compute_total_area()

    blade.find_camber_surface(bladed_block)
    blade.find_ss_surface(bladed_block)
    blade.find_ps_surface(bladed_block)
    blade.plot_camber_surface()
    blade.compute_camber_vectors()
    blade.compute_blade_camber_angles()
    blade.compute_blade_thickness()
    blade.compute_blade_blockage(36, save_filename='nasar37', folder_name=folder_out)
    blade.show_blade_angles_contour(save_filename='nasar37', folder_name=folder_out)

    blade_process = Grid.src.MeridionalProcess(config, data, bladed_block, blade=blade)
    blade_process.compute_camber_angles()
    blade_process.compute_streamline_length()
    blade_process.compute_spanwise_length()
    blade_process.interpolate_on_working_grid()
    blade_process.compute_regressed_fields()
    blade_process.contour_plot(field='rho')
    blade_process.compute_derived_quantities()
    blade_process.compute_bfm_axial(save_fig=True, mode='averaged')
    blade_process.compute_body_fource_S('rotor')
    blade_process.compute_averaged_fluxes()
    delattr(blade_process, 'data')
    blade_process.compute_body_force_residuals()
    blade_process.override_baseflow()
    blade_process.contour_plot(field='uz', quiver=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTLET BLOCK PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if OUTLET_BLOCK:
    print("\nOUTLET BLOCK PROCESSING...")
    block = Grid.src.Block(config, nstream=config.get_streamwise_points()[2], nspan=config.get_spanwise_points())
    block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
    block.extend_inlet_outlet_curves()
    block.find_intersections()
    block.outlet_zone_trim(mode=config.get_blade_outlet_type())
    block.spline_of_hub_shroud()
    block.spline_of_inlet()
    block.sample_hub_shroud()
    block.sample_inlet_outlet()
    block.compute_grid_points()
    block.compute_double_grid()
    block.compute_total_area()

    outlet_process = Grid.src.MeridionalProcess(config, data, block, blade=blade)
    outlet_process.compute_streamline_length()
    outlet_process.compute_spanwise_length()
    outlet_process.interpolate_on_working_grid()
    outlet_process.compute_regressed_fields()
    outlet_process.compute_derived_quantities()
    outlet_process.compute_averaged_fluxes()
    outlet_process.compute_body_fource_S('unbladed')
    outlet_process.compute_body_force_residuals()
    delattr(outlet_process, 'data')
    outlet_process.override_baseflow()
#
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
    obj.compute_streamline_length()
    obj.show_grid(save_filename=config.picture_name_template, folder_name=folder_out)

    obj.contour_fields(save_filename=config.picture_name_template, folder_name=folder_out)
    obj.contour_field_gradients(save_filename=config.picture_name_template, folder_name=folder_out)
    obj.compute_performance()
    obj.print_performance()
    obj.store_pickle(file_name=config.get_cfd_filepath().split("/")[-1].split('.')[0]+'_%i_%i_%i_%i'
                               %(strwise_pts[0], strwise_pts[1], strwise_pts[2], spwise_pts))
    obj.print_memory_info()

end_time = time.time()
delta_time = end_time - start_time
print('Total time: %d sec' % delta_time)


plt.show()
