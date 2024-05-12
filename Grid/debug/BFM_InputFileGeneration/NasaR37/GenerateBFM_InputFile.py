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
configuration_file = 'NasaR37.ini'
config = Config(configuration_file)
BLADE_BLOCK = True

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BLADE GEO AND CFD DATA READING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
blade = Grid.src.Blade(config)
blade.find_inlet_points()
blade.find_outlet_points()

strwise_pts = config.get_streamwise_points()
spwise_pts = config.get_spanwise_points()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BLADE BLOCK PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bladed_block = Grid.src.Block(config, nstream=config.get_streamwise_points()[1], nspan=config.get_spanwise_points())
bladed_block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
bladed_block.extend_inlet_outlet_curves()
bladed_block.find_intersections()
bladed_block.bladed_zone_trim(machine_type='axial')
bladed_block.spline_of_hub_shroud()
bladed_block.spline_of_leading_trailing_edge()
bladed_block.sample_hub_shroud()
bladed_block.sample_inlet_outlet()
bladed_block.compute_grid_points(block_counter=1)
bladed_block.compute_dual_grid()
bladed_block.compute_total_area()

blade.find_camber_surface(bladed_block)
blade.find_ss_surface(bladed_block)
blade.find_ps_surface(bladed_block)
blade.plot_camber_surface()
blade.plot_camber_meridional_grid()
blade.compute_camber_vectors()
blade.plot_camber_normal_contour(folder_name=folder_out, save_filename='normal')
# blade.compute_blade_camber_angles()
blade.compute_blade_thickness()
blade.compute_blade_blockage(config.get_blades_number(), save_filename='nasar37', folder_name=folder_out)
blade.write_bfm_input_file()





# blade.show_blade_angles_contour(save_filename='nasar37', folder_name=folder_out)

#
# blade_process = Grid.src.MeridionalProcess(config, data, bladed_block, blade=blade)
# blade_process.compute_camber_angles()
#
# blade_process.compute_streamline_length()
# blade_process.compute_spanwise_length()
# blade_process.interpolate_on_working_grid()
# blade_process.compute_derived_quantities()
# blade_process.compute_bfm_axial(save_fig=True, mode='averaged')
# blade_process.compute_body_fource_S('rotor')
# blade_process.compute_averaged_fluxes()
# delattr(blade_process, 'data')
# blade_process.compute_body_force_residuals()



# plt.show()
