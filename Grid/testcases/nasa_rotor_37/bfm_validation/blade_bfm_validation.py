import time
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import Grid
from Grid.src.config import Config

start_time = time.time()
print('Start execution:')
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
configuration_file = 'nasa_rotor_37.ini'
picture_prefix_names = configuration_file.split('.')[0]
config = Config(configuration_file)
BLADE_BLOCK = True


blade = Grid.src.Blade(config)
blade.find_inlet_points()
blade.find_outlet_points()

data = Grid.src.CfdData(config, blade)
data.process_from_ansys_csv()

strwise_pts = config.get_streamwise_points()
spwise_pts = config.get_spanwise_points()


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
blade.compute_blade_blockage(36, save_filename='nasar37')
blade.show_blade_angles_contour(save_filename='nasar37')

blade_process = Grid.src.MeridionalProcess(config, data, bladed_block, blade=blade)
blade_process.compute_camber_angles()
blade_process.compute_streamline_length()
blade_process.compute_spanwise_length()
blade_process.interpolate_on_working_grid()
blade_process.compute_derived_quantities()
blade_process.compute_bfm_axial(save_fig=True)
# blade_process.compute_body_fource_S('rotor')
# blade_process.compute_averaged_fluxes()
blade_process.compute_mass_flow_rate()
blade_process.compute_mass_flow_in_out()
blade_process.check_mass_flow_streamwise()
blade_process.check_bfm_local()
blade_process.check_bfm_global()






# blade_process.plot_stream_line_superposed('F_turn', [4, 20, 36], save_filename='nasar37_Fturn')
# blade_process.plot_stream_line_superposed('F_loss', [4, 20, 36], save_filename='nasar37_Floss')
# blade_process.plot_span_line_superposed('F_loss', [15, 25], save_filename='nasar37_Floss')
# blade_process.contour_all_plots(save_filename='field')

plt.show()
