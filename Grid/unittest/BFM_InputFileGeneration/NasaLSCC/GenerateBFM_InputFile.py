import time
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import Grid
from grid.src.config import Config
from grid.src.functions import create_folder

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% USER INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
configuration_file = 'NasaLSCC.ini'
folder_out = 'pictures'
config = Config(configuration_file)
blade = grid.src.Blade(config)
blade.find_inlet_points()
blade.find_outlet_points()

strwise_pts = config.get_streamwise_points()
spwise_pts = config.get_spanwise_points()


bladed_block = grid.src.Block(config, nstream=config.get_streamwise_points()[1], nspan=config.get_spanwise_points())
bladed_block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
bladed_block.extend_inlet_outlet_curves()
bladed_block.find_intersections()
bladed_block.bladed_zone_trim(machine_type='radial')
bladed_block.spline_of_hub_shroud()
bladed_block.spline_of_leading_trailing_edge()
bladed_block.sample_hub_shroud()
bladed_block.sample_inlet_outlet()
bladed_block.compute_grid_points(1)
bladed_block.plot_full_grid(save_filename='blade_grid', save_foldername=folder_out)

blade.find_camber_surface(bladed_block)
blade.find_ss_surface(bladed_block)
blade.find_ps_surface(bladed_block)
blade.compute_streamline_length()
blade.plot_streamline_length_contour(folder_name=folder_out, save_filename='streamline_length')
blade.plot_camber_surface()
blade.plot_camber_meridional_grid()
blade.compute_camber_vectors(fix='plus')
blade.plot_camber_normal_contour(folder_name=folder_out, save_filename='normal')
blade.compute_blade_thickness()
blade.compute_blade_blockage(config.get_blades_number(), save_filename='nasar37', folder_name=folder_out)
blade.plot_bladetoblade_profile(span='all', folder_name=folder_out, save_filename='blade_profile')
blade.write_bfm_input_file()


# plt.show()
