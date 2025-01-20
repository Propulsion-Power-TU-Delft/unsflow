from Grid.src.blade import Blade
from Grid.src.block import Block
from Grid.src.config import Config
from Grid.src.bfm_writer import BFM_Writer
import matplotlib.pyplot as plt

configuration_file = 'input.ini'
config = Config(configuration_file)

block_counter = 0
blade_counter = 0

blade = Blade(config, iblade=blade_counter, iblock=block_counter)
blade.compute_thickness()
blade.compute_blade_blockage_on_camber_loft()
blade.compute_normal_vectors_on_reference_surface()
blade.plot_camber_normal_contour_on_loft()
blade.find_inlet_points()
blade.find_outlet_points()

bladed_block = Block(config, iblade=blade_counter, iblock=block_counter)
bladed_block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
bladed_block.extend_inlet_outlet_curves()
bladed_block.find_intersections()
bladed_block.bladed_zone_trim()
bladed_block.spline_of_hub_shroud()
bladed_block.spline_of_leading_trailing_edge()
bladed_block.sample_hub_shroud()
bladed_block.sample_inlet_outlet()
bladed_block.compute_grid_points()
bladed_block.plot_full_grid(save_filename='blade_grid')

blade.add_meridional_grid(bladed_block.z_grid_cg, bladed_block.r_grid_cg)
blade.compute_streamline_length()
blade.plot_streamline_length_contour(save_filename='HECC')
blade.compute_spanline_length()
blade.plot_spanline_length_contour(save_filename='HECC')
blade.obtain_quantities_on_meridional_grid()
blade.compute_camber_vectors()
blade.plot_camber_surface()
blade.plot_camber_normal_contour(save_filename='HECC')
blade.plot_blockage_contour(save_filename='HECC')
blade.compute_blade_camber_angles()
blade.show_blade_angles_contour()
blade.compute_streamline_length()
blade.compute_spanline_length()
blade.plot_inlet_outlet_metal_angle()
blade.compute_blade_blockage_gradient(save_filename='nasaHECC')
blade.plot_blockage_and_grad_leading_to_trailing(save_filename='nasaHECC')

plt.show()
