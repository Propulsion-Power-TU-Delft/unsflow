import matplotlib.pyplot as plt
from Grid.src.blade import Blade
from Grid.src.block import Block
from Grid.src.config import Config

configuration_file = 'input.ini'
config = Config(configuration_file)

blade = Blade(config, iblock=0, iblade=0)
# blade.compute_thickness()
# blade.compute_blade_blockage_on_camber_loft()
# blade.compute_normal_vectors_on_reference_surface()
# blade.plot_camber_normal_contour_on_loft()
blade.find_inlet_points()
blade.find_outlet_points()

bladed_block = Block(config, iblock=0, iblade=0)
bladed_block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
bladed_block.extend_inlet_outlet_curves()
bladed_block.find_intersections()
bladed_block.bladed_zone_trim()
bladed_block.spline_of_hub_shroud()
bladed_block.spline_of_leading_trailing_edge()
bladed_block.sample_hub_shroud()
bladed_block.sample_inlet_outlet()
bladed_block.compute_grid_points()
bladed_block.plot_full_grid(save_filename='NasaR37')

blade.add_meridional_grid(bladed_block.z_grid_cg, bladed_block.r_grid_cg)
blade.compute_streamline_length()
blade.plot_streamline_length_contour(save_filename='NasaR37')
blade.compute_spanline_length()
blade.plot_spanline_length_contour(save_filename='NasaR37')
blade.obtain_quantities_on_meridional_grid_secondversion()
blade.compute_camber_vectors()
# blade.plot_camber_surface()
blade.plot_camber_normal_contour(save_filename='NasaR37')
blade.plot_blockage_contour(save_filename='NasaR37')
blade.compute_blade_camber_angles()
blade.show_blade_angles_contour(save_filename='NasaR37')
blade.plot_inlet_outlet_metal_angle(save_filename='NasaR37')

plt.show()