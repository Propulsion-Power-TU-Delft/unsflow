import time
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import Grid
from Grid.src.config import Config
import os
from Grid.src.multiblock import MultiBlock

# SETTINGS
configuration_file = 'input.ini'
config = Config(configuration_file)

blade_counter = 0

# BLADE
blade = Grid.src.Blade(config, iblock=1, iblade=0)
# blade.compute_thickness()
# blade.compute_blade_blockage_on_camber_loft()
# blade.compute_normal_vectors_on_reference_surface()
# blade.plot_camber_normal_contour_on_loft()
blade.find_inlet_points()
blade.find_outlet_points()



# INLET BLOCK
block_counter = 0
print("\nINLET BLOCK PROCESSING...")
block = Grid.src.Block(config, iblock=block_counter)
block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
block.extend_inlet_outlet_curves()
block.find_intersections()
block.inlet_zone_trim(mode=config.get_blade_inlet_type()[blade_counter])
block.spline_of_hub_shroud()
block.spline_of_outlet()
block.sample_hub_shroud()
block.sample_inlet_outlet()
block.compute_grid_points()
block.plot_full_grid()




# BLADE BLOCK
block_counter = 1
print("\nBLADE BLOCK PROCESSING...")
bladed_block = Grid.src.Block(config, iblock=block_counter, iblade=0)
bladed_block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
bladed_block.extend_inlet_outlet_curves()
bladed_block.find_intersections()
bladed_block.bladed_zone_trim()
bladed_block.spline_of_hub_shroud()
bladed_block.spline_of_leading_trailing_edge()
bladed_block.sample_hub_shroud()
bladed_block.sample_inlet_outlet()
bladed_block.compute_grid_points()
bladed_block.plot_full_grid()





# OUTLET BLOCK
block_counter = 2
print("\nOUTLET BLOCK PROCESSING...")
outlet_block = Grid.src.Block(config, iblock=2)
outlet_block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
outlet_block.extend_inlet_outlet_curves()
outlet_block.find_intersections()
outlet_block.outlet_zone_trim(mode=config.get_blade_outlet_type()[blade_counter])
outlet_block.spline_of_hub_shroud()
outlet_block.spline_of_inlet()
outlet_block.sample_hub_shroud()
outlet_block.sample_inlet_outlet()
outlet_block.compute_grid_points()
outlet_block.plot_full_grid()





print("\nASSEMBLY PROCESSING...")
multimesh = MultiBlock(config, block, bladed_block, outlet_block)
multimesh.assemble_grid()
multimesh.plot_full_grid(save_filename='NR37')
multimesh.export_meridional_spline(span=0.1)
multimesh.export_meridional_spline(span=0.3)
multimesh.export_meridional_spline(span=0.5)
multimesh.export_meridional_spline(span=0.7)
multimesh.export_meridional_spline(span=0.9)
# multimesh.plot_blockage(save_filename='NR37')
# multimesh.plot_rpm(save_filename='NR37')
# multimesh.plot_normal_camber(save_filename='NR37')
# multimesh.plot_streamline_length(save_filename='NR37')
# multimesh.write_turbobfm_grid_file_2D(blockage=True, normal=True, rpm=True, stwl=True)


plt.show()