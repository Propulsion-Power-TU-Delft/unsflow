import time
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import Grid
from Grid.src.config import Config
from Grid.src.functions import create_folder
from Grid.src.multiblock import MultiBlock

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
strwise_pts = config.get_streamwise_points()
spwise_pts = config.get_spanwise_points()

block_counter = 0
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
    block.compute_grid_points(block_counter)
    block.plot_full_grid()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BLADE BLOCK PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
block_counter += 1
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
    bladed_block.compute_grid_points(block_counter)
    bladed_block.plot_full_grid()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTLET BLOCK PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
block_counter += 1
if OUTLET_BLOCK:
    print("\nOUTLET BLOCK PROCESSING...")
    outlet_block = Grid.src.Block(config, nstream=config.get_streamwise_points()[2], nspan=config.get_spanwise_points())
    outlet_block.add_inlet_outlet_curves(blade.inlet, blade.outlet)
    outlet_block.extend_inlet_outlet_curves()
    outlet_block.find_intersections()
    outlet_block.outlet_zone_trim(mode=config.get_blade_outlet_type())
    outlet_block.spline_of_hub_shroud()
    outlet_block.spline_of_inlet()
    outlet_block.sample_hub_shroud()
    outlet_block.sample_inlet_outlet()
    outlet_block.compute_grid_points(block_counter)
    outlet_block.plot_full_grid()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ASSEMBLY PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if INLET_BLOCK and BLADE_BLOCK and OUTLET_BLOCK:
    print("\nASSEMBLY PROCESSING...")
    multimesh = MultiBlock(block, bladed_block, outlet_block)
    multimesh.assemble_grid()
    multimesh.plot_full_grid()
    multimesh.compute_three_dimensional_mesh(10 + 1)
    multimesh.save_mesh_pickle()





plt.show()
