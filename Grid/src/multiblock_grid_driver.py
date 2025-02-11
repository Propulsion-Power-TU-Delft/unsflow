import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from Utils.styles import *
from Grid.src.blade import Blade
from Grid.src.block import Block
from Grid.src.multiblock import MultiBlock

class MultiBlockGridDriver:
    """
    Driver to perform grid generation and everything else related
    """

    def __init__(self, config):
        self.config = config
        self.numberBlades = self.config.get_blade_rows_number()
        self.numberBlocks = self.numberBlades + (self.numberBlades + 1)
        self.blades = []
        self.blocks = []
            
    
    def GenerateGrid(self):
        """
        Generate the grid stacking together the different bladed and unbladed blocks. It computes only the grid points, 
        nothing is interpolated yet on it due to the blades.
        """
        for iblade in range(self.numberBlades):
            iblock = 1 + iblade*2
            blade = self.ReconstructBlade(iblade, iblock)
            self.blades.append(blade)
        
        for iblock in range(self.numberBlocks):
            block = self.ReconstructBlock(iblock)
            self.blocks.append(block)

    
    def ReconstructBlock(self, iblock):
        """
        Reconstruct a block at the specified index.

        This function creates a Block object and configures it based on its position
        in the grid. It adds inlet and outlet curves, extends them, and performs various
        operations to prepare the block for grid computation.

        :param iblock: Index of the block to be reconstructed.
        :return: A configured Block object.
        """
        block = Block(self.config, iblock=iblock)
        iblade = int((iblock-1)/2)
        
        if iblock==0:
            # the block is the first one
            block.add_inlet_outlet_curves(block.inletLine, self.blades[iblade].inlet)
        
        elif iblock==self.numberBlocks-1:
            # the block is the last one
            block.add_inlet_outlet_curves(self.blades[iblade].outlet, block.outletLine)
        
        elif (iblock-1)%2==0:
            # the block corresponds the blade iblade
            block.add_inlet_outlet_curves(self.blades[iblade].inlet, self.blades[iblade].outlet)
        
        else:
            # the block is downstream of the blade iblade
            block.add_inlet_outlet_curves(self.blades[iblade].outlet, self.blades[iblade+1].inlet)
        
        block.extend_inlet_outlet_curves()
        block.find_intersections()
        block.bladed_zone_trim()
        block.spline_of_hub_shroud()
        block.spline_of_leading_trailing_edge(iblade)
        block.sample_hub_shroud()
        block.sample_inlet_outlet()
        block.compute_grid_points()
        return block
        
    
    def ReconstructBlade(self, iblade, iblock):
        """
        Reconstructs a blade of the turbomachine and returns it as a Blade
        object. It adds inlet and outlet points to the blade and performs
        various operations to prepare the blade for the computation of
        quantities on its meridional grid.

        :param iblade: Index of the blade to be reconstructed.
        :param iblock: Index of the block to which the blade belongs.
        :return: A configured Blade object.
        """
        blade = Blade(self.config, iblock=iblock, iblade=iblade)
        blade.find_inlet_points()
        blade.find_outlet_points()
        return blade        
    
    
    def ComputeBladesData(self):
        """
        Compute the quantities on the meridional grid of the blades.
        """
        for iblade in range(self.numberBlades):
            iblock = (iblade+1)+iblade
            self.blades[iblade].add_meridional_grid(self.blocks[iblock].z_grid_cg, self.blocks[iblock].r_grid_cg)
            self.blades[iblade].compute_streamline_length()
            self.blades[iblade].compute_spanline_length()
            self.blades[iblade].obtain_quantities_on_meridional_grid_thirdversion()
            self.blades[iblade].compute_camber_vectors()
            self.blocks[iblock].add_blockage_grid(self.blades[iblade].blockage)
            self.blocks[iblock].add_camber_grid(self.blades[iblade].n_camber_z, self.blades[iblade].n_camber_r, self.blades[iblade].n_camber_t)
            self.blocks[iblock].add_streamline_length_grid(self.blades[iblade].streamline_length)
    
    
    def AssembleMultiBlockGrid(self):
        """
        Assemble the grid of the different blocks in a single one.
        """
        self.multiBlockGrid = MultiBlock(self.config, *self.blocks)
        self.multiBlockGrid.assemble_grid()
        self.multiBlockGrid.plot_full_grid(save_filename=self.config.get_machine_name())
        self.multiBlockGrid.write_turbobfm_grid_file_2D()
        
        

    