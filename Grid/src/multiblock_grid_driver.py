#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:29:29 2023
@author: F. Neri, TU Delft
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from Utils.styles import *
from Grid.src.blade import Blade
from Grid.src.block import Block

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

            
        # return self.blades, self.blocks
    
    
    def ReconstructBlock(self, iblock):
        """_summary_

        Args:
            iblock (_type_): _description_
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
        
        
        
        
        
        
    
    def ReconstructBlade(self, iblade, iblock):
        """
        Reconstruct the blades
        """
        blade = Blade(self.config, iblock=iblock, iblade=iblade)
        blade.find_inlet_points()
        blade.find_outlet_points()
        return blade        
        
        

    