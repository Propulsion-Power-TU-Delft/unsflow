import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from unsflow.utils.plot_styles import *
from .blade import Blade
from .block import Block
from .multiblock import MultiBlock
from .su2_mesh_generator import generate_SU2mesh
from unsflow.utils.formatting import print_banner_begin, print_banner_end
from .functions import contour_template, compute_meridional_streamwise_coordinates, compute_meridional_spanwise_coordinates


class MultiBlockGridDriver:
    """
    Driver to perform grid generation and everything else related
    """

    def __init__(self, config):
        self.config = config
        self.numberBlades = self.config.get_blade_rows_number()
        self.numberSplitterBlades = self.config.get_splitter_blade_rows_number()
        
        self.driverType = self.config.get_multiblock_driver_type()
        if self.driverType=='multiblock':
            self.numberBlocks = self.numberBlades + (self.numberBlades + 1)
        elif self.driverType=='full_machine' or self.driverType=='single_blade':
            self.numberBlocks = 1
        else:
            raise ValueError('Multiblock driver type not recognized. Possible options are (multiblock, full_machine, single_blade)')
        self.blades = []
        self.splitter_blades = []
        self.blocks = []
            
    
    def GenerateGrid(self):
        """
        Generate the grid stacking together the different bladed and unbladed blocks. It computes only the grid points, 
        nothing is interpolated yet on it due to the blades.
        """
        if self.driverType!='full_machine':
            for iblade in range(self.numberBlades):
                if self.driverType=='multiblock':
                    iblock = 1 + iblade*2
                else:
                    iblock = 0
                blade = self.ReconstructBlade(iblade, iblock)
                self.blades.append(blade)
                
                if self.config.is_splitter_blade_present(iblade):
                    splitterBlade = self.ReconstructBlade(iblade, iblock, bladeType='splitter')
                    self.splitter_blades.append(splitterBlade)
                    
                
                
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
        
        if self.driverType=='multiblock':
            if iblock==0:
                # the block is the first one
                block.trim_inlet()
                block.add_inlet_outlet_curves(block.inletLine, self.blades[iblade].inlet)
            
            elif iblock==self.numberBlocks-1:
                # the block is the last one
                block.trim_outlet()
                block.add_inlet_outlet_curves(self.blades[iblade].outlet, block.outletLine)
            
            elif (iblock-1)%2==0:
                # the block corresponds the blade iblade
                block.add_inlet_outlet_curves(self.blades[iblade].inlet, self.blades[iblade].outlet)
            
            else:
                # the block is downstream of the blade iblade
                block.add_inlet_outlet_curves(self.blades[iblade].outlet, self.blades[iblade+1].inlet)
        
        elif self.driverType=='single_blade':
            block.add_inlet_outlet_curves(self.blades[iblade].inlet, self.blades[iblade].outlet)
        
        elif self.driverType=='full_machine':
            block.trim_inlet()
            block.trim_outlet()
            block.add_inlet_outlet_curves(block.inletLine, block.outletLine)
        
        block.extend_inlet_outlet_curves()
        block.find_intersections()
        block.internal_zone_trim()
        block.spline_of_hub_shroud()
        block.spline_of_leading_trailing_edge(iblade)
        block.sample_hub_shroud()
        block.sample_inlet_outlet()
        block.compute_grid_points()
        block.streamwise_normalized_grid = compute_meridional_streamwise_coordinates(block.z_grid, block.r_grid, normalize=True)
        block.spanwise_normalized_grid = compute_meridional_spanwise_coordinates(block.z_grid, block.r_grid, normalize=True)
        return block
        
    
    def ReconstructBlade(self, iblade, iblock, bladeType='main'):
        """
        Reconstructs a blade of the turbomachine and returns it as a Blade
        object. It adds inlet and outlet points to the blade and performs
        various operations to prepare the blade for the computation of
        quantities on its meridional grid.

        :param iblade: Index of the blade to be reconstructed.
        :param iblock: Index of the block to which the blade belongs.
        :return: A configured Blade object.
        """
        blade = Blade(self.config, iblock=iblock, iblade=iblade, bladeType=bladeType)
        blade.find_inlet_points()
        blade.find_outlet_points()
        return blade        
    
    
    def ComputeBladesData(self):
        """
        Compute the quantities on the meridional grid of the blades.
        """
        if self.driverType=='full_machine':
            print('In full_machine mode there are no blades specified. You cannot compute any blade related data.')
            return
        
        for iblade in range(self.numberBlades):
            
            # compute the right counter for the blade and block objects
            if self.driverType=='single_blade':
                iblock = 0
            else:
                iblock = (iblade+1)+iblade
            
            # compute the meridional grid
            self.blades[iblade].add_meridional_grid(self.blocks[iblock].z_grid, self.blocks[iblock].r_grid)
            self.blades[iblade].streamwise_normalized_coord = compute_meridional_streamwise_coordinates(self.blocks[iblock].z_grid, self.blocks[iblock].r_grid, normalize=True)
            self.blades[iblade].spanwise_normalized_coord = compute_meridional_spanwise_coordinates(self.blocks[iblock].z_grid, self.blocks[iblock].r_grid, normalize=True)
            self.blades[iblade].streamwise_coord = compute_meridional_streamwise_coordinates(self.blocks[iblock].z_grid, self.blocks[iblock].r_grid, normalize=False)
            self.blades[iblade].spanwise_coord = compute_meridional_spanwise_coordinates(self.blocks[iblock].z_grid, self.blocks[iblock].r_grid, normalize=False)
            self.blades[iblade].plot_meridional_coordinates(save_filename=self.config.get_machine_name() + '_blade_%02i' % iblade)
            
            # compute the thickness of eventual splitter blades (for centrifigual machines)
            iSplitter = 0
            if self.config.is_splitter_blade_present(iblade):
                zSplitterPS = self.splitter_blades[iSplitter].z_psSurface
                rSplitterPS = self.splitter_blades[iSplitter].r_psSurface
                thetaSplitterPS = self.splitter_blades[iSplitter].theta_psSurface
                zSplitterSS = self.splitter_blades[iSplitter].z_ssSurface
                rSplitterSS = self.splitter_blades[iSplitter].r_ssSurface
                thetaSplitterSS = self.splitter_blades[iSplitter].theta_ssSurface
                self.blades[iblade].compute_splitter_thickness((zSplitterPS, rSplitterPS, thetaSplitterPS), (zSplitterSS, rSplitterSS, thetaSplitterSS))
                iSplitter += 1
            
            # interpolate and compute geometrical factors on the meridional grid
            self.blades[iblade].obtain_quantities_on_meridional_grid_thirdversion()
            self.blades[iblade].plot_blockage_contour(save_filename=self.config.get_machine_name() + '_blade_%02i' % iblade)
            self.blades[iblade].compute_blade_camber_angles()
            self.blades[iblade].compute_blade_camber_curvature()
            self.blades[iblade].plot_camber_normal_contour(save_filename=self.config.get_machine_name() + '_blade_%02i' % iblade)
            self.blades[iblade].compute_endwalls_gaps()
            self.blades[iblade].show_blade_angles_contour(save_filename=self.config.get_machine_name() + '_blade_%02i' % iblade)
            
            # copy quantities to the block objects for later use
            fieldsForBFM = self.blades[iblade].getOutputFieldsForMeshBFM()
            self.blocks[iblock].addFieldsForBFM(fieldsForBFM)
            
            # if required, compute the body force fields
            if self.config.perform_body_force_reconstruction(): # compute it from circumferential averages
                self.blades[iblade].extract_body_force(self.blades[iblade].blade_metal_angle) 
            elif self.config.perform_body_force_interpolation(): # interpolate from another file
                self.blades[iblade].interpolate_body_force()
            elif self.config.perform_body_force_inference():
                self.blades[iblade].infer_body_force()
                self.blocks[iblock].add_inference_info(self.blades[iblade].inviscidForceInference, self.blades[iblade].viscousForceInference)
            else:
                print("No body force reconstruction requested.")
            

            
                
            
            
    
    
    def AssembleMultiBlockGrid(self):
        """
        Assemble the grid of the different blocks in a single one.
        """
        self.multiBlockGrid = MultiBlock(self.config, self.blocks, self.blades)
        self.multiBlockGrid.assemble_grid()
        self.multiBlockGrid.plot_full_grid(save_filename=self.config.get_machine_name(), ticks=True)
        if self.driverType=='multiblock':
            self.multiBlockGrid.plot_all_relevant_contours()
    
    
    def SaveOutput(self):
        """
        Save the output of the grid generation in a pickle file. 4 different options, possibly cumulative.
        The output type is taken from the input.ini file, under the voice OUTPUT_TYPE. Possible options are:
        1) turbobfm: save the turbobfm axisymmetric grid file with associated blade data
        2) pickle: save the full object in a pickle file
        3) su2mesh: export the su2 mesh file (3D, periodic boundaries for the moment)
        4) meridional_splines: export the meridional splines in a paraview readable format for Paraview span macros
        5) meridional_grid: export the meridional grid in csv format for paraview macro writing
        """
        print_banner_begin('MULTIBLOCK GRID OUTPUT')
        outputFolder = self.config.get_output_data_folder()
        os.makedirs(outputFolder, exist_ok=True)
        outputTypes = self.config.get_output_type()
        for outputType in outputTypes:
            
            if outputType.lower()=='cturbobfm':
                if self.driverType=='single_blade' or self.driverType=='full_machine':
                    raise ValueError('The output type cturbobfm is not available for single_blade or full_machine driver configurations.')
                self.multiBlockGrid.write_cturbobfm_grid_file()
            
            elif outputType.lower()=='pickle':
                filePath = os.path.join(outputFolder, self.config.get_machine_name() + '.pik')
                with open(filePath, 'wb') as f:
                    pickle.dump(self, f)
                print('Object saved in %s' %(filePath))
            
            elif outputType.lower()=='pickle_blades':
                for i,blade in enumerate(self.blades):
                    filePath = os.path.join(outputFolder, self.config.get_machine_name() + '_blade_%02i.pkl' % i)
                    with open(filePath, 'wb') as f:
                        pickle.dump(blade, f)
                    print('Blade %02i pkl saved in %s' %(i, filePath))
            
            elif outputType.lower()=='su2mesh':
                if self.driverType=='single_blade' or self.driverType=='full_machine':
                    raise ValueError('The output type su2mesh is not available for single_blade or full_machine driver configurations.')
                self.multiBlockGrid.compute_three_dimensional_mesh(self.config, nodes_number=5)
                generate_SU2mesh(self.multiBlockGrid.X_mesh, self.multiBlockGrid.Y_mesh, self.multiBlockGrid.Z_mesh, kind_elem=12, kind_bound=9, filename=outputFolder+'/mesh_%.4f.su2' %(self.multiBlockGrid.deltatheta_periodic))
                print('SU2 mesh file written in %s/mesh_%.4f.su2' %(outputFolder, self.multiBlockGrid.deltatheta_periodic))
            
            elif outputType.lower()=='meridional_splines':
                spanValues = [0.2, 0.5, 0.8] # default span values, modify if needed
                for span in spanValues:
                    self.multiBlockGrid.export_meridional_spline(folder=outputFolder, filename='meridional_spline_%.2f' %span, span=span)
            
            elif outputType.lower()=='meridional_grid':
                self.multiBlockGrid.write_paraview_grid_file(foldername=outputFolder, filename='meridional_grid.csv')
            
            elif outputType.lower()=='spanwise_profiles':
                self.multiBlockGrid.write_spanwise_splines(foldername=outputFolder)
            
            elif outputType=='none':
                print('No output type specified, therefore no output saved.')
            
            elif outputType=='blade':
                for i,blade in enumerate(self.blades):
                    filePath = outputFolder + '/blade_%02i.pik' % i
                    with open(filePath, 'wb') as file:
                        pickle.dump(blade, file)
                    print(f'Saved Blade object pickle {i} in: {filePath}.')
            
            elif outputType=='luminary_cad':
                self.multiBlockGrid.write_thetaWrapped_hub_shroud_curves(foldername=outputFolder, filename='tud-glra')
            
            else:
                print('Output type %s not recognized, therefore ignored.' %(outputType))
        print_banner_end('')