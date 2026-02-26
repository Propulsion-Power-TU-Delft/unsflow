from unsflow.sun.sun_model_multiblock import SunModelMultiBlock
from unsflow.sun.config import Config
from unsflow.sun.sun_model import SunModel
from unsflow.utils.plot_styles import *

# this could be an example of a multiblock driver

config = Config('input.ini')
nBlocks = config.getNumberOfBlocks()
sunBlocks = []
for iblock in range(nBlocks):
    block = SunModel(config, iblock)
    block.ComputeBoundaryNormals()
    block.ComputeSpectralGrid()
    block.ComputeJacobianPhysical()
    block.AddAMatrixToNodes_francesco()
    block.AddBMatrixToNodes_francesco()
    block.AddCMatrixToNodes_francesco()
    block.AddEMatrixToNodes_francesco()
    block.AddRMatrixToNodes_francesco()
    block.AddSMatrixToNodes()
    block.AddHatMatricesToNodes()
    block.ApplySpectralDifferentiation()
    block.build_A_global_matrix()
    block.build_C_global_matrix()
    block.build_R_global_matrix()
    block.build_S_global_matrix()
    block.build_Z_global_matrix()
    block.compute_L_matrices()
    block.read_boundary_conditions()
    block.apply_boundary_conditions_generalized()
    sunBlocks.append(block)
    
multiBlock = SunModelMultiBlock(sunBlocks, config)
multiBlock.construct_L_global_matrices()
multiBlock.apply_matching_conditions()
multiBlock.compute_P_Y_matrices()
multiBlock.solve_evp()
multiBlock.extract_eigenfields()
multiBlock.SaveOutput()
