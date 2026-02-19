import pickle
import matplotlib.pyplot as plt
import Sun
from sun.src.sun_model_multiblock import SunModelMultiBlock
from grid.src.config import Config
from grid.src.functions import create_folder

folder_out = 'pictures'
create_folder(folder_out)

config = Config('nasa_rotor_37.ini')
with open(config.get_meridional_pickle_filepath(), "rb") as file:
    meridional_obj = pickle.load(file)

# STABILITY ANALYSIS
sun_blocks = []
for meridional_block in meridional_obj.group:
    compressor_grid = sun.src.sun_grid.SunGrid(meridional_block)
    sun_blocks.append(sun.src.SunModel(compressor_grid, config))

ii = 0
for sun_obj in sun_blocks:
    sun_obj.ComputeBoundaryNormals()
    sun_obj.set_overwriting_equation_euler_wall('uz')
    sun_obj.ComputeSpectralGrid()
    sun_obj.ComputeJacobianPhysical()
    sun_obj.AddAMatrixToNodes_francesco()
    sun_obj.AddBMatrixToNodes_francesco()
    sun_obj.AddCMatrixToNodes_francesco()
    sun_obj.AddEMatrixToNodes_francesco()
    sun_obj.AddRMatrixToNodes_francesco()
    sun_obj.AddSMatrixToNodes()
    sun_obj.AddHatMatricesToNodes()
    sun_obj.ApplySpectralDifferentiationKronecker()
    sun_obj.build_A_global_matrix()
    sun_obj.build_C_global_matrix()
    sun_obj.build_R_global_matrix()
    sun_obj.build_S_global_matrix()
    sun_obj.build_Z_global_matrix()
    sun_obj.compute_L_matrices(ii)
    sun_obj.set_boundary_conditions()
    sun_obj.apply_boundary_conditions_generalized()
    ii += 1

sun_multiblock = SunModelMultiBlock(sun_blocks, config)
sun_multiblock.construct_L_global_matrices()
sun_multiblock.apply_matching_conditions(mode='collocation method')
sun_multiblock.compute_P_Y_matrices()
sun_multiblock.solve_evp()
sun_multiblock.extract_eigenfields()
sun_multiblock.plot_eigenfrequencies(save_filename='eigenfrequencies', save_foldername=folder_out)
sun_multiblock.plot_eigenfields(n=20, save_filename='eigenmode', save_foldername=folder_out)
sun_multiblock.write_results()

# plt.show()
