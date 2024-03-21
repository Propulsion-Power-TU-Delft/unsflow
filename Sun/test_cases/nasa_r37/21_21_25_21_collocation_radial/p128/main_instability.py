import pickle
import matplotlib.pyplot as plt
import Sun
from Sun.src.sun_model_multiblock import SunModelMultiBlock
from Grid.src.config import Config
from Grid.src.functions import create_folder

folder_out = 'pictures'
create_folder(folder_out)

config = Config('nasa_rotor_37.ini')
with open(config.get_meridional_pickle_filepath(), "rb") as file:
    meridional_obj = pickle.load(file)

for meridional_block in meridional_obj.group:
    meridional_block.contour_all_plots(save_filename='check')

# STABILITY ANALYSIS
sun_blocks = []
for meridional_block in meridional_obj.group:
    compressor_grid = Sun.src.sun_grid.SunGrid(meridional_block)
    sun_blocks.append(Sun.src.SunModel(compressor_grid, config))

ii = 0
for sun_obj in sun_blocks:
    sun_obj.ComputeBoundaryNormals()
    sun_obj.set_overwriting_equation_euler_wall('ur')
    sun_obj.ComputeSpectralGrid()
    sun_obj.ComputeJacobianPhysical()
    sun_obj.AddAMatrixToNodesFrancesco2()
    sun_obj.AddBMatrixToNodesFrancesco2()
    sun_obj.AddCMatrixToNodesFrancesco2()
    sun_obj.AddEMatrixToNodesFrancesco2()
    sun_obj.AddRMatrixToNodesFrancesco2()
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
