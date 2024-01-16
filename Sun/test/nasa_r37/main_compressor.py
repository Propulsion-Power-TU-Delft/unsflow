import pickle
import matplotlib.pyplot as plt
import Sun
from Sun.src.sun_model_multiblock import SunModelMultiBlock
from Grid.src.config import Config

config = Config('../../../Grid/testcases/nasa_rotor_37/nasa_rotor_37.ini')
with open(config.get_meridional_pickle_filepath(), "rb") as file:
    meridional_obj = pickle.load(file)

# STABILITY ANALYSIS
sun_blocks = []
for meridional_block in meridional_obj.group:
    compressor_grid = Sun.src.sun_grid.SunGrid(meridional_block)
    sun_blocks.append(Sun.src.SunModel(compressor_grid, config))

ii = 0
for sun_obj in sun_blocks:
    sun_obj.ComputeBoundaryNormals()
    sun_obj.set_overwriting_equation_euler_wall('utheta')
    # sun_obj.ShowPhysicalGrid(save_filename='physical_grid', mode='lines')
    sun_obj.ComputeSpectralGrid()
    # sun_obj.ShowSpectralGrid(save_filename='computational_grid', mode='lines')
    sun_obj.ComputeJacobianPhysical()
    # sun_obj.ContourTransformation(save_filename='jacobian')
    sun_obj.AddAMatrixToNodesFrancesco2()
    sun_obj.AddBMatrixToNodesFrancesco2()
    sun_obj.AddCMatrixToNodesFrancesco2()
    sun_obj.AddEMatrixToNodesFrancesco2()
    sun_obj.AddRMatrixToNodesFrancesco2()
    sun_obj.AddSMatrixToNodes()
    sun_obj.AddHatMatricesToNodes()
    sun_obj.ApplySpectralDifferentiation()
    sun_obj.build_A_global_matrix()
    sun_obj.build_C_global_matrix()
    sun_obj.build_R_global_matrix()
    sun_obj.build_S_global_matrix()
    sun_obj.build_Z_global_matrix()
    sun_obj.set_boundary_conditions()
    sun_obj.apply_boundary_conditions_generalized()
    sun_obj.compute_L_matrices(ii)
    sun_obj.compute_block_Y_P_matrices()
    ii += 1

sun_multiblock = SunModelMultiBlock(sun_blocks, config)
sun_multiblock.construct_L_global_matrices()
sun_multiblock.apply_matching_conditions()
sun_multiblock.compute_P_Y_matrices()
sun_multiblock.solve_evp()
sun_multiblock.extract_eigenfields()
sun_multiblock.plot_eigenfrequencies(save_filename='eigenfrequencies')
sun_multiblock.plot_eigenfields(n=10, save_filename='eigenmode')

# single block outline
# sun_obj.extract_eigenfields()
# sun_obj.plot_eigenfrequencies(save_filename='eigenfrequencies')
# sun_obj.plot_eigenfields(save_filename='eigenmode')
# sun_obj.write_results()

plt.show()
