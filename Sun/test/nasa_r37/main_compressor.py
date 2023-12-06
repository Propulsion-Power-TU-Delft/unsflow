import pickle
import matplotlib.pyplot as plt
import Sun
from Grid.src.config import Config

config = Config('nasa_rotor_37.ini')
with open(config.get_meridional_pickle_filepath(), "rb") as file:
    meridional_obj = pickle.load(file)

# STABILITY ANALYSIS
sun_blocks = []
for meridional_block in meridional_obj.group:
    compressor_grid = Sun.src.sun_grid.SunGrid(meridional_block)
    sun_blocks.append(Sun.src.SunModel(compressor_grid, config))

for sun_obj in sun_blocks:
    sun_obj.ComputeBoundaryNormals()
    sun_obj.set_overwriting_equation_euler_wall('utheta')
    sun_obj.ShowPhysicalGrid(save_filename='physical_grid', mode='lines')
    sun_obj.ComputeSpectralGrid()
    sun_obj.ShowSpectralGrid(save_filename='computational_grid', mode='lines')
    sun_obj.ComputeJacobianPhysical()
    sun_obj.ContourTransformation(save_filename='jacobian')
    sun_obj.AddAMatrixToNodes()
    sun_obj.AddBMatrixToNodes()
    sun_obj.AddCMatrixToNodes()
    sun_obj.AddEMatrixToNodes()
    sun_obj.AddRMatrixToNodes()
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
    sun_obj.compute_block_Y_P_matrices()

for sun_obj in sun_blocks:
    Y = sun_obj.Y
    P = sun_obj.P
    # now concatenate, apply interface boundary conditions, and solve the final system. easily said than done






    # sun_obj.extract_eigenfields()
    # sun_obj.plot_eigenfrequencies(save_filename='eigenfrequencies')
    # sun_obj.plot_eigenfields(save_filename='eigenmode')
    # sun_obj.write_results()

    # plt.show()
