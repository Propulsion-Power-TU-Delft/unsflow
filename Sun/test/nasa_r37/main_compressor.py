import pickle
import matplotlib.pyplot as plt
import Sun

# INPUT
filename = '../../../Grid/testcases/nasa rotor 37/data/meta/inlet_25_blade_25_outlet_45_nspan_25.pickle'
with open(filename, "rb") as file:
    meridional_obj = pickle.load(file)
rpm = -17.189e3
m = 1
gradient_routine = 'numpy'
gradient_order = 2




# STABILITY ANALYSIS
compressor_grid = Sun.src.sun_grid.SunGrid(meridional_obj)
sun_obj = Sun.src.SunModel(compressor_grid)
sun_obj.ComputeBoundaryNormals()
sun_obj.add_shaft_rpm(rpm)
sun_obj.AddNormalizationQuantities(1, 1, 1)
sun_obj.ShowPhysicalGrid(save_filename='physical_grid', mode='lines')
sun_obj.ComputeSpectralGrid()
sun_obj.ShowSpectralGrid(save_filename='computational_grid', mode='lines')
sun_obj.ComputeJacobianPhysical(routine=gradient_routine, order=gradient_order)
sun_obj.ContourTransformation(save_filename='jacobian')
sun_obj.AddAMatrixToNodesFrancesco2()
sun_obj.AddBMatrixToNodesFrancesco2()
sun_obj.AddCMatrixToNodesFrancesco2(m=m)
sun_obj.AddEMatrixToNodesFrancesco2()
sun_obj.AddRMatrixToNodesFrancesco2()
sun_obj.AddSMatrixToNodes(turbo=True)
sun_obj.AddHatMatricesToNodes()
sun_obj.ApplySpectralDifferentiation()
sun_obj.build_A_global_matrix()
sun_obj.build_C_global_matrix()
sun_obj.build_R_global_matrix()
sun_obj.build_S_global_matrix()
sun_obj.build_Z_global_matrix()
sun_obj.set_boundary_conditions('compressor inlet', 'compressor outlet', 'euler wall', 'euler wall')
sun_obj.apply_boundary_conditions_generalized()
sun_obj.solve_evp_arnoldi(number_search=15)
sun_obj.plot_eigenfrequencies(save_filename='eigenfrequencies')
sun_obj.extract_eigenfields()
sun_obj.sort_eigensolution()
sun_obj.plot_eigenfields(save_filename='eigenmode')
sun_obj.write_results()

# plt.show()
