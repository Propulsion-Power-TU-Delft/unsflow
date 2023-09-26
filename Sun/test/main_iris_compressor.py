import sys
sys.path.append('../../')
import Sun
import pickle
import numpy as np
import matplotlib.pyplot as plt

# import the data from the pickle meridional object
filename = '../../Grid/testcases/iris/data/meta/iris_35_20_blade_50_20_outlet_35_20.pickle'
with open(filename, "rb") as file:
    meridional_obj = pickle.load(file)



#%%sun model
compressor_grid = Sun.src.sun_grid.SunGrid(meridional_obj)
compressor_grid.ShowGrid()
#Sun Model workflow
sun_obj = Sun.src.SunModel(compressor_grid)
sun_obj.ComputeBoundaryNormals()
sun_obj.ShowNormals()

#reference quantities
rho_ref = 1.014  # ref density
u_ref = 100  # ref velocity
x_ref = 0.003  # ref length

rpm = 85e3
sun_obj.AddNormalizationQuantities(1, 1, 1)
sun_obj.add_shaft_rpm(rpm)
sun_obj.NormalizeData()
sun_obj.ShowPhysicalGrid(save_filename='iris_blade_physical_grid_35_15', mode='lines')
sun_obj.ComputeSpectralGrid()
sun_obj.ShowSpectralGrid(save_filename='iris_blade_computational_grid_35_15', mode='lines')
gradient_routine = 'findiff'
gradient_order = 6
sun_obj.ComputeJacobianPhysical(routine=gradient_routine, order=gradient_order)
sun_obj.ContourTransformation(save_filename='iris_blade_jacobians_grid_35_15')
sun_obj.AddAMatrixToNodesFrancesco2()
sun_obj.AddBMatrixToNodesFrancesco2()
sun_obj.AddCMatrixToNodesFrancesco2()
sun_obj.AddEMatrixToNodesFrancesco2()
sun_obj.AddRMatrixToNodesFrancesco2()
sun_obj.AddSMatrixToNodes(BFM='radial')
sun_obj.AddHatMatricesToNodes()
sun_obj.ApplySpectralDifferentiation()
sun_obj.build_A_global_matrix()
sun_obj.build_C_global_matrix()
sun_obj.build_R_global_matrix()
sun_obj.build_S_global_matrix()
sun_obj.build_Z_global_matrix()
sun_obj.impose_boundary_conditions('zero pressure', 'zero pressure')


# #settings for the research of poles
RS = [-5, 5]
DF = [-3, 3]
grid=[10, 10]
sun_obj.ComputeSVDcompressor(RS_domain=RS, DF_domain=DF)
sun_obj.PlotInverseConditionNumberCompressor(save_filename='chi_map_iris')
plt.show()
