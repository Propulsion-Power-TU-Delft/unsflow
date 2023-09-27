import sys
sys.path.append('../../')
import Sun
import pickle
import numpy as np
import matplotlib.pyplot as plt

# meridional object list, in order from inlet to outlet
meridional_obj = []

filename = '../../Grid/testcases/nasa rotor 37/data/meta/nasa_rotor_config_01_inlet_10_10.pickle'
with open(filename, "rb") as file:
    meridional_obj.append(pickle.load(file))

filename = '../../Grid/testcases/nasa rotor 37/data/meta/nasa_rotor_config_01_blade_20_10.pickle'
with open(filename, "rb") as file:
    meridional_obj.append(pickle.load(file))

filename = '../../Grid/testcases/nasa rotor 37/data/meta/nasa_rotor_config_01_outlet_15_10.pickle'
with open(filename, "rb") as file:
    meridional_obj.append(pickle.load(file))

sun_objs = []
for obj in meridional_obj:
    compressor_grid = Sun.src.sun_grid.SunGrid(obj)
    # compressor_grid.ShowGrid()
    sun_obj = Sun.src.SunModel(compressor_grid)
    sun_obj.ComputeBoundaryNormals()
    rpm = -17.189e3
    sun_obj.add_shaft_rpm(rpm)
    sun_obj.AddNormalizationQuantities(1, 1, 1)
    sun_obj.ShowPhysicalGrid(mode='lines')
    sun_obj.ComputeSpectralGrid()
    # sun_obj.ShowSpectralGrid(mode='lines')
    gradient_routine = 'numpy'
    gradient_order = 2
    sun_obj.ComputeJacobianPhysical(routine=gradient_routine, order=gradient_order)
    # sun_obj.ContourTransformation(save_filename='iris_blade_jacobians_grid_35_15')
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
    sun_obj.impose_boundary_conditions('zero pressure', 'zero pressure')
    sun_objs.append(sun_obj)



# #settings for the research of poles
RS = [-3, 3]
DF = [0, 10]
grid=[10, 10]
sun_objs[2].ComputeSVDcompressor(RS_domain=RS, DF_domain=DF)
sun_objs[2].PlotInverseConditionNumberCompressor(save_filename='chi_map_iris')
plt.show()
