import sys
sys.path.append('../../')
import Sun
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs


# import the data from the pickle meridional object
filename = '../../../Grid/testcases/iris/data/meta/inlet_10_15_blade_30_15_outlet_10_15.pickle'
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
# rho_ref = 1.014  # ref density
# u_ref = 100  # ref velocity
# x_ref = 0.003  # ref length

rpm = 85e3
sun_obj.add_shaft_rpm(rpm)
sun_obj.AddNormalizationQuantities(1, 1, 1)
sun_obj.ShowPhysicalGrid(save_filename='iris_blade_physical_grid_35_15', mode='lines')
sun_obj.ComputeSpectralGrid()
sun_obj.ShowSpectralGrid(save_filename='iris_blade_computational_grid_35_15', mode='lines')
gradient_routine = 'findiff'
gradient_order = 6
sun_obj.ComputeJacobianPhysical(routine=gradient_routine, order=gradient_order)
sun_obj.ContourTransformation(save_filename='iris_blade_jacobians_grid_35_15')
sun_obj.AddAMatrixToNodesFrancesco2()
sun_obj.AddBMatrixToNodesFrancesco2()
sun_obj.AddCMatrixToNodesFrancesco2(m=1)
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
sun_obj.impose_boundary_conditions('zero perturbation', 'zero pressure')
sun_obj.apply_boundary_conditions_generalized()

omega_search = 0
omega_ref = np.abs(rpm)*2*np.pi/60
sigma = omega_search / omega_ref
A = sun_obj.Z_g
M = sun_obj.A_g
print("Transforming generalized EVP in standard one...")
C = np.linalg.inv(A - sigma * M)
C = np.dot(C, M)
number_search = 300
print("Solving EVP...")
eigenvalues, eigenvectors = eigs(C, k=number_search)
eigenvalues = sigma + 1 / eigenvalues
eigenvalues *= omega_ref

#
# # #settings for the research of poles
# RS = [-1, 1]
# DF = [-5, 5]
# grid=[25, 25]
# sun_obj.ComputeSVDcompressor(RS_domain=RS, DF_domain=DF, grid=grid)
# sun_obj.PlotInverseConditionNumberCompressor(save_filename='chi_map_nasar37')
# plt.show()

marker_size = 20
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(eigenvalues.real/omega_ref, eigenvalues.imag/omega_ref, marker='o', facecolors='red', edgecolors='red',
           s=marker_size, label=r'numerical')
ax.set_xlabel(r'RS [-]')
ax.set_ylabel(r'DF [-]')
ax.legend()
# ax.set_xlim([7500, 35000])
# ax.set_ylim([-8000, 8000])
ax.grid(alpha=0.3)
# fig.savefig('pictures/%i/chi_map_arnoldi_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')













# # EIGENFUNCTIONS
# Nz = sun_obj.data.meridional_obj.nstream
# Nr = sun_obj.data.meridional_obj.nspan
# mode_name = "unknown"
# z_grid = sun_obj.data.meridional_obj.z_cg
# r_grid = sun_obj.data.meridional_obj.r_cg
# rho_eig = []
# ur_eig = []
# ut_eig = []
# uz_eig = []
# p_eig = []
#
# for i in range(len(eigenvectors)):
#     if (i) % 5 == 0:
#         rho_eig.append(eigenvectors[i])
#     elif (i - 1) % 5 == 0 and i != 0:
#         ur_eig.append(eigenvectors[i])
#     elif (i - 2) % 5 == 0 and i != 0:
#         ut_eig.append(eigenvectors[i])
#     elif (i - 3) % 5 == 0 and i != 0:
#         uz_eig.append(eigenvectors[i])
#     elif (i - 4) % 5 == 0 and i != 0:
#         p_eig.append(eigenvectors[i])
#     else:
#         raise ValueError("Not correct indexing for eigenvector retrieval!")
#
#
# def scaled_eigenvector_real(eig_list):
#     array = np.array(eig_list, dtype=complex)
#     array = np.reshape(array, (Nz, Nr))
#     array_real_scaled = array.real / (np.max(array.real) - np.min(array.real))
#     return array_real_scaled
#
#
# rho_eig_r = scaled_eigenvector_real(rho_eig)
# ur_eig_r = scaled_eigenvector_real(ur_eig)
# ut_eig_r = scaled_eigenvector_real(ut_eig)
# uz_eig_r = scaled_eigenvector_real(uz_eig)
# p_eig_r = scaled_eigenvector_real(p_eig)
#
#
#
# plt.figure(figsize=(7, 5))
# plt.contourf(z_grid, r_grid, rho_eig_r, levels=30, cmap='RdBu')
# plt.ylabel(r'$r$ [-]')
# plt.xlabel(r'$z$ [-]')
# plt.title(r'$\tilde{\rho} \quad$'+mode_name)
# plt.colorbar()
# # plt.savefig('pictures/%i/eigenfunction_rho_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
# plt.figure(figsize=(7, 5))
# plt.contourf(z_grid, r_grid, ur_eig_r, levels=30, cmap='RdBu')
# plt.ylabel(r'$r$ [-]')
# plt.xlabel(r'$z$ [-]')
# plt.title(r'$\tilde{u}_r \quad$' + mode_name)
# plt.colorbar()
# # plt.savefig('pictures/%i/eigenfunction_ur_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
# plt.figure(figsize=(7, 5))
# plt.contourf(z_grid, r_grid, ut_eig_r, levels=30, cmap='RdBu')
# plt.ylabel(r'$r$ [-]')
# plt.xlabel(r'$z$ [-]')
# plt.title(r'$\tilde{u}_{\theta} \quad$' + mode_name)
# plt.colorbar()
# # plt.savefig('pictures/%i/eigenfunction_ut_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
# plt.figure(figsize=(7, 5))
# plt.contourf(z_grid, r_grid, uz_eig_r, levels=30, cmap='RdBu')
# plt.ylabel(r'$r$ [-]')
# plt.xlabel(r'$z$ [-]')
# plt.title(r'$\tilde{u}_z \quad$' + mode_name)
# plt.colorbar()
# # plt.savefig('pictures/%i/eigenfunction_uz_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
# plt.figure(figsize=(7, 5))
# plt.contourf(z_grid, r_grid, p_eig_r, levels=30, cmap='RdBu')
# plt.ylabel(r'$r$ [-]')
# plt.xlabel(r'$z$ [-]')
# plt.title(r'$\tilde{p} \quad$' + mode_name)
# plt.colorbar()
# # plt.savefig('pictures/%i/eigenfunction_p_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
#
plt.show()