import pickle
import matplotlib.pyplot as plt
import Sun
from Sun.src.sun_model_multiblock import SunModelMultiBlock
from Grid.src.config import Config
import numpy as np
from scipy.sparse.linalg import eigs

config = Config('nasa_rotor_37.ini')
with open(config.get_meridional_pickle_filepath(), "rb") as file:
    meridional_obj = pickle.load(file)

# STABILITY ANALYSIS
sun_blocks = []
for meridional_block in meridional_obj.group:
    compressor_grid = Sun.src.sun_grid.SunGrid(meridional_block)
    # compressor_grid.check_fields()
    sun_blocks.append(Sun.src.SunModel(compressor_grid, config))

ii = 0
for sun_obj in sun_blocks:
    sun_obj.ComputeBoundaryNormals()
    # sun_obj.ShowNormals()
    sun_obj.set_overwriting_equation_euler_wall('utheta')
    # sun_obj.ShowPhysicalGrid(save_filename='physical_grid_%i' % (ii), mode='lines')
    sun_obj.ComputeSpectralGrid()
    # sun_obj.ShowSpectralGrid(save_filename='spectral_grid_%i' % (ii), mode='lines')
    sun_obj.ComputeJacobianPhysical()
    # sun_obj.ContourTransformation(save_filename='jacobian_%i' % (ii))
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
    sun_obj.compute_L_matrices(ii)
    sun_obj.set_boundary_conditions()
    # sun_obj.inspect_L_matrices(save_foldername='bc_logic', save_filename='before_bc_block_%i' %(ii))
    sun_obj.apply_boundary_conditions_generalized()
    # sun_obj.inspect_L_matrices(save_foldername='bc_logic', save_filename='after_bc_block_%i' %(ii))
    ii += 1

sun_multiblock = SunModelMultiBlock(sun_blocks, config)
sun_multiblock.construct_L_global_matrices()
sun_multiblock.inspect_L_matrices(save_foldername='bc_logic', save_filename='before_bc_multiblock')
sun_multiblock.apply_matching_conditions()
# sun_multiblock.inspect_L_matrices(save_foldername='bc_logic', save_filename='after_bc_multiblock')
# sun_multiblock.compute_P_Y_matrices()
# sun_multiblock.solve_evp()
# sun_multiblock.extract_eigenfields()
# sun_multiblock.plot_eigenfrequencies(save_filename='eigenfrequencies')
# sun_multiblock.plot_eigenfields(n=10, save_filename='eigenmode')
# sun_multiblock.write_results()


# linear alternative
omega_search = 0
omega_ref = config.get_reference_omega()
sigma = 0
A = sun_multiblock.L0
M = -sun_multiblock.L1
C = np.linalg.inv(A - sigma * M)
C = np.dot(C, M)
print('Searching Eigenvalues with ARPACK...')
eigenvalues, eigenvectors = eigs(C, k=config.get_research_number_omega_eigenvalues())
eigenvalues = sigma + 1 / eigenvalues
eigenvalues *= omega_ref

# make copies of the arrays to sort
eigenfreqs = np.copy(eigenvalues)
df = np.copy(eigenvalues.imag)
rs = np.copy(eigenvalues.real)
eigenvecs = np.copy(eigenvectors)
sorted_indices = sorted(range(len(rs)), key=lambda i: rs[i], reverse=False)

# order the original arrays following the sorting indices
for i in range(len(sorted_indices)):
    eigenvalues[i] = eigenfreqs[sorted_indices[i]]
    eigenvectors[:, i] = eigenvecs[:, sorted_indices[i]]

# PLOT RESULTS
marker_size = 50
fig, ax = plt.subplots()
ax.scatter(eigenvalues.real/omega_ref, eigenvalues.imag/omega_ref, marker='o', facecolors='none', edgecolors='red',
           s=marker_size, label=r'numerical')
ax.set_xlabel(r'RS')
ax.set_ylabel(r'DF')
ax.legend()
ax.grid(alpha=0.3)
fig.savefig('pictures/chi_map_arnoldi.pdf', bbox_inches='tight')

# EIGENFUNCTIONS
z_grid = sun_multiblock.z_grid
r_grid = sun_multiblock.r_grid

for ivec in range(np.shape(eigenvectors)[1]):
    eigenvec = eigenvectors[:, ivec]
    rho_eig = []
    ur_eig = []
    ut_eig = []
    uz_eig = []
    p_eig = []

    for i in range(len(eigenvec)):
        if (i) % 5 == 0:
            rho_eig.append(eigenvec[i])
        elif (i - 1) % 5 == 0 and i != 0:
            ur_eig.append(eigenvec[i])
        elif (i - 2) % 5 == 0 and i != 0:
            ut_eig.append(eigenvec[i])
        elif (i - 3) % 5 == 0 and i != 0:
            uz_eig.append(eigenvec[i])
        elif (i - 4) % 5 == 0 and i != 0:
            p_eig.append(eigenvec[i])
        else:
            raise ValueError("Not correct indexing for eigenvector retrieval!")


    def scaled_eigenvector_real(eig_list):
        array = np.array(eig_list, dtype=complex)
        array = np.reshape(array, (sun_multiblock.z_grid.shape[0], sun_multiblock.z_grid.shape[1]))
        array_real_scaled = array.real / (np.max(array.real) - np.min(array.real))
        return array_real_scaled


    rho_eig_r = scaled_eigenvector_real(rho_eig)
    ur_eig_r = scaled_eigenvector_real(ur_eig)
    ut_eig_r = scaled_eigenvector_real(ut_eig)
    uz_eig_r = scaled_eigenvector_real(uz_eig)
    p_eig_r = scaled_eigenvector_real(p_eig)

    plt.figure(figsize=(7, 5))
    plt.contourf(sun_multiblock.z_grid, sun_multiblock.r_grid, rho_eig_r, levels=20, cmap='bwr')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{\rho}_{%i}$' % (ivec + 1))
    plt.colorbar()
    # plt.savefig('pictures/%i_%i/eigenfunction_rho_%i.pdf' % (Nz*3, Nr, ivec + 1), bbox_inches='tight')

    plt.figure(figsize=(7, 5))
    plt.contourf(sun_multiblock.z_grid, sun_multiblock.r_grid, ur_eig_r, levels=20, cmap='bwr')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{u}_{r,%i}$' % (ivec + 1))
    plt.colorbar()
    # plt.savefig('pictures/%i_%i/eigenfunction_ur_%i.pdf' % (Nz*3, Nr, ivec + 1), bbox_inches='tight')

    plt.figure(figsize=(7, 5))
    plt.contourf(sun_multiblock.z_grid, sun_multiblock.r_grid, ut_eig_r, levels=20, cmap='bwr')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{u}_{\theta,%i}$' % (ivec + 1))
    plt.colorbar()
    # plt.savefig('pictures/%i_%i/eigenfunction_ut_%i.pdf' % (Nz*3, Nr, ivec + 1), bbox_inches='tight')

    plt.figure(figsize=(7, 5))
    plt.contourf(sun_multiblock.z_grid, sun_multiblock.r_grid, uz_eig_r, levels=20, cmap='bwr')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{u}_{z,%i}$' % (ivec + 1))
    plt.colorbar()
    # plt.savefig('pictures/%i_%i/eigenfunction_uz_%i.pdf' % (Nz*3, Nr, ivec + 1), bbox_inches='tight')

    plt.figure(figsize=(7, 5))
    plt.contourf(sun_multiblock.z_grid, sun_multiblock.r_grid, p_eig_r, levels=20, cmap='bwr')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{p}_{%i}$' % (ivec + 1))
    plt.colorbar()
    # plt.savefig('pictures/%i_%i/eigenfunction_p_%i.pdf' % (Nz*3, Nr, ivec + 1), bbox_inches='tight')

plt.show()
