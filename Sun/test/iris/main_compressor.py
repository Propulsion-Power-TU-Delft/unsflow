import sys
sys.path.append('../../')
import Sun
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs


# import the data from the pickle meridional object
filename = '../../../Grid/testcases/iris/data/meta/inlet_10_blade_40_outlet_20_nspan_20.pickle'
with open(filename, "rb") as file:
    meridional_obj = pickle.load(file)


#%%sun model
compressor_grid = Sun.src.sun_grid.SunGrid(meridional_obj)
compressor_grid.ShowGrid()
#Sun Model workflow
sun_obj = Sun.src.SunModel(compressor_grid)
sun_obj.ComputeBoundaryNormals()
sun_obj.ShowNormals()


rpm = 85e3
Omega = rpm * 2 * np.pi / 60
omega_ref = np.abs(Omega)
t_ref = 1/omega_ref
tau = np.abs(2 * np.pi / Omega)
m = 1
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
sun_obj.AddCMatrixToNodesFrancesco2(m=m)
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
sun_obj.set_boundary_conditions('compressor inlet', 'compressor outlet', 'euler wall', 'euler wall')
sun_obj.apply_boundary_conditions_generalized()
sun_obj.solve_evp_arnoldi(m=m)










#%%%%%%%%%%%%% SOLVE EVP %%%%%%%%%%%%%%%%%%%%%
omega_search = 0
sigma = omega_search / omega_ref

L0 = sun_obj.Z_g * (1 + 1j * m * Omega/omega_ref * tau/t_ref) + sun_obj.S_g
plt.figure()
plt.spy(L0.real**2 + L0.imag**2)
plt.title('L0')

L1 = sun_obj.A_g * (m * Omega/omega_ref * tau/t_ref - 1j) - 1j * tau/t_ref * sun_obj.Z_g
plt.figure()
plt.spy(L1.real**2 + L1.imag**2)
plt.title('L1')

L2 = -tau/t_ref * sun_obj.A_g
plt.figure()
plt.spy(L2.real**2 + L2.imag**2)
plt.title('L2')

Y1 = np.concatenate((-L0, np.zeros_like(L0)), axis=1)
Y2 = np.concatenate((np.zeros_like(L0), np.eye(L0.shape[0])), axis=1)
Y = np.concatenate((Y1, Y2), axis=0)
plt.figure()
plt.spy(Y.real**2 + Y.imag**2)
plt.title('Y')

P1 = np.concatenate((L1, L2), axis=1)
P2 = np.concatenate((np.eye(L0.shape[0]), np.zeros_like(L0)), axis=1)
P = np.concatenate((P1, P2), axis=0)
plt.figure()
plt.spy(P.real**2 + P.imag**2)
plt.title('P')

print("Transforming generalized EVP in standard one...")
Y_tilde = np.linalg.inv(Y - sigma * P)
Y_tilde = np.dot(Y_tilde, P)
number_search = 1
print("Solving EVP...")
eigenvalues, eigenvectors = eigs(Y_tilde, k=number_search)
eigenvalues = sigma + 1 / eigenvalues
eigenvalues *= omega_ref

marker_size = 20
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(eigenvalues.real / omega_ref, eigenvalues.imag / omega_ref, marker='o', facecolors='red', edgecolors='red',
           s=marker_size, label=r'numerical')
ax.set_xlabel(r'RS [-]')
ax.set_ylabel(r'DF [-]')
# ax.legend()
# ax.set_xlim([-1.5, 1.5])
# ax.set_ylim([-2, 0.5])
ax.grid(alpha=0.3)
fig.savefig('pictures/chi_map_arnoldi_%i.pdf' % (eigenvalues[0].real), bbox_inches='tight')
#
#
# # # EIGENFUNCTIONS
if number_search == 1:
    Nz = sun_obj.data.meridional_obj.nstream
    Nr = sun_obj.data.meridional_obj.nspan
    mode_name = "unknown"
    z_grid = sun_obj.data.meridional_obj.z_cg
    r_grid = sun_obj.data.meridional_obj.r_cg
    rho_eig = []
    ur_eig = []
    ut_eig = []
    uz_eig = []
    p_eig = []
    #
    for i in range(len(eigenvectors)//2):
        if (i) % 5 == 0:
            rho_eig.append(eigenvectors[i])
        elif (i - 1) % 5 == 0 and i != 0:
            ur_eig.append(eigenvectors[i])
        elif (i - 2) % 5 == 0 and i != 0:
            ut_eig.append(eigenvectors[i])
        elif (i - 3) % 5 == 0 and i != 0:
            uz_eig.append(eigenvectors[i])
        elif (i - 4) % 5 == 0 and i != 0:
            p_eig.append(eigenvectors[i])
        else:
            raise ValueError("Not correct indexing for eigenvector retrieval!")


    def scaled_eigenvector_real(eig_list):
        array = np.array(eig_list, dtype=complex)
        array = np.reshape(array, (Nz, Nr))
        array_real_scaled = array.real / (np.max(array.real) - np.min(array.real))
        return array_real_scaled


    rho_eig_r = scaled_eigenvector_real(rho_eig)
    ur_eig_r = scaled_eigenvector_real(ur_eig)
    ut_eig_r = scaled_eigenvector_real(ut_eig)
    uz_eig_r = scaled_eigenvector_real(uz_eig)
    p_eig_r = scaled_eigenvector_real(p_eig)



    plt.figure(figsize=(7, 5))
    plt.contourf(z_grid, r_grid, rho_eig_r, levels=100, cmap='coolwarm')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{\rho} \quad$'+mode_name)
    plt.colorbar()
    plt.savefig('pictures/eigenfunction_rho_2D_%i_%i_%i.pdf' % (Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
    plt.figure(figsize=(7, 5))
    plt.contourf(z_grid, r_grid, ur_eig_r, levels=100, cmap='coolwarm')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{u}_r \quad$' + mode_name)
    plt.colorbar()
    plt.savefig('pictures/eigenfunction_ur_2D_%i_%i_%i.pdf' % (Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
    plt.figure(figsize=(7, 5))
    plt.contourf(z_grid, r_grid, ut_eig_r, levels=100, cmap='coolwarm')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{u}_{\theta} \quad$' + mode_name)
    plt.colorbar()
    plt.savefig('pictures/eigenfunction_ut_2D_%i_%i_%i.pdf' % (Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
    plt.figure(figsize=(7, 5))
    plt.contourf(z_grid, r_grid, uz_eig_r, levels=100, cmap='coolwarm')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{u}_z \quad$' + mode_name)
    plt.colorbar()
    plt.savefig('pictures/eigenfunction_uz_2D_%i_%i_%i.pdf' % (Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
    plt.figure(figsize=(7, 5))
    plt.contourf(z_grid, r_grid, p_eig_r, levels=100, cmap='coolwarm')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{p} \quad$' + mode_name)
    plt.colorbar()
    plt.savefig('pictures/eigenfunction_p_2D_%i_%i_%i.pdf' % (Nz, Nr, eigenvalues[0].real), bbox_inches='tight')

plt.show()
