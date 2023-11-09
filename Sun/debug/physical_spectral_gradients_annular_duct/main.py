import matplotlib.pyplot as plt
import numpy as np
import Sun
import os

# input data of the problem (SI units)
r1 = 0.1826  # inner radius [m]
r2 = 0.2487  # outer radius [m]
M = 0.015  # Mach number
p = 100e3  # pressure [Pa]
T = 288  # temperature [K]
L = 0.08  # length [m]
R = 287.058  # air gas constant [kJ/kgK]
gmma = 1.4  # cp/cv ratio of air
rho = p / (R * T)  # density [kg/m3]
a = np.sqrt(gmma * p / rho)  # ideal speed of sound [m/s]
HARMONIC_ORDER = 1

# non-dimensionalization terms:
x_ref = r1
u_ref = M * a
rho_ref = rho
t_ref = x_ref / u_ref
omega_ref = 1 / t_ref
p_ref = rho_ref * u_ref ** 2

# %%%%%%%%%%%%%%%%%%%%%%% NUMERICAL PART FOLLOWING SUN MODULE METHODS %%%%%%%%%%%%%%%%%%%%%%%
# number of grid nodes in the computational domain
Nz = 60
Nr = 20
number_search = 15
gradient_routine = 'findiff'
gradient_order = 12
folder_path = "pictures/" + str(Nz) + "_" + str(Nr)  # Replace with the desired folder path
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# implement a constant uniform flow in the annulus duct
density = np.zeros((Nz, Nr))
axialVel = np.zeros((Nz, Nr))
radialVel = np.zeros((Nz, Nr))
tangentialVel = np.zeros((Nz, Nr))
pressure = np.zeros((Nz, Nr))
for ii in range(0, Nz):
    for jj in range(0, Nr):
        density[ii, jj] = rho
        axialVel[ii, jj] = M * a
        pressure[ii, jj] = p

# create a meridional object, having the same information of the meridional post-process object of a compressor
duct_Obj = Sun.src.AnnulusMeridional(0, L, r1, r2, Nz, Nr,
                                     density, radialVel, tangentialVel, axialVel, pressure, grid_refinement=1)
duct_Obj.normalize_data(rho_ref, u_ref, x_ref)
duct_grid = Sun.src.sun_grid.SunGrid(duct_Obj)
duct_grid.ShowGrid()

# general workflow of the sun model
sun_obj = Sun.src.SunModel(duct_grid)
sun_obj.set_overwriting_equation_euler_wall('utheta')
sun_obj.ComputeBoundaryNormals()
sun_obj.ShowNormals()
sun_obj.add_shaft_rpm(60 * omega_ref / 2 / np.pi)
sun_obj.set_normalization_quantities(mode='duct object')
sun_obj.ComputeSpectralGrid()
sun_obj.ComputeJacobianPhysical(routine=gradient_routine, order=gradient_order, method='nearest')


# sun_obj.ContourTransformation(save_filename='%i_%i/transformation' % (Nz, Nr))


# %%%%%%%%%%%%%%%%%%%%%%% ANALYTICAL PART %%%%%%%%%%%%%%%%%%%%%%%
def compute_analytical_derivative(z, L1, L2):
    """
    analytical transformation for the annular problem. L1,L2 represent the extremes. z is the physical array cordinate.
    It can be used for both axial and radial transformation, using proper inputs.
    """
    dcomputational_dphysical = -np.sin(np.pi * (z - L1) / (L2 - L1)) * np.pi / (L2 - L1)
    return dcomputational_dphysical


z = np.linspace(0, L, Nz)
r = np.linspace(r1, r2, Nr)
dxi_dz = compute_analytical_derivative(z / x_ref, 0 / x_ref, L / x_ref)
deta_dr = compute_analytical_derivative(r / x_ref, r1 / x_ref, r2 / x_ref)

plt.figure()
plt.plot(z / x_ref, dxi_dz, label='analytical')
plt.plot(z / x_ref, sun_obj.dxdz[:, 0], '--s', label='numerical')
plt.xlabel(r'$\hat{z} \ \mathrm{[-]}$')
plt.ylabel(r'$\partial \xi / \partial \hat{z} \ \mathrm{[-]}$')
plt.legend()
plt.savefig(folder_path + '/dxi_dz_%s_%i.pdf' % (gradient_routine, gradient_order), bbox_inches='tight')

plt.figure()
plt.plot(r / x_ref, deta_dr, label='analytical')
plt.plot(r / x_ref, sun_obj.dydr[0, :], '--s', label='numerical')
plt.xlabel(r'$\hat{r} \ \mathrm{[-]}$')
plt.ylabel(r'$\partial \eta / \partial \hat{r} \ \mathrm{[-]}$')
plt.legend()
plt.savefig(folder_path + '/deta_dr_%s_%i.pdf' % (gradient_routine, gradient_order), bbox_inches='tight')

# 2D comparison
Dxi_Dz = np.zeros((Nz, Nr))
Deta_Dr = np.zeros((Nz, Nr))
for i in range(Nz):
    for j in range(Nr):
        Dxi_Dz[i, j] = dxi_dz[i]
        Deta_Dr[i, j] = deta_dr[j]

R, Z = np.meshgrid(r, z)
R /= x_ref
Z /= x_ref

#percentage errors matrices
dxi_dz_err = (Dxi_Dz - sun_obj.dxdz)/np.linalg.norm(Dxi_Dz)
deta_dr_err = (Deta_Dr - sun_obj.dydr)/np.linalg.norm(Deta_Dr)


plt.figure()
plt.contourf(Z, R, dxi_dz_err*100, levels = 100, cmap='jet')
plt.xlabel(r'$\hat{z} \ \mathrm{[-]}$')
plt.ylabel(r'$\hat{r} \ \mathrm{[-]}$')
plt.title(r'$(\partial \xi / \partial \hat{z})_{err} \ \mathrm{[\%]}$')
plt.colorbar()
plt.savefig(folder_path + '/dxi_dz_2d_%s_%i.pdf' % (gradient_routine, gradient_order), bbox_inches='tight')


plt.figure()
plt.contourf(Z, R, deta_dr_err*100, levels = 100, cmap='jet')
plt.xlabel(r'$\hat{z} \ \mathrm{[-]}$')
plt.ylabel(r'$\hat{r} \ \mathrm{[-]}$')
plt.title(r'$(\partial \eta / \partial \hat{r})_{err} \ \mathrm{[\%]}$')
plt.colorbar()
plt.savefig(folder_path + '/deta_dr_2d_%s_%i.pdf' % (gradient_routine, gradient_order), bbox_inches='tight')


# plt.show()
