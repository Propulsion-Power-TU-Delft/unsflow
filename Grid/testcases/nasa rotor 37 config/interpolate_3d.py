import numpy as np
from numpy import pi
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

b = time.time()


# Define the 3D function f(x, y, z)
def f(x, y, z):
    return np.sin(x*y*z)


# Generate random data points for interpolation
np.random.seed(42)
N = 10
x_data = np.linspace(0, 1, N)
y_data = np.linspace(0, 2, N)
z_data = np.linspace(0, 3, N)
X, Y, Z = np.meshgrid(x_data, y_data, z_data, indexing='ij')
values = f(X, Y, Z)

N_eval = 30
x_eval = np.linspace(np.min(x_data), np.max(x_data), N_eval)
y_eval = np.linspace(np.min(y_data), np.max(y_data), N_eval)
z_eval = np.linspace(np.min(z_data), np.max(z_data), N_eval)
X_eval, Y_eval, Z_eval = np.meshgrid(x_eval, y_eval, z_eval, indexing='ij')
interp_values = griddata((X.flatten(), Y.flatten(), Z.flatten()), values.flatten(),
                          (X_eval, Y_eval, Z_eval), method='linear')
interp_values.reshape(np.shape(X_eval))


# compute error of the interpolated data with respect to the real function values
truth_vales = f(X_eval, Y_eval, Z_eval)
abs_error = truth_vales-interp_values
rel_error = abs_error/np.linalg.norm(truth_vales)

# Plot absolute and rel error
fig = plt.figure(figsize=(12, 6))

# Plot original data
ax1 = fig.add_subplot(121, projection='3d')
scatter = ax1.scatter(X_eval, Y_eval, Z_eval, c=abs_error, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Absolute error')
c = plt.colorbar(scatter, ax=ax1)

# Plot interpolated data
ax2 = fig.add_subplot(122, projection='3d')
scatter = ax2.scatter(X_eval, Y_eval, Z_eval, c=rel_error, cmap='viridis')
ax2.set_title('Relative Error')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
c = plt.colorbar(scatter, ax=ax2)







# Plot original and interpolated data
fig = plt.figure(figsize=(12, 6))

# Plot original data
ax1 = fig.add_subplot(121, projection='3d')
scatter = ax1.scatter(X, Y, Z, c=values, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Original Data')
c = plt.colorbar(scatter, ax=ax1)

# Plot interpolated data
ax2 = fig.add_subplot(122, projection='3d')
scatter = ax2.scatter(X_eval, Y_eval, Z_eval, c=interp_values, cmap='viridis')
ax2.set_title('Interpolated Data')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
c = plt.colorbar(scatter, ax=ax2)



e = time.time()
print("total time: %.2f" % (e - b))
plt.show()
