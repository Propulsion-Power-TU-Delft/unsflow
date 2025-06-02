import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Grid.src.functions import computeTwoDimensionalRegression



# 1. Create a known 2D function
def true_function(x, y):
    return x**2/(y-5) + y*x**2

# 2. Generate meshgrid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z_true = true_function(X, Y)

# 3. Sample noisy training data
np.random.seed(0)
x_sample = np.linspace(-3, 3, 10)
y_sample = np.linspace(-3, 3, 10)
X_sample, Y_sample = np.meshgrid(x_sample, y_sample)
Z_sample = true_function(X_sample, Y_sample)

# 4. Compute regression surface
Z_regressed = computeTwoDimensionalRegression(
    degree=12,
    xData=X_sample,
    yData=Y_sample,
    zData=Z_sample,
    xEval=X,
    yEval=Y
)

# 5. Modified Plotting: True Function (contourf) and Normalized Error

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: True function as contour
contour1 = axes[0].contourf(X_sample, Y_sample, Z_sample, cmap='viridis', levels=25)
axes[0].set_title("True Function (Contour)")
fig.colorbar(contour1, ax=axes[0])

# Right: Normalized error as contour
contour2 = axes[1].contourf(X, Y, Z_regressed, cmap='viridis', levels=25)
axes[1].set_title("Regressed Function (Contour)")
fig.colorbar(contour2, ax=axes[1])

plt.tight_layout()
plt.show()
