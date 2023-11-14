import numpy as np
import matplotlib.pyplot as plt
import Grid
from Grid.src.functions import elliptic_grid_generation

""" 
test case to debug the elliptic grid generation method. Define a figure delimited by 4 curved
borders, and see how it goes
"""

nx = 30
ny = 30

# parameteric picture
L = 1
H = 1


def non_linear_spacing(start, stop, num_points, exponent=2):
    # Generate non-linearly spaced points
    spacing = np.linspace(0, 1, num_points) ** exponent
    scaled_spacing = spacing * (stop - start)
    return start + scaled_spacing


# Use the non-linear spacing function
non_linear_points = non_linear_spacing(0, H, ny, exponent=2)
# left border
c_left = np.array((np.zeros(ny),
                   non_linear_spacing(0, H, ny, exponent=2)))

# bottom border
c_bottom = np.array((non_linear_spacing(0, L, nx, exponent=2),
                     np.zeros(nx)))

# right border
c_right = np.array((np.zeros(ny) + L,
                    non_linear_spacing(0, H, ny, exponent=2)))

# top border
c_top = np.array((non_linear_spacing(0, L, nx, exponent=2),
                  np.zeros(nx) + H))

orthogonality = True
x_stretching = 'sigmoid'
y_stretching = 'sigmoid'
X, Y = elliptic_grid_generation(c_left, c_bottom, c_right, c_top, orthogonality=True, sigmoid_coeff_x=8,
                                sigmoid_coeff_y=8, x_stretching='sigmoid_left', y_stretching='sigmoid_down', tol=1e-3,
                                save_filename='orth_%s__xstr_%s__ystr_%s' % (orthogonality, x_stretching, y_stretching),
                                method='intersection')
