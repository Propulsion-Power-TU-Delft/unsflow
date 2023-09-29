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
L = 2
R = 1

"""  left border  """
theta = np.linspace(0, np.pi, ny)
x_left = np.zeros(ny)
y_left = R-R*np.cos(theta)
c_left = np.array((x_left,
                   y_left))

"""  bottom border  """
c_bottom = np.array((np.linspace(0, L, nx),
                     np.zeros(nx)))

"""  right border  """
theta = np.linspace(0, 2*np.pi, ny)
x_right = L+R/5*np.sin(theta)
y_right = np.linspace(0, 2*R, ny)
c_right = np.array((x_right,
                   y_right))

"""  top border  """
c_top = np.array((np.linspace(0, L, nx),
                np.zeros(nx)+2*R))


orthogonality = False
x_stretching = 'sigmoid'
y_stretching = 'sigmoid'


X, Y = elliptic_grid_generation(c_left, c_bottom, c_right, c_top, orthogonality=orthogonality, sigmoid_coeff=7,
                                x_stretching=x_stretching, y_stretching=y_stretching, tol=1e-3,
                                save_filename='orth_%s__xstr_%s__ystr_%s' %(orthogonality, x_stretching, y_stretching))
