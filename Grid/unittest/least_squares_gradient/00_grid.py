import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import pickle
from scipy.optimize import fsolve
from TurboBFM.Preprocess.grid_generation import transfinite_grid_generation


"""
Test case taken from section 11.5.2 of Numerical Computation of Internal and External Flows: 
The Fundamentals of Computational Fluid Dynamics (Second Edition) by Charles Hirsch.
"""

OUTPUT_FOLDER = 'Grid'
L = 1
NX = 128*3
NY = 64*3
SPAN = L/10
NZ = NX//10
STREAMWISE_COEFF = 2
SPANWISE_COEFF = 2
BUMP_PENETRATION = 0.2*L




def func(alpha):
    y = np.cos(alpha)/2/np.sin(alpha) + BUMP_PENETRATION-1/2/np.sin(alpha)
    return y
alpha = fsolve(func, 1)[0]
r_bump = 1/2/np.sin(alpha)

NX_bump = NX//3

x1 = np.linspace(0, L, NX_bump)
y1 = np.zeros_like(x1)

theta = np.linspace(0, 2*alpha, NX_bump)
x2 = 1.5*L + r_bump*np.cos(np.pi/2+alpha-theta)
y2 = -(r_bump-BUMP_PENETRATION)+r_bump*np.sin(np.pi/2+alpha-theta)

x3 = np.linspace(2*L, 3*L, NX_bump)
y3 = np.zeros_like(x3)

# three blocks
x_wall = [x1, x2, x3]
y_wall = [y1, y2, y3]

x_inlet = [np.zeros(NY),
           np.zeros(NY)+x1[-1],
           np.zeros(NY)+x2[-1]]
y_inlet = [np.linspace(0, L, NY),
           np.linspace(0, L, NY),
           np.linspace(0, L, NY)]

x_outlet = [np.zeros(NY)+x1[-1],
           np.zeros(NY)+x2[-1],
           np.zeros(NY)+x3[-1]]
y_outlet = [np.linspace(0, L, NY),
           np.linspace(0, L, NY),
           np.linspace(0, L, NY)]

x_up = [np.linspace(0, L, NX_bump),
        np.linspace(L, 2*L, NX_bump),
        np.linspace(2*L, 3*L, NX_bump)]
y_up = [np.zeros(NX_bump)+L,
        L-y2,
        np.zeros(NX_bump)+L]

Xmulti, Ymulti = [], []
stretch_stream = ['right', 'both', 'left']
stretch_span = ['both', 'both', 'both']
for i in range(3):
    xgrid, ygrid = transfinite_grid_generation(np.vstack((x_inlet[i], y_inlet[i])), 
                                               np.vstack((x_wall[i], y_wall[i])), 
                                               np.vstack((x_outlet[i], y_outlet[i])), 
                                               np.vstack((x_up[i], y_up[i])),
                                               stretch_type_stream=stretch_stream[i], stretch_type_span=stretch_span[i],
                                               streamwise_coeff=STREAMWISE_COEFF, spanwise_coeff=SPANWISE_COEFF)
    Xmulti.append(xgrid)
    Ymulti.append(ygrid)

# aseemble a single block
X = np.concatenate((Xmulti[0], Xmulti[1][1:,:], Xmulti[2][1:,:]), axis=0)
Y = np.concatenate((Ymulti[0], Ymulti[1][1:,:], Ymulti[2][1:,:]), axis=0)

X = Xmulti[1]
Y = Ymulti[1]
nx,ny = X.shape
geometry = {'X': X, 'Y': Y}
with open('grid_%i_%i.pik' %(nx,ny), 'wb') as file:
    pickle.dump(geometry, file)


plt.show()







