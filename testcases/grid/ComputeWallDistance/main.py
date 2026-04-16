import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from unsflow.grid.functions import contour_template, compute_2dSpline_curve, computeMinDistanceFromWalls

N = 5000


    
inputFile = "lscc.csv"
with open(inputFile, 'r') as f:
        ni = int(f.readline().strip().split('=')[1])
        nj = int(f.readline().strip().split('=')[1])
        nk = int(f.readline().strip().split('=')[1])
    
    
df = pd.read_csv(inputFile, skiprows=3)
data = {col: df[col].to_numpy().reshape((ni, nj)) for col in df.columns}


data['ax'] = data['x']
data['rad'] = np.sqrt(data['y']**2+data['z']**2)

hub = (data['ax'][:,0], data['rad'][:,0])
shroud = (data['ax'][:,-1], data['rad'][:,-1])

data["WallDistance"] = np.zeros((ni, nj))
for i in range(ni):
    for j in range(1, nj-1):
        print("Progress: %i/%i" % ((nj-2)*i+j, ni*(nj-2)))
        data['WallDistance'][i,j] = computeMinDistanceFromWalls((data['ax'][i,j], data['rad'][i,j]), hub, shroud, N)
        



plt.figure()
plt.contourf(data['ax'], data['rad'], data['WallDistance'], levels=20, cmap='turbo')
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Wall Distance')




# plt.show()