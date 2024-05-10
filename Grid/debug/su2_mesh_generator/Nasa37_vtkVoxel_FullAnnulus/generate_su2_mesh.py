import pickle
import numpy as np
import matplotlib.pyplot as plt
from Grid.src.su2_mesh_generator import generate_SU2mesh

pickle_mesh = 'mesh_68_30_100.pickle'
with open(pickle_mesh, 'rb') as f:
    data = pickle.load(f)
su2_meshName = pickle_mesh.split('.')[0] + '.su2'



X = data['x']
Y = data['y']
Z = data['z']

generate_SU2mesh(X, Y, Z, kind_elem=12, kind_bound=9, full_annulus=True, filename=su2_meshName)


