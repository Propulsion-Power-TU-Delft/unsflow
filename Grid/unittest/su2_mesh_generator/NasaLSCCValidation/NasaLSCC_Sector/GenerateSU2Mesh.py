import pickle
import numpy as np
from Grid.src.su2_mesh_generator import generate_SU2mesh

pickle_mesh = 'mesh_78_20_10.pickle'
with open(pickle_mesh, 'rb') as f:
    data = pickle.load(f)
su2_meshName = pickle_mesh.split('.')[0] + '.su2'

X = data['x']
Y = data['y']
Z = data['z']

KindElem = 12  # Quad
KindBound = 9  # Line
generate_SU2mesh(X, Y, Z, kind_elem=KindElem, kind_bound=KindBound, full_annulus=False, filename=su2_meshName)
