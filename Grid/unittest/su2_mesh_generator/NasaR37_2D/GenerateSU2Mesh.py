import pickle
import numpy as np
from grid.src.su2_mesh_generator import generate_SU2mesh

pickle_mesh = 'mesh_68_30_10.pickle'
with open(pickle_mesh, 'rb') as f:
    data = pickle.load(f)
su2_meshName = pickle_mesh.split('.')[0] + '.su2'

X = data['x']
Y = data['y']
Z = data['z']
R = np.sqrt(X**2+Y**2)

KindElem = 9  # Quad
KindBound = 3  # Line
generate_SU2mesh(Z, R, kind_elem=KindElem, kind_bound=KindBound, full_annulus=False, filename=su2_meshName)
