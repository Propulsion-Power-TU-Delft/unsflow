import pickle
import numpy as np

pickle_mesh = 'mesh_21_21_26.pickle'
with open(pickle_mesh, 'rb') as f:
    data = pickle.load(f)

X = data['x']
Y = data['y']
Z = data['z']
R = np.sqrt(X**2 + Y**2)

Ni = X.shape[0]
Nj = X.shape[1]
print('Number of nodes: [%i, %i], total: %i' %(Ni, Nj, Ni*Nj))
print('Number of elements: [%i, %i], total: %i' %(Ni-1, Nj-1, (Ni-1)*(Nj-1)))


# Set the VTK type for the interior elements and the boundary elements
KindElem = 9  # Quadrilateral
KindBound = 3  # Line

Mesh_File = open(pickle_mesh.split('.')[0] + '2d.su2', "w")

# Write the dimension of the problem and the number of interior elements
Mesh_File.write("%\n")
Mesh_File.write("% Problem dimension\n")
Mesh_File.write("%\n")
Mesh_File.write("NDIME= 2\n")
Mesh_File.write("%\n")
Mesh_File.write("% Inner element connectivity\n")
Mesh_File.write("%\n")
Mesh_File.write("NELEM= %s\n" % ((Ni - 1) * (Nj - 1)))

# Write the element connectivity
dummy = 0
Ni -= 1
Nj -= 1
for ii in range(Ni):
    for jj in range(Nj):
        elemID = ii * Nj + jj  # element identifier
        seID = ii * Nj + jj  # south-east node identifier, on the frontal face
        swID = (ii+1) * Nj + jj  # south-west node identifier, on the frontal face
        nwID = (ii+1) * Nj + jj+1  # north-west node identifier, on the frontal face
        neID = ii * Nj + jj+1  # north-east node identifier, on the frontal face
        Mesh_File.write("%s \t %s \t %s \t %s \t %s \t %s\n" % (
            KindElem, seID, swID, nwID, neID, elemID))
        dummy += 1
print('Written %i elements to mesh file' % dummy)

# Compute the number of nodes and write the node coordinates
Ni += 1
Nj += 1
nPoint = Ni * Nj
Mesh_File.write("%\n")
Mesh_File.write("% Node coordinates\n")
Mesh_File.write("%\n")
Mesh_File.write("NPOIN= %s\n" % nPoint)
iPoint = 0
for iNode in range(Ni):
    for jNode in range(Nj):
        Mesh_File.write("%15.14f \t %15.14f \t %s\n" % (
            Z[iNode, jNode, 0], R[iNode, jNode, 0], iPoint))
        iPoint = iPoint + 1
print('Written %i nodes to mesh file' % iPoint)

# Write the header information for the boundary markers
Mesh_File.write("%\n")
Mesh_File.write("% Boundary elements\n")
Mesh_File.write("%\n")
Mesh_File.write("NMARK= 4\n")

Ni -= 1
Nj -= 1


# Write the boundary information for each marker
Mesh_File.write("MARKER_TAG= lower\n")
Mesh_File.write("MARKER_ELEMS= %s\n" % (Ni))
dummy = 0
for ii in range(Ni):
    jj = 0
    seID = ii * Nj + jj  # south-east node identifier, on the frontal face
    swID = (ii + 1) * Nj + jj  # south-west node identifier, on the frontal face
    Mesh_File.write("%s \t %s \t %s\n" % (KindBound, seID, swID))
    dummy += 1


Mesh_File.write("MARKER_TAG= upper\n")
Mesh_File.write("MARKER_ELEMS= %s\n" % (Ni))
dummy = 0
for ii in range(Ni):
    jj = Nj-1
    neID = ii * Nj + jj+1  # south-east node identifier, on the frontal face
    nwID = (ii + 1) * Nj + jj+1  # south-west node identifier, on the frontal face
    Mesh_File.write("%s \t %s \t %s\n" % (KindBound, neID, nwID))
    dummy += 1

Mesh_File.write("MARKER_TAG= left\n")
Mesh_File.write("MARKER_ELEMS= %s\n" % (Nj))
dummy = 0
for jj in range(Nj):
    ii = 0
    seID = ii * Nj + jj  # south-east node identifier, on the frontal face
    neID = ii * Nj + jj + 1  # north-east node identifier, on the frontal face
    Mesh_File.write("%s \t %s \t %s\n" % (KindBound, seID, neID))
    dummy += 1

Mesh_File.write("MARKER_TAG= right\n")
Mesh_File.write("MARKER_ELEMS= %s\n" % (Nj))
dummy = 0
for jj in range(Nj):
    ii = Ni-1
    swID = (ii + 1) * Nj + jj  # south-west node identifier, on the frontal face
    nwID = (ii + 1) * Nj + jj + 1  # north-west node identifier, on the frontal face
    Mesh_File.write("%s \t %s \t %s\n" % (KindBound, swID, nwID))
    dummy += 1

Mesh_File.close()
