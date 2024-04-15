import pickle

pickle_mesh = 'mesh_21_21_26.pickle'

with open(pickle_mesh, 'rb') as f:
    data = pickle.load(f)

X = data['x']
Y = data['y']
Z = data['z']

Ni = X.shape[0]
Nj = X.shape[1]
Nk = X.shape[2]

# Set the VTK type for the interior elements and the boundary elements
KindElem = 12  # Hexahedral
KindBound = 9  # Quadrilateral

Mesh_File = open(pickle_mesh.split('.')[0] + '.su2', "w")

# Write the dimension of the problem and the number of interior elements
Mesh_File.write("%\n")
Mesh_File.write("% Problem dimension\n")
Mesh_File.write("%\n")
Mesh_File.write("NDIME= 3\n")
Mesh_File.write("%\n")
Mesh_File.write("% Inner element connectivity\n")
Mesh_File.write("%\n")
Mesh_File.write("NELEM= %s\n" % ((Ni - 1) * (Nj - 1) * (Nk - 1)))

# Write the element connectivity. Element -> oriented nodes (8) -> element identification
iElem = 0
for iNode in range(Ni - 1):
    for jNode in range(Nj - 1):
        for kNode in range(Nk - 1):
            elemID = iNode * Nj * Nk + jNode * Nk + kNode  # element identifier
            seID = elemID  # south-east node identifier, on the frontal face
            swID = (iNode + 1) * Nj * Nk + jNode * Nk + kNode  # south-west node identifier, on the frontal face
            nwID = (iNode + 1) * Nj * Nk + (jNode + 1) * Nk + kNode  # north-west node identifier, on the frontal face
            neID = iNode * Nj * Nk + (jNode + 1) * Nk + kNode  # north-east node identifier, on the frontal face
            seIDb = iNode * Nj * Nk + jNode * Nk + kNode + 1  # south-east node identifier, on the back face
            swIDb = (iNode + 1) * Nj * Nk + jNode * Nk + kNode + 1  # south-west node identifier, on the back face
            nwIDb = (iNode + 1) * Nj * Nk + (jNode + 1) * Nk + kNode + 1  # north-west node identifier, on the back face
            neIDb = iNode * Nj * Nk + (jNode + 1) * Nk + kNode + 1  # north-east node identifier, on the back face
            Mesh_File.write("%s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s\n" % (
                KindElem, elemID, seID, swID, neID, nwID, seIDb, swIDb, nwIDb, neIDb))
            print(KindElem, elemID, seID, swID, neID, nwID, seIDb, swIDb, nwIDb, neIDb)

# Compute the number of nodes and write the node coordinates
nPoint = Ni * Nj * Nk
Mesh_File.write("%\n")
Mesh_File.write("% Node coordinates\n")
Mesh_File.write("%\n")
Mesh_File.write("NPOIN= %s\n" % nPoint)
iPoint = 0
for iNode in range(Ni):
    for jNode in range(Nj):
        for kNode in range(Nk):
            Mesh_File.write("%15.14f \t %15.14f \t %15.14f \t %s\n" % (
                X[iNode, jNode, kNode], Y[iNode, jNode, kNode], Z[iNode, jNode, kNode], iPoint))
            iPoint = iPoint + 1

# Write the header information for the boundary markers
Mesh_File.write("%\n")
Mesh_File.write("% Boundary elements\n")
Mesh_File.write("%\n")
Mesh_File.write("NMARK= 4\n")

# Write the boundary information for each marker
Mesh_File.write("MARKER_TAG= lower\n")
Mesh_File.write("MARKER_ELEMS= %s\n" % (Ni * Nk))
for iNode in range(Ni - 1):
    for kNode in range(Nk - 1):
        jNode = 0
        elemID = iNode * Nj * Nk + jNode * Nk + kNode
        seID = elemID
        swID = (iNode + 1) * Nj * Nk + jNode * Nk + kNode
        seIDb = iNode * Nj * Nk + jNode * Nk + kNode + 1
        swIDb = (iNode + 1) * Nj * Nk + jNode * Nk + kNode + 1
        Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, seID, swID, seIDb, swIDb))

Mesh_File.write("MARKER_TAG= upper\n")
Mesh_File.write("MARKER_ELEMS= %s\n" % (Ni * Nk))
for iNode in range(Ni - 1):
    for kNode in range(Nk - 1):
        jNode = Nj - 1
        elemID = iNode * Nj * Nk + jNode * Nk + kNode
        nwID = (iNode + 1) * Nj * Nk + (jNode + 1) * Nk + kNode
        neID = iNode * Nj * Nk + (jNode + 1) * Nk + kNode
        nwIDb = (iNode + 1) * Nj * Nk + (jNode + 1) * Nk + kNode + 1
        neIDb = iNode * Nj * Nk + (jNode + 1) * Nk + kNode + 1
        Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, neID, nwID, nwIDb, neIDb))

Mesh_File.write("MARKER_TAG= left\n")
Mesh_File.write("MARKER_ELEMS= %s\n" % (Ni * Nk))
for jNode in range(Nj - 1):
    for kNode in range(Nk - 1):
        iNode = 0
        elemID = iNode * Nj * Nk + jNode * Nk + kNode
        seID = elemID
        neID = iNode * Nj * Nk + (jNode + 1) * Nk + kNode
        seIDb = iNode * Nj * Nk + jNode * Nk + kNode + 1
        neIDb = iNode * Nj * Nk + (jNode + 1) * Nk + kNode + 1
        Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, seID, neID, neIDb, seIDb))

Mesh_File.write("MARKER_TAG= right\n")
Mesh_File.write("MARKER_ELEMS= %s\n" % (Ni * Nk))
for jNode in range(Nj - 1):
    for kNode in range(Nk - 1):
        iNode = Ni - 1
        elemID = iNode * Nj * Nk + jNode * Nk + kNode
        swID = (iNode + 1) * Nj * Nk + jNode * Nk + kNode
        nwID = (iNode + 1) * Nj * Nk + (jNode + 1) * Nk + kNode
        swIDb = (iNode + 1) * Nj * Nk + jNode * Nk + kNode + 1
        nwIDb = (iNode + 1) * Nj * Nk + (jNode + 1) * Nk + kNode + 1
        Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, swID, nwID, nwIDb, swIDb))

Mesh_File.close()
