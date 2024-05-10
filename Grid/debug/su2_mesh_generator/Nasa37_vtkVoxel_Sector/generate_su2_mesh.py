import pickle
import numpy as np
from Grid.src.su2_mesh_generator import generate_SU2mesh

pickle_mesh = 'mesh_68_30_15.pickle'
with open(pickle_mesh, 'rb') as f:
    data = pickle.load(f)
su2_meshName = pickle_mesh.split('.')[0] + '.su2'

X = data['x']
Y = data['y']
Z = data['z']

KindElem = 12  # Quad
KindBound = 9  # Line
generate_SU2mesh(X, Y, Z, kind_elem=KindElem, kind_bound=KindBound, full_annulus=False, filename=su2_meshName)



# # Set the VTK type for the interior elements and the boundary elements

#
# # Store the number of nodes and open the output mesh file
# nNode = X.shape[0]
# mNode = X.shape[1]
# lNode = X.shape[2]
# Mesh_File = open(su2_meshName, "w")
#
# # Write the dimension of the problem and the number of interior elements
# Mesh_File.write("%\n")
# Mesh_File.write("% Problem dimension\n")
# Mesh_File.write("%\n")
# Mesh_File.write("NDIME= 3\n")
# Mesh_File.write("%\n")
# Mesh_File.write("% Inner element connectivity\n")
# Mesh_File.write("%\n")
# Mesh_File.write("NELEM= %s\n" % ((nNode - 1) * (mNode - 1) * (lNode - 1)))
#
# # Write the element connectivity
# iElem = 0
# for iNode in range(nNode - 1):
#     for jNode in range(mNode - 1):
#         for kNode in range(lNode - 1):
#             zero = kNode + jNode*lNode + iNode*lNode*mNode
#             one = kNode + (jNode+1)*lNode + iNode*lNode*mNode
#             two = kNode + (jNode+1)*lNode + (iNode+1)*lNode*mNode
#             three = kNode + jNode*lNode + (iNode+1)*lNode*mNode
#             four = kNode+1 + jNode*lNode + iNode*lNode*mNode
#             five = kNode+1 + (jNode+1)*lNode + iNode*lNode*mNode
#             six = kNode+1 + (jNode+1)*lNode + (iNode+1)*lNode*mNode
#             seven = kNode+1 + jNode*lNode + (iNode+1)*lNode*mNode
#             Mesh_File.write("%s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s\n"
#                             % (KindElem, zero, one, two, three, four, five, six, seven))
#             iElem = iElem + 1
#
# # Compute the number of nodes and write the node coordinates
# nPoint = nNode * mNode * lNode
# Mesh_File.write("%\n")
# Mesh_File.write("% Node coordinates\n")
# Mesh_File.write("%\n")
# Mesh_File.write("NPOIN= %s\n" % (nNode * mNode * lNode))
# iPoint = 0
# for iNode in range(nNode):
#     for jNode in range(mNode):
#         for kNode in range(lNode):
#             Mesh_File.write("%15.14f \t %15.14f \t %15.14f \t %s\n" % (X[iNode, jNode, kNode],
#                                                                        Y[iNode, jNode, kNode],
#                                                                        Z[iNode, jNode, kNode],
#                                                                        iPoint))
#             iPoint = iPoint + 1
#
# # Write the header information for the boundary markers
# Mesh_File.write("%\n")
# Mesh_File.write("% Boundary elements\n")
# Mesh_File.write("%\n")
# Mesh_File.write("NMARK= 4\n")
#
# # Write the boundary information for each marker
# Mesh_File.write("MARKER_TAG= HUB\n")
# Mesh_File.write("MARKER_ELEMS= %s\n" % ((nNode - 1)*(lNode - 1)))
# for iNode in range(nNode - 1):
#     for kNode in range(lNode - 1):
#         jNode = 0
#         zero = kNode + jNode*lNode + iNode*lNode*mNode
#         three = kNode + jNode*lNode + (iNode+1)*lNode*mNode
#         four = kNode+1 + jNode*lNode + iNode*lNode*mNode
#         seven = kNode+1 + jNode*lNode + (iNode+1)*lNode*mNode
#         Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, zero, four, seven, three))
#
# Mesh_File.write("MARKER_TAG= OUTLET\n")
# Mesh_File.write("MARKER_ELEMS= %s\n" % ((mNode - 1)*(lNode - 1)))
# for jNode in range(mNode - 1):
#     for kNode in range(lNode - 1):
#         iNode = nNode-2
#         two = kNode + (jNode+1)*lNode + (iNode+1)*lNode*mNode
#         three = kNode + jNode*lNode + (iNode+1)*lNode*mNode
#         six = kNode+1 + (jNode+1)*lNode + (iNode+1)*lNode*mNode
#         seven = kNode+1 + jNode*lNode + (iNode+1)*lNode*mNode
#         Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, two, six, seven, three))
#
# Mesh_File.write("MARKER_TAG= SHROUD\n")
# Mesh_File.write("MARKER_ELEMS= %s\n" % ((nNode - 1)*(lNode -1)))
# for iNode in range(nNode - 1):
#     for kNode in range(lNode - 1):
#         jNode = mNode-2
#         one = kNode + (jNode + 1) * lNode + iNode * lNode * mNode
#         two = kNode + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
#         five = kNode + 1 + (jNode + 1) * lNode + iNode * lNode * mNode
#         six = kNode + 1 + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
#         Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, one, two, six, five))
#
# Mesh_File.write("MARKER_TAG= INLET\n")
# Mesh_File.write("MARKER_ELEMS= %s\n" % ((mNode - 1)*(lNode - 1)))
# for jNode in range(mNode - 1):
#     for kNode in range(lNode - 1):
#         iNode = 0
#         zero = kNode + jNode * lNode + iNode * lNode * mNode
#         one = kNode + (jNode + 1) * lNode + iNode * lNode * mNode
#         four = kNode + 1 + jNode * lNode + iNode * lNode * mNode
#         five = kNode + 1 + (jNode + 1) * lNode + iNode * lNode * mNode
#         Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, zero, one, five, four))
#
# Mesh_File.write("MARKER_TAG= PER0\n")
# Mesh_File.write("MARKER_ELEMS= %s\n" % ((mNode - 1)*(nNode - 1)))
# for iNode in range(nNode - 1):
#     for jNode in range(mNode - 1):
#         kNode = 0
#         zero = kNode + jNode*lNode + iNode*lNode*mNode
#         one = kNode + (jNode+1)*lNode + iNode*lNode*mNode
#         two = kNode + (jNode+1)*lNode + (iNode+1)*lNode*mNode
#         three = kNode + jNode*lNode + (iNode+1)*lNode*mNode
#         Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, zero, three, two, one))
#
# Mesh_File.write("MARKER_TAG= PER1\n")
# Mesh_File.write("MARKER_ELEMS= %s\n" % ((mNode - 1)*(nNode - 1)))
# for iNode in range(nNode - 1):
#     for jNode in range(mNode - 1):
#         kNode = lNode - 2
#         four = kNode + 1 + jNode * lNode + iNode * lNode * mNode
#         five = kNode + 1 + (jNode + 1) * lNode + iNode * lNode * mNode
#         six = kNode + 1 + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
#         seven = kNode + 1 + jNode * lNode + (iNode + 1) * lNode * mNode
#         Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, four, seven, six, five))
#
#
#
# # Close the mesh file and exit
# Mesh_File.close()