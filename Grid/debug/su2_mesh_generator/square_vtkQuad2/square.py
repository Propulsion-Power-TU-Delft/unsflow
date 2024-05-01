#!/usr/bin/python

## \file square.py
#  \brief Python script for creating a .su2 mesh of a simple square domain.
#  \author Thomas D. Economon
#  \version 1.0.

# Import the option parser and parse input options
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename", default="square.su2", help="write mesh to FILE", metavar="FILE")
parser.add_option("-n", "--nNode", dest="nNode", default=10, help="use this NNODE in x direction", metavar="NNODE")
parser.add_option("-m", "--mNode", dest="mNode", default=10, help="use this MNODE in x direction", metavar="MNODE")
(options, args) = parser.parse_args()

# Set the VTK type for the interior elements and the boundary elements
KindElem = 9  # Quad
KindBound = 3  # Line

# Store the number of nodes and open the output mesh file
nNode = int(options.nNode)
mNode = int(options.mNode)
Mesh_File = open(options.filename, "w")

# Write the dimension of the problem and the number of interior elements
Mesh_File.write("%\n")
Mesh_File.write("% Problem dimension\n")
Mesh_File.write("%\n")
Mesh_File.write("NDIME= 2\n")
Mesh_File.write("%\n")
Mesh_File.write("% Inner element connectivity\n")
Mesh_File.write("%\n")
Mesh_File.write("NELEM= %s\n" % (1 * (nNode - 1) * (mNode - 1)))

# Write the element connectivity
iElem = 0
for iNode in range(nNode - 1):
    for jNode in range(mNode - 1):
        zero = iNode * mNode + jNode
        one = (iNode + 1) * mNode + jNode
        two = (iNode + 1) * mNode + jNode + 1
        three = iNode * mNode + jNode + 1
        Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindElem, zero, one, two, three))
        iElem = iElem + 1

# Compute the number of nodes and write the node coordinates
nPoint = (nNode) * (mNode)
Mesh_File.write("%\n")
Mesh_File.write("% Node coordinates\n")
Mesh_File.write("%\n")
Mesh_File.write("NPOIN= %s\n" % ((nNode) * (mNode)))
iPoint = 0
for iNode in range(nNode):
    for jNode in range(mNode):
        Mesh_File.write("%15.14f \t %15.14f \t %s\n" % (float(iNode) / float(nNode - 1), float(jNode) / float(mNode - 1), iPoint))
        iPoint = iPoint + 1

# Write the header information for the boundary markers
Mesh_File.write("%\n")
Mesh_File.write("% Boundary elements\n")
Mesh_File.write("%\n")
Mesh_File.write("NMARK= 4\n")

# Write the boundary information for each marker
Mesh_File.write("MARKER_TAG= lower\n")
Mesh_File.write("MARKER_ELEMS= %s\n" % (nNode - 1))
for iNode in range(nNode - 1):
    Mesh_File.write("%s \t %s \t %s\n" % (KindBound, iNode * mNode, (iNode + 1) * mNode))
Mesh_File.write("MARKER_TAG= right\n")
Mesh_File.write("MARKER_ELEMS= %s\n" % (mNode - 1))
for jNode in range(mNode - 1):
    Mesh_File.write("%s \t %s \t %s\n" % (KindBound, jNode + (nNode-1)*mNode, jNode + 1 + (nNode-1)*mNode))
Mesh_File.write("MARKER_TAG= upper\n")
Mesh_File.write("MARKER_ELEMS= %s\n" % (nNode - 1))
for iNode in range(nNode - 1):
    Mesh_File.write("%s \t %s \t %s\n" % (KindBound, iNode * mNode + (mNode-1), (iNode + 1) * mNode + (mNode-1)))
Mesh_File.write("MARKER_TAG= left\n")
Mesh_File.write("MARKER_ELEMS= %s\n" % (mNode - 1))
for jNode in range(mNode - 1):
    Mesh_File.write("%s \t %s \t %s\n" % (KindBound, jNode, jNode + 1))

# Close the mesh file and exit
Mesh_File.close()
