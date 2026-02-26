import copy
import numpy as np

class Node:
    """
    Class of Node object, contaning relevant information, such as local matrices.
    """

    def __init__(self, z, r, marker, nodeCounter):
        """
        Builds the node object.
        :param z: z cordinate
        :param r: r cordinate
        :param marker: type of node, to distinguish boundary conditions
        :param nodeCounter: counter of the node.
        """
        self.marker_types = ['inlet', 'outlet', 'hub', 'shroud', 'internal']

        if not isinstance(z, float):
            raise TypeError("z must be a float")
        if not isinstance(r, float):
            raise TypeError("r must be a float")
        if marker not in self.marker_types:
            raise ValueError("Marker type is not valid")
        if not isinstance(nodeCounter, int):
            raise TypeError("nodeCounter must be an int")

        self.z = z  # axial cordinate
        self.r = r  # radial cordinate
        self.marker = marker  # type of node if belonging to boundaries
        self.nodeCounter = nodeCounter  # identifier of the node


    def AddNormalVersor(self, n):
        """
        Adds the normal vector information to the node (used only for hub and shroud nodes).
        :param n: normal vector in (r,theta,z) ref. frame.
        """
        if n.dtype != np.float64 or n.shape!=(3,):
            raise ValueError("n must have dtype 'float' of size 3")
        self.n_wall = n

    def AddAMatrix(self, A):
        """
        It adds the A matrix.
        :param A: matrix to add
        """
        if A.shape != (5, 5):
            raise ValueError("A must have shape (5, 5)")
        if A.dtype != np.complex128:
            raise ValueError("A must have dtype 'complex128' (float complex)")
        self.A = A

    def AddBMatrix(self, B):
        """
        It adds the B matrix at the node level.
        :param B: matrix to add
        """
        if B.shape != (5, 5):
            raise ValueError("B must have shape (5, 5)")
        if B.dtype != np.complex128:
            raise ValueError("B must have dtype 'complex128' (float complex)")
        self.B = B

    def AddCMatrix(self, C):
        """
        It adds the C matrix, already multiplied by m and j at the node level.
        :param C: matrix to add
        """
        if C.shape != (5, 5):
            raise ValueError("C must have shape (5, 5)")
        if C.dtype != np.complex128:
            raise ValueError("C must have dtype 'complex128' (float complex)")
        self.C = C

    def AddEMatrix(self, E):
        """
        It adds the E matrix at the node level.
        :param E: matrix to add
        """
        if E.shape != (5, 5):
            raise ValueError("E must have shape (5, 5)")
        if E.dtype != np.complex128:
            raise ValueError("E must have dtype 'complex128' (float complex)")
        self.E = E


    def AddRMatrix(self, R):
        """
        It adds the R matrix at the node level.
        :param R: matrix to add
        """
        if R.shape != (5, 5):
            raise ValueError("R must have shape (5, 5)")
        if R.dtype != np.complex128:
            raise ValueError("R must have dtype 'complex128' (float complex)")
        self.R = R

    def AddSMatrix(self, S):
        """
        It adds the S matrix at the node level.
        :param S: matrix to add
        """
        # if S.shape != (5, 5):
        #     raise ValueError("S must have shape (5, 5)")
        # if S.dtype != np.complex128:
        #     raise ValueError("S must have dtype 'complex128' (float complex)")
        self.S = S

    def AddTransformationGradients(self, dzdx, dzdy, drdx, drdy):
        """
        It adds the physical jacobian as a function of the  spectral cordinates at the node level
        :param dzdx: 2D array
        :param dzdy: 2D array
        :param drdx: 2D array
        :param drdy: 2D array
        """
        if not all(isinstance(arg, float) for arg in (dzdx, dzdy, drdx, drdy)):
            raise TypeError("All arguments must be floats or ints")
        self.dzdx, self.dzdy, self.drdx, self.drdy = dzdx, dzdy, drdx, drdy

    def AddJacobian(self, J):
        """
        It adds the inverse jacobian as a function of the  spectral cordinates at the node level.
        :param J: 2D array of Jacobian values
        """
        if not isinstance(J, float):
            raise TypeError("J must be a float")
        self.J = J

    def AddHatMatrices(self, Bhat, Ehat):
        """
        It adds the hat{B}, hat{E} matrix at the node level.
        :param Bhat: 2D array
        :param Ehat: 2D array
        """
        for arg in (Bhat, Ehat):
            if arg.shape != (5, 5):
                raise ValueError("arg must have shape (5, 5)")
            if arg.dtype != np.complex128:
                raise ValueError("arg must have dtype 'complex128' (float complex)")

        self.Bhat = Bhat
        self.Ehat = Ehat
