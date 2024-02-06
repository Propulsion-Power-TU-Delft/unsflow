class Node:
    """
    Class of Node object, contaning cordinates, fluid dynamics field, markers and cordinates of the grid point.
    """

    def __init__(self, z, r, marker, nodeCounter):
        """
        Builds the node object.
        :param z: z cordinate
        :param r: r cordinate
        :param marker: type of node, to distinguish boundary conditions
        :param nodeCounter: counter of the node.
        """
        self.z = z  # axial cordinate
        self.r = r  # radial cordinate
        self.marker = marker  # type of node if belonging to boundaries
        self.nodeCounter = nodeCounter  # identifier of the node

    def AppendDensityInfo(self, rho, drho_dr, drho_dz):
        """
        Ddd density related information to the node.
        :param rho: density
        :param drho_dr: drho_dr
        :param drho_dz: drho_dz
        """
        self.rho = rho.copy()
        self.drho_dr = drho_dr.copy()
        self.drho_dz = drho_dz.copy()

    def AppendVelocityInfo(self, ur, ut, uz, dur_dr, dur_dz, dut_dr, dut_dz, duz_dr, duz_dz):
        """
        Add velocity related information to the node.
        :param ur: ur
        :param ut: ut
        :param uz: uz
        :param dur_dr: dur_dr
        :param dur_dz: dur_dz
        :param dut_dr: dut_dr
        :param dut_dz: dut_dz
        :param duz_dr: duz_dr
        :param duz_dz: duz_dz
        """
        self.ur = ur
        self.ut = ut
        self.uz = uz
        self.dur_dr = dur_dr
        self.dur_dz = dur_dz
        self.dut_dr = dut_dr
        self.dut_dz = dut_dz
        self.duz_dr = duz_dr
        self.duz_dz = duz_dz

    def AppendPressureInfo(self, p, dp_dr, dp_dz):
        """
        Add pressure related information to the node.
        :param p: pressure
        :param dp_dr: dp_dr
        :param dp_dz: dp_dz
        """
        self.p = p
        self.dp_dr = dp_dr
        self.dp_dz = dp_dz

    def PrintInfo(self, datafile='terminal'):
        """
        Print some info of the node.
        :param datafile: specify terminal or datafile where printing the information.
        """
        if datafile == 'terminal':
            print('marker: ' + self.marker)
            print('r: %.2f' % (self.r))
            print('z: %.2f' % (self.z))
            print('-----------------------------------------------')
        else:
            with open(datafile, 'a') as f:
                print('marker: ' + self.marker, file=f)
                print('r: %.2f' % (self.r), file=f)
                print('z: %.2f' % (self.z), file=f)
                print('-----------------------------------------------', file=f)

    def Normalize(self, rho_ref, u_ref, x_ref):
        """
        It normalizes all the data belonging to the nodes, with the reference quantities given in the problem.
        :param rho_ref: reference density [kg/m3]
        :param u_ref: reference velocity [m/s]
        :param x_ref: reference length [m]
        """
        p_ref = rho_ref * u_ref ** 2  # ref pressure

        # normalize the data
        self.z /= x_ref
        self.r /= x_ref
        self.rho /= rho_ref
        self.ur /= u_ref
        self.ut /= u_ref
        self.uz /= u_ref
        self.p /= p_ref

        # normalize the data gradients
        self.drho_dr /= (rho_ref / x_ref)
        self.drho_dz /= (rho_ref / x_ref)
        self.dur_dr /= (u_ref / x_ref)
        self.dur_dz /= (u_ref / x_ref)
        self.dut_dr /= (u_ref / x_ref)
        self.dut_dz /= (u_ref / x_ref)
        self.duz_dr /= (u_ref / x_ref)
        self.duz_dz /= (u_ref / x_ref)
        self.dp_dr /= (p_ref / x_ref)
        self.dp_dz /= (p_ref / x_ref)

    def AddNormalVersor(self, n):
        """
        Adds the normal vector information to the node (used only for hub and shroud nodes).
        :param n: normal vector in (r,theta,z) ref. frame.
        """
        self.n_wall = n

    def AddAMatrix(self, A):
        """
        It adds the A matrix.
        :param A: matrix to add
        """
        self.A = A.copy()

    def AddBMatrix(self, B):
        """
        It adds the B matrix at the node level.
        :param B: matrix to add
        """
        self.B = B.copy()

    def AddCMatrix(self, C):
        """
        It adds the C matrix, already multiplied by m and j at the node level.
        :param C: matrix to add
        """
        self.C = C.copy()

    def AddEMatrix(self, E):
        """
        It adds the E matrix at the node level.
        :param E: matrix to add
        """
        self.E = E.copy()

    def AddRMatrix(self, R):
        """
        It adds the R matrix at the node level.
        :param R: matrix to add
        """
        self.R = R.copy()

    def AddSMatrix(self, S):
        """
        It adds the S matrix at the node level.
        :param S: matrix to add
        """
        self.S = S.copy()

    def AddTransformationGradients(self, dzdx, dzdy, drdx, drdy):
        """
        It adds the physical jacobian as a function of the  spectral cordinates at the node level
        :param dzdx: 2D array
        :param dzdy: 2D array
        :param drdx: 2D array
        :param drdy: 2D array
        """
        self.dzdx, self.dzdy, self.drdx, self.drdy = dzdx.copy(), dzdy.copy(), drdx.copy(), drdy.copy()

    def AddJacobian(self, J):
        """
        It adds the inverse jacobian as a function of the  spectral cordinates at the node level.
        :param J: 2D array of Jacobian values
        """
        self.J = J.copy()

    def AddHatMatrices(self, Bhat, Ehat):
        """
        It adds the hat{B}, hat{E} matrix at the node level.
        :param Bhat: 2D array
        :param Ehat: 2D array
        """
        self.Bhat = Bhat.copy()
        self.Ehat = Ehat.copy()
