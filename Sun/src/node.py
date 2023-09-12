class Node:
    """
    Class of Node object, contaning cordinates, fluid dynamics field, markers and cordinates of the grid point
    """

    def __init__(self, z, r, marker, nodeCounter):
        self.z = z  # axial cordinate
        self.r = r  # radial cordinate
        self.marker = marker  # type of node if belonging to boundaries
        self.nodeCounter = nodeCounter  # identifier of the node

    def AppendDensityInfo(self, rho, drho_dr, drho_dz):
        """
        add density related information to the node
        """
        self.rho = rho
        self.drho_dr = drho_dr
        self.drho_dz = drho_dz

    def AppendVelocityInfo(self, ur, ut, uz, dur_dr, dur_dz, dut_dr, dut_dz, duz_dr, duz_dz):
        """
        add velocity related information to the node
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
        add pressure related information to the node
        """
        self.p = p
        self.dp_dr = dp_dr
        self.dp_dz = dp_dz

    def PrintInfo(self, datafile='terminal'):
        """
        print some info of the node
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
        it normalizes all the data belonging to the nodes, with the reference quantities given in the problem
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
        adds the normal vector information to the node (used only for hub and shroud nodes)
        """
        self.n_wall = n

    def AddAMatrix(self, A):
        """
        It adds the A matrix, already multiplied by omega and j at the node level
        """
        self.A = A

    def AddBMatrix(self, B):
        """
        It adds the B matrix at the node level
        """
        self.B = B

    def AddCMatrix(self, C):
        """
        It adds the C matrix, already multiplied by m and j at the node level
        """
        self.C = C

    def AddEMatrix(self, E):
        """
        It adds the E matrix at the node level
        """
        self.E = E

    def AddRMatrix(self, R):
        """
        It adds the R matrix at the node level
        """
        self.R = R

    def AddSMatrix(self, S):
        """
        It adds the S matrix at the node level
        """
        self.S = S

    def AddTransformationGradients(self, dzdx, dzdy, drdx, drdy):
        """
        It adds the physical jacobian as a function of the  spectral cordinates at the node level
        """
        self.dzdx, self.dzdy, self.drdx, self.drdy = dzdx, dzdy, drdx, drdy

    def AddJacobian(self, J):
        """
        It adds the inverse jacobian as a function of the  spectral cordinates at the node level
        """
        self.J = J

    def AddHatMatrices(self, Bhat, Ehat):
        """
        It adds the hat{B}, hat{E} matrix at the node level
        """
        self.Bhat, self.Ehat = Bhat, Ehat
