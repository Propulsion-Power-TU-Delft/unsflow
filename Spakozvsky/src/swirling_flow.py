from .functions import Trad_n
from numpy import pi


class SwirlingFlow:
    """
    this class contains the swirling flow model for the Spakovszky model
    """
    def __init__(self, r_1, ur_1, ut_1, r_eval):
        """

        Args:
            r_1: location where data are given, usually inner radius
            ur_1: radial velocity
            ut_1: tangential velocity
            r_eval: evaluation point of the swirling flow dynamics
        """

        self.r_1 = r_1
        self.ur_1 = ur_1
        self.ut_1 = ut_1
        self.Q = 2 * pi * r_1 * ur_1
        self.GAMMA = 2 * pi * r_1 * ut_1
        self.r_eval = r_eval




    def transfer_function(self, s, n, theta=0):
        """
        compute the component transfer function
        Args:
            r: radial cordinate
            theta: tangential cordinate
            s: laplace variable
            n: circumferential harmonic

        Returns:
            M: swirling flow transfer function at a given location (r, theta, s, n)

        """
        M = Trad_n(self.r_eval, self.r_1, n, s, self.Q, self.GAMMA, theta)
        return M
