from .functions import *


class AxialDuct:
    """
    this class contains the axial duct component info for the spakovszky model
    """
    def __init__(self, ut, uz):
        """
        take care of providing normalized quantities
        Args:
            ut: tangential velocity
            uz: axial velocity
        """
        self.ut = ut
        self.uz = uz


    def transfer_function(self, z, theta, s, n):
        """
        compute the component transfer function
        Args:
            z: axial cordinate
            theta: tangential cordinate
            s: laplace variable
            n: circumferential harmonic

        Returns:
            M: axial inlet duct transfer function

        """
        M = Tax_n(z, s, n, self.uz, self.ut, theta)
        return M
