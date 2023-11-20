from .functions import *


class AxialDuct:
    """
    this class contains the axial duct component info for the spakovszky model
    """
    def __init__(self, ut, uz, z_eval):
        """
        take care of providing normalized quantities
        Args:
            ut: tangential velocity
            uz: axial velocity
        """
        self.ut = ut
        self.uz = uz
        self.z_eval = z_eval


    def transfer_function(self, s, n, theta=0):
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
        M = Tax_n(self.z_eval, s, n, self.uz, self.ut, theta)
        return M
