from .functions import Bgap_n


class AxialGap:
    """
    this class contains the axial gap component info for the spakovszky model
    """
    def __init__(self, z_1, z_2, uz, ut):
        """
        take care of providing normalized quantities
        Args:
            z_1: axial cordinate at beginning of the gap
            z_2: axial cordinate at end of the gap
            ut: tangential velocity
            uz: axial velocity
        """
        self.z_1 = z_1
        self.z_2 = z_2
        self.ut = ut
        self.uz = uz


    def transfer_function(self, theta, s, n):
        """
        compute the component transfer function
        Args:
            theta: tangential cordinate
            s: laplace variable
            n: circumferential harmonic

        Returns:
            M: axial gap transfer function

        """
        M = Bgap_n(self.z_1, self.z_2, s, n, self.uz, self.ut, theta)
        return M
