from .functions import Bgap_n
from unsflow.utils.formatting import print_banner
from unsflow.utils.formatting import total_chars, total_chars_mid
from numpy import pi as pi

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
        self.print_info()


    def transfer_function(self, s, n, theta=0):
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

    def print_info(self):
        """
        Print the information of the component
        """
        print_banner('AXIAL GAP')
        print(f"{'Inlet Coordinate [-]:':<{total_chars_mid}}{self.z_1:>{total_chars_mid}.2f}")
        print(f"{'Outlet Coordinate [-]:':<{total_chars_mid}}{self.z_2:>{total_chars_mid}.2f}")
        print(f"{'Axial Velocity [-]:':<{total_chars_mid}}{self.uz:>{total_chars_mid}.2f}")
        print(f"{'Tang. Velocity [-]:':<{total_chars_mid}}{self.ut:>{total_chars_mid}.2f}")
        print_banner()
