from .functions import *
from Sun.src.general_functions import print_banner_begin, print_banner_end
from Utils.styles import total_chars, total_chars_mid


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
        self.print_info()


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

    def print_info(self):
        """
        Print the information of the component
        """
        print_banner_begin('AXIAL DUCT')
        print(f"{'Axial Velocity [-]:':<{total_chars_mid}}{self.uz:>{total_chars_mid}.2f}")
        print(f"{'Tangential Velocity [-]:':<{total_chars_mid}}{self.ut:>{total_chars_mid}.2f}")
        print(f"{'Evaluation Coordinate [-]:':<{total_chars_mid}}{self.z_eval:>{total_chars_mid}.2f}")
        print_banner_end()

