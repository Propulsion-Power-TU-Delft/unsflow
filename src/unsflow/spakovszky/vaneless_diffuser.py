from .functions import Bvlsd_n
from numpy import pi
from unsflow.utils.formatting import print_banner
from unsflow.utils.formatting import total_chars, total_chars_mid
from numpy import pi as pi


class VanelessDiffuser:
    """
    this class contains the swirling flow model for the Spakovszky model
    """
    def __init__(self, r_1, r_2, ur_1, ut_1):
        """
        provide non-dimensional quantities
        Args:
            r_1: inlet radius
            r_2: outlet radius
            ur_1: radial velocity at inlet
            ut_1: tangential velocity at inlet
        """

        self.r_1 = r_1
        self.r_2 = r_2
        self.ur_1 = ur_1
        self.ut_1 = ut_1
        self.Q = 2 * pi * r_1 * ur_1
        self.GAMMA = 2 * pi * r_1 * ut_1
        self.print_info()




    def transfer_function(self, s, n, theta=0):
        """
        compute the component transfer function
        Args:
            r: radial cordinate
            theta: tangential cordinate
            s: laplace variable
            n: circumferential harmonic

        Returns:
            M: vaneless diffuser transfer function at a given location (r, theta, s, n)

        """
        M = Bvlsd_n(s, n, self.r_1, self.r_2, self.r_1, self.Q, self.GAMMA, theta)
        return M

    def print_info(self):
        """
        Print the information of the component
        """
        print_banner('VANELESS DIFFUSER')
        print(f"{'Inlet Radius [-]:':<{total_chars_mid}}{self.r_1:>{total_chars_mid}.2f}")
        print(f"{'Outlet Radius [-]:':<{total_chars_mid}}{self.r_2:>{total_chars_mid}.2f}")
        print(f"{'Inlet Radial Velocity [-]:':<{total_chars_mid}}{self.ur_1:>{total_chars_mid}.2f}")
        print(f"{'Radial Source Term Q [-]:':<{total_chars_mid}}{self.Q:>{total_chars_mid}.2f}")
        print(f"{'Circulation Term GAMMA [-]:':<{total_chars_mid}}{self.GAMMA:>{total_chars_mid}.2f}")
        print_banner()
