from .functions import Bdif_n
from Sun.src.general_functions import print_banner_begin, print_banner_end
from Utils.styles import total_chars, total_chars_mid
from numpy import pi as pi

class VanedDiffuser:
    """
    this class contains the radial impeller component info for the spakovszky model
    """
    def __init__(self, r_1, r_2, rho_1, rho_2, A_1, A_2, ur_1, ut_1, ur_2, ut_2, alpha_1, beta_1, alpha_2,
                 s_dif, dLd_dTana, tau_d):
        """
        take care to provide non-dimensional initialization data. (or provide non-dimensionaliztion inside as a routine)
        Args:
            r_1: mean radius at impeller inlet
            r_2: outlet impeller radius
            rho_1: mean inlet density
            rho_2: mean outlet density
            A_1: inlet area
            A_2: outlet are
            ur_1: mean radial speed at inlet
            ut_1: mean tangential speed at inlet
            ur_2: mean radial speed outlet
            ut_2: mean tangential speed outlet
            alpha_1: mean inlet swirl flow angle
            beta_1: mean inlet swirle relative flow angle
            alpha_2: mean outlet swirl flow angle
            s_dif: mean pathlength along streamline
            dLd_dTana: loss derivative
            tau_d: delay constant
        """
        self.r_1 = r_1
        self.r_2 = r_2
        self. rho_1 = rho_1
        self.rho_2 = rho_2
        self.A_1 = A_1
        self.A_2 = A_2
        self.ur_1 = ur_1
        self.ut_1 = ut_1
        self.ur_2 = ur_2
        self.ut_2 = ut_2
        self.alpha_1 = alpha_1
        self.beta_1 = beta_1
        self.alpha_2 = alpha_2
        self.s_dif = s_dif
        self.dLd_dTana = dLd_dTana
        self.tau_d = tau_d
        self.print_info()




    def transfer_function(self, s, n, theta=0):
        """
        compute the component transfer function
        Args:
            theta: tangential cordinate
            s: laplace variable
            n: circumferential harmonic

        Returns:
            M: vaned diffuser transfer function

        """
        M = Bdif_n(s, n, self.ur_1, self.ur_2, self.ut_1, self.ut_2, self.alpha_1, self.beta_1, self.alpha_2,
                   self.r_1, self.r_2, self.rho_1, self.rho_2, self.A_1, self.A_2, self.s_dif, self.dLd_dTana,
                   theta, self.tau_d)
        return M

    def print_info(self):
        """
        Print the information of the component
        """
        print_banner_begin('VANED DIFFUSER')
        print(f"{'Inlet Radius [-]:':<{total_chars_mid}}{self.r_1:>{total_chars_mid}.2f}")
        print(f"{'Outlet Radius [-]:':<{total_chars_mid}}{self.r_2:>{total_chars_mid}.2f}")
        print(f"{'Inlet Density [-]:':<{total_chars_mid}}{self.rho_1:>{total_chars_mid}.2f}")
        print(f"{'Outlet Density [-]:':<{total_chars_mid}}{self.rho_2:>{total_chars_mid}.2f}")
        print(f"{'Inlet Area [-]:':<{total_chars_mid}}{self.A_1:>{total_chars_mid}.2f}")
        print(f"{'Outlet Area [-]:':<{total_chars_mid}}{self.A_2:>{total_chars_mid}.2f}")
        print(f"{'Inlet Radial Velocity [-]:':<{total_chars_mid}}{self.ur_1:>{total_chars_mid}.2f}")
        print(f"{'Inlet Tang. Velocity [-]:':<{total_chars_mid}}{self.ut_1:>{total_chars_mid}.2f}")
        print(f"{'Outlet Radial Velocity [-]:':<{total_chars_mid}}{self.ur_2:>{total_chars_mid}.2f}")
        print(f"{'Outlet Tang. Velocity [-]:':<{total_chars_mid}}{self.ut_2:>{total_chars_mid}.2f}")
        print(f"{'Inlet Abs. Angle [deg]:':<{total_chars_mid}}{self.alpha_1*180/pi:>{total_chars_mid}.2f}")
        print(f"{'Inlet Rel. Angle [deg]:':<{total_chars_mid}}{self.beta_1 * 180 / pi:>{total_chars_mid}.2f}")
        print(f"{'Outlet Abs. Angle [deg]:':<{total_chars_mid}}{self.alpha_2 * 180 / pi:>{total_chars_mid}.2f}")
        print(f"{'Path Length [-]:':<{total_chars_mid}}{self.s_dif:>{total_chars_mid}.2f}")
        print(f"{'Loss coefficient [-]:':<{total_chars_mid}}{self.dLd_dTana:>{total_chars_mid}.2f}")
        print(f"{'Time Lag Parameter [-]:':<{total_chars_mid}}{self.tau_d:>{total_chars_mid}.2f}")
        print_banner_end()