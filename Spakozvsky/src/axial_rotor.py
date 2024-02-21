from .functions import Brot_n
from Sun.src.general_functions import print_banner_begin, print_banner_end
from Utils.styles import total_chars, total_chars_mid
from numpy import pi as pi

class AxialRotor:
    """
    this class contains the axial rotor blade wor flow model for the Spakovszky model
    """
    def __init__(self, uz, ut_1, ut_2, alpha_1, beta_1, beta_2, lambda_r, dLr_dTanb, tau_r=0):
        """
        provide non-dimensional quantities
        Args:
            uz: mean axial velocity
            ut_1: mean inlet tngential velocity
            ut_2: mean outlet tangential velocity
            alpha_1: mean inlet flow angle
            beta_1: mean inlet relative flow angle
            beta_2: mean outlet relative flow angle
            lambda_r: inertia parameter of the blade row
            dLr_dTanb: loss function
            tau_r: delay constant
        """

        self.uz = uz
        self.ut_1 = ut_1
        self.ut_2 = ut_2
        self.alpha_1 = alpha_1
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lambda_r = lambda_r
        self.dLr_dTanb = dLr_dTanb
        self.tau_r = tau_r
        self.print_info()



    def transfer_function(self, s, n, theta=0):
        """
        compute the component transfer function
        Args:
            theta: tangential cordinate
            s: laplace variable
            n: circumferential harmonic

        Returns:
            M: axial stator row flow transfer function at a given location (theta, s, n)

        """
        M = Brot_n(s, n, self.uz, self.ut_1, self.ut_2, self.alpha_1, self.beta_1, self.beta_2, self.lambda_r,
                   self.dLr_dTanb, theta, self.tau_r)
        return M

    def print_info(self):
        """
        Print the information of the component
        """
        print_banner_begin('AXIAL ROTOR')
        print(f"{'Inlet Axial Velocity [-]:':<{total_chars_mid}}{self.uz:>{total_chars_mid}.2f}")
        print(f"{'Inlet Tang. Velocity [-]:':<{total_chars_mid}}{self.ut_1:>{total_chars_mid}.2f}")
        print(f"{'Outlet Tang. Velocity [-]:':<{total_chars_mid}}{self.ut_2:>{total_chars_mid}.2f}")
        print(f"{'Inlet Abs. Angle [deg]:':<{total_chars_mid}}{self.alpha_1 * 180 / pi:>{total_chars_mid}.2f}")
        print(f"{'Inlet Rel. Angle [deg]:':<{total_chars_mid}}{self.beta_1 * 180 / pi:>{total_chars_mid}.2f}")
        print(f"{'Outlet Rel. Angle [deg]:':<{total_chars_mid}}{self.beta_2 * 180 / pi:>{total_chars_mid}.2f}")
        print(f"{'Inertia Parameter [-]:':<{total_chars_mid}}{self.lambda_r:>{total_chars_mid}.2f}")
        print(f"{'Loss Coefficient [-]:':<{total_chars_mid}}{self.dLr_dTanb:>{total_chars_mid}.2f}")
        print(f"{'Time Lag Parameter [-]:':<{total_chars_mid}}{self.tau_r:>{total_chars_mid}.2f}")
        print_banner_end()
