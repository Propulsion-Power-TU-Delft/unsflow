from .functions import Bsta_n


class AxialStator:
    """
    this class contains the axial stator blade row component of Spakovszky model
    """
    def __init__(self, uz, ut_1, ut_2, alpha_1, alpha_2, lambda_s, dLs_dTana, tau_s):
        """
        provide non-dimensional quantities
        Args:
            uz: axial velocity
            ut_1: mean tangential velocity at inlet
            ut_2: mean tangential velocity at outlet
            alpha_1: inlet mean swirl flow angle
            alpha_2: outlet mean swirl flow angle
            lambda_s: row inertia parameter
            dLs_dTana: loss derivative with respect to absolute flow angle
            tau_s: delay constant
        """

        self.uz = uz
        self.ut_1 = ut_1
        self.ut_2 = ut_2
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_s = lambda_s
        self.dLs_dTana = dLs_dTana
        self.tau_s = tau_s



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
        M = Bsta_n(s, n, self.uz, self.ut_1, self.ut_2, self.alpha_1, self.alpha_2, self.lambda_s,
                   self.dLs_dTana, theta, self.tau_s)
        return M
