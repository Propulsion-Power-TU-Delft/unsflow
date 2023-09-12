from .functions import Brot_n


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



    def transfer_function(self, theta, s, n):
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
