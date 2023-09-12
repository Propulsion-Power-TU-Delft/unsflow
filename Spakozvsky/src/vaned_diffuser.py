from .functions import Bdif_n


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




    def transfer_function(self, theta, s, n):
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
