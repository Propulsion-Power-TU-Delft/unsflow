"""
unsflow.senoo.senoo_diffuser
============================
Core solver for the Senoo & Kinoshita vaneless diffuser rotating stall
and surge prediction model.

Includes:
1. Differential 3D boundary layer marching solver (Senoo 1977 Paper 1 & 2)
2. 1D compressible swirling flow with friction solver (Word doc & Senoo 1978)
3. Density-corrected compressibility mapping for rotating stall and surge
4. Fast digitized Senoo solution map mode
5. Radial-convention reporting for operating, stall, and surge angles
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

from unsflow.utils.formatting import print_banner, total_chars, total_chars_mid
from .config import SenooConfig
from .profiles import (
    compute_beta_table,
    evaluate_layer_integrals,
    evaluate_core_integrals,
    wall_shear_stresses,
    width_at_radius,
)
from .map import senoo_solution_map
from .results import SenooResults, SenooFlowField


class SenooDiffuser:
    """
    Implements the Senoo vaneless diffuser model for turbomachinery instability,
    reverse flow, rotating stall, and surge inception analysis.

    References:
    - Senoo, Y., Kinoshita, Y., and Ishida, M., "Asymmetric Flow in Vaneless
      Diffusers of Centrifugal Blowers," ASME J. Fluids Eng., 99(1), 104-111, 1977.
    - Senoo, Y., and Kinoshita, Y., "Influence of Inlet Flow Conditions and
      Geometries of Centrifugal Vaneless Diffusers on Critical Flow Angle for
      Reverse Flow," ASME J. Fluids Eng., 99(1), 98-103, 1977.
    - Senoo, Y., and Kinoshita, Y., "Limits of Rotating Stall and Stall in Vaneless
      Diffuser of Centrifugal Compressors," ASME Paper 78-GT-19, 1978.
    """

    def __init__(self, config: Optional[SenooConfig] = None, **kwargs):
        """
        Initialize the Senoo diffuser solver with a SenooConfig or keyword arguments.
        """
        if config is not None:
            self.config = config
        else:
            self.config = SenooConfig(**kwargs)

        n_p = self.config.n_power if self.config.n_power is not None else 7.0
        self.beta_tab = compute_beta_table(n_p, self.config.m)

    def print_info(self):
        """Print configuration summary."""
        self.config.print_info()

    # -------------------------------------------------------------------------
    # 1. 1D Compressible Flow-With-Friction Solver
    # -------------------------------------------------------------------------
    def solve_1d_compressible(
        self,
        r_eval: Optional[np.ndarray] = None,
        alpha_inlet: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Solves 1D compressible swirling flow with wall friction along the diffuser:
            d(r C_theta)/dr = - (c_f * C * C_theta * r) / (b * C_r)
            dp/dr = rho * C_theta^2 / r - rho * C_r * dC_r/dr - (c_f * rho * C * C_r) / b
            Continuity: mdot = 2*pi*r*b(r)*rho(r)*C_r(r)
            Energy: T_t = const
        """
        cfg = self.config
        r1, r2 = cfg.r1, cfg.r2
        b1, b2 = cfg.b1, cfg.b2
        gamma = cfg.gamma
        R_gas = cfg.R_gas
        cp = cfg.cp
        cf = cfg.cf_1d

        # Determine inlet velocity components
        if alpha_inlet is not None:
            # If passed, alpha_inlet in radial convention
            a_rad = np.radians(alpha_inlet)
            V1 = cfg.V1_mean
            Cr1 = V1 * np.cos(a_rad)
            Ct1 = V1 * np.sin(a_rad)
        else:
            Cr1 = cfg.Cr1
            Ct1 = cfg.Ct1
            V1 = cfg.V1_mean

        p1 = cfg.p1
        T1 = cfg.T1
        rho1 = cfg.rho
        Tt1 = cfg.T0

        if r_eval is None:
            r_eval = np.linspace(r1, r2, cfg.nr)

        def odefun(r, y):
            Cr, Ct, p = y
            Cr = max(Cr, 1e-6)
            C2 = Cr**2 + Ct**2
            C = np.sqrt(C2)
            T = max(Tt1 - 0.5 * C2 / cp, 50.0)
            rho = max(p / (R_gas * T), 0.05)
            a2 = gamma * R_gas * T
            Mr2 = Cr**2 / a2

            b, db_dr = cfg.get_width(r)

            # Tangential momentum with friction on 2 walls
            dCt_dr = - Ct / r - (cf * C * Ct) / (b * Cr)

            # Continuity & Energy coupling for dCr/dr
            term_geo = - 1.0 / r - db_dr / b
            term_centrif = - (gamma / a2) * (Ct**2 / r - (cf * C * Cr) / b)
            term_tang = - ((gamma - 1.0) / a2) * (Ct * dCt_dr)

            denom = max(1.0 - Mr2, 0.05)
            dCr_dr = Cr * (term_geo + term_centrif + term_tang) / denom
            dp_dr = rho * Ct**2 / r - rho * Cr * dCr_dr - (cf * rho * C * Cr) / b

            return [dCr_dr, dCt_dr, dp_dr]

        sol = solve_ivp(
            odefun,
            [r1, r2],
            [Cr1, Ct1, p1],
            t_eval=r_eval,
            method='RK45',
            rtol=1e-5,
            atol=1e-6
        )

        r_out = sol.t
        Cr_out = np.maximum(sol.y[0], 1e-6)
        Ct_out = np.maximum(sol.y[1], 1e-6)
        p_out = sol.y[2]

        C_tot = np.sqrt(Cr_out**2 + Ct_out**2)
        T_out = np.maximum(Tt1 - 0.5 * C_tot**2 / cp, 50.0)
        rho_out = p_out / (R_gas * T_out)
        a_sound = np.sqrt(gamma * R_gas * T_out)
        mach_out = C_tot / a_sound

        beta_out = np.degrees(np.arctan2(Cr_out, Ct_out))
        alpha_out = 90.0 - beta_out

        # Incompressible reference at same radii
        b_arr = np.array([cfg.get_width(rv)[0] for rv in r_out])
        Cr_incomp = (r1 * b1 / (r_out * b_arr)) * Cr1
        # Tangential velocity with same wall friction or angular momentum
        Ct_incomp = Ct_out
        beta_incomp = np.degrees(np.arctan2(Cr_incomp, Ct_incomp))
        alpha_incomp = 90.0 - beta_incomp

        delta_beta = beta_incomp - beta_out
        density_ratio = rho_out / rho1

        return {
            'r': r_out,
            'Cr': Cr_out,
            'Ct': Ct_out,
            'p': p_out,
            'T': T_out,
            'rho': rho_out,
            'mach': mach_out,
            'beta': beta_out,
            'alpha': alpha_out,
            'beta_incomp': beta_incomp,
            'alpha_incomp': alpha_incomp,
            'delta_beta': delta_beta,
            'density_ratio': density_ratio,
        }

    # -------------------------------------------------------------------------
    # 2. 3D Differential Boundary Layer Marching Solver (Senoo 1977)
    # -------------------------------------------------------------------------
    def _calc_entry_integrals(
        self,
        d1: float,
        ee1: float,
        d2: float,
        ee2: float,
        ur1: float,
        ut1: float,
        ur2: float,
        ut2: float,
        b: float,
        c_comp: float = 0.0,
    ) -> np.ndarray:
        """Helper to evaluate all 12 entry zone profile integrals."""
        hc = max(b - d1 - d2, 1e-7)
        Gr = (ur2 - ur1) / hc
        Gt = (ut2 - ut1) / hc

        n_p = self.config.n_power if self.config.n_power is not None else 7.0

        Ir1, Irr1, Itt1, Irt1 = evaluate_layer_integrals(
            d1, ee1, ur1, ut1, Gr, Gt, self.config.m, n_p, 1, self.beta_tab, c_comp
        )
        Ir2, Irr2, Itt2, Irt2 = evaluate_layer_integrals(
            d2, ee2, ur2, ut2, Gr, Gt, self.config.m, n_p, 2, self.beta_tab, c_comp
        )
        Jr, Jrr, Jtt, Jrt = evaluate_core_integrals(hc, ur1, ur2, ut1, ut2)

        return np.array([Ir1, Irr1, Itt1, Irt1, Ir2, Irr2, Itt2, Irt2, Jr, Jrr, Jtt, Jrt])

    def _entry_rhs(
        self,
        r: float,
        y_vec: np.ndarray,
        wall_choice: int,
    ) -> np.ndarray:
        """
        RHS derivative vector for the Entry Zone (9 coupled ODEs):
        y = [delta1, e1, delta2, e2, Ur1, Ut1, Ur2, Ut2, p]
        """
        cfg = self.config
        b, db_dr = cfg.get_width(r)

        d1, ee1, d2, ee2, ur1, ut1, ur2, ut2, p_val = y_vec
        hc = b - d1 - d2
        if d1 <= 1e-6 or d2 <= 1e-6 or hc <= 1e-6:
            return np.zeros(9)

        c_comp = 0.0
        rho_curr = cfg.rho
        mach_curr = 0.0
        if cfg.compressible:
            U_avg = np.sqrt(0.5 * (ur1**2 + ur2**2 + ut1**2 + ut2**2))
            cp = cfg.cp
            T_curr = max(cfg.T0 - 0.5 * U_avg**2 / cp, 100.0)
            rho_curr = max(p_val / (cfg.R_gas * T_curr), 0.1)
            mach_curr = U_avg / np.sqrt(cfg.gamma * cfg.R_gas * T_curr)
            c_comp = cfg.zeta * 0.5 * U_avg**2 / (cp * T_curr)

        y8 = y_vec[:8]
        ints = self._calc_entry_integrals(d1, ee1, d2, ee2, ur1, ut1, ur2, ut2, b, c_comp)
        Ir1, Irr1, Itt1, Irt1, Ir2, Irr2, Itt2, Irt2, Jr, Jrr, Jtt, Jrt = ints

        # Numerical Jacobian of integrals w.r.t y8
        J_ints = np.zeros((12, 8))
        for j in range(8):
            h = 1e-8 * max(abs(y8[j]), 1e-4)
            yp = y8.copy()
            yp[j] += h
            ym = y8.copy()
            ym[j] -= h
            J_ints[:, j] = (
                self._calc_entry_integrals(
                    yp[0], yp[1], yp[2], yp[3], yp[4], yp[5], yp[6], yp[7], b, c_comp
                )
                - self._calc_entry_integrals(
                    ym[0], ym[1], ym[2], ym[3], ym[4], ym[5], ym[6], ym[7], b, c_comp
                )
            ) / (2.0 * h)

        tau0r_1, tau0t_1 = wall_shear_stresses(
            ur1, ut1, d1, ee1, rho_curr, cfg.nu, cfg.Kf, mach_curr, cfg.compressible
        )
        tau0r_2, tau0t_2 = wall_shear_stresses(
            ur2, ut2, d2, ee2, rho_curr, cfg.nu, cfg.Kf, mach_curr, cfg.compressible
        )

        Gr = (ur2 - ur1) / hc
        Gt = (ut2 - ut1) / hc

        A = np.zeros((9, 9))
        b_vec = np.zeros(9)

        # 0: BL1 radial momentum (Eq. 4)
        A[0, :8] = J_ints[1, :] - ur1 * J_ints[0, :]
        A[0, 8] = d1 / rho_curr
        b_vec[0] = - (Irr1 - Itt1 - ur1 * Ir1) / r - tau0r_1 / rho_curr

        # 1: BL1 tangential momentum (Eq. 5)
        A[1, :8] = J_ints[3, :] - ut1 * J_ints[0, :]
        b_vec[1] = - (2.0 * Irt1 - ut1 * Ir1) / r - tau0t_1 / rho_curr

        # 2: BL2 radial momentum (Eq. 4)
        A[2, :8] = J_ints[5, :] - ur2 * J_ints[4, :]
        A[2, 8] = d2 / rho_curr
        b_vec[2] = - (Irr2 - Itt2 - ur2 * Ir2) / r - tau0r_2 / rho_curr

        # 3: BL2 tangential momentum (Eq. 5)
        A[3, :8] = J_ints[7, :] - ut2 * J_ints[4, :]
        b_vec[3] = - (2.0 * Irt2 - ut2 * Ir2) / r - tau0t_2 / rho_curr

        # 4: Core radial momentum (Eq. 6)
        A[4, :8] = J_ints[9, :] + ur1 * J_ints[0, :] + ur2 * J_ints[4, :]
        A[4, 8] = hc / rho_curr
        b_vec[4] = - (Jrr - Jtt + ur1 * Ir1 + ur2 * Ir2) / r

        # 5: Core tangential momentum (Eq. 7)
        A[5, :8] = J_ints[11, :] + ut1 * J_ints[0, :] + ut2 * J_ints[4, :]
        b_vec[5] = - (2.0 * Jrt + ut1 * Ir1 + ut2 * Ir2) / r

        # 6: Continuity equation (Eq. 8)
        A[6, :8] = J_ints[0, :] + J_ints[8, :] + J_ints[4, :]
        b_vec[6] = - (Ir1 + Jr + Ir2) / r

        # Streamline matching conditions (Eqs. 23 & 24)
        if wall_choice == 1:
            # Wall 1 angular momentum
            A[7, 5] = 1.0
            A[7, :8] -= (Gt / ur1) * J_ints[0, :]
            b_vec[7] = - ut1 / r + (Gt * Ir1) / (r * ur1)

            # Wall 1 Bernoulli
            A[8, 8] = 1.0 / rho_curr
            A[8, 4] = ur1
            A[8, 5] = ut1
            term_G = (ur1 * Gr + ut1 * Gt) / ur1
            A[8, :8] -= term_G * J_ints[0, :]
            b_vec[8] = term_G * Ir1 / r
        else:
            # Wall 2 angular momentum
            A[7, 7] = 1.0
            A[7, :8] += (Gt / ur2) * J_ints[4, :]
            b_vec[7] = - ut2 / r - (Gt * Ir2) / (r * ur2)

            # Wall 2 Bernoulli
            A[8, 8] = 1.0 / rho_curr
            A[8, 6] = ur2
            A[8, 7] = ut2
            term_G = (ur2 * Gr + ut2 * Gt) / ur2
            A[8, :8] += term_G * J_ints[4, :]
            b_vec[8] = - term_G * Ir2 / r

        try:
            ydot = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            ydot = np.linalg.lstsq(A, b_vec, rcond=None)[0]

        return ydot

    def _calc_merged_integrals(
        self,
        d1: float,
        ee1: float,
        ee2: float,
        ur: float,
        ut: float,
        b: float,
        c_comp: float = 0.0,
    ) -> np.ndarray:
        """Helper to evaluate layer integrals in merged downstream zone."""
        d2 = max(b - d1, 1e-7)
        n_p = self.config.n_power if self.config.n_power is not None else 7.0

        Ir1, Irr1, Itt1, Irt1 = evaluate_layer_integrals(
            d1, ee1, ur, ut, 0.0, 0.0, self.config.m, n_p, 1, self.beta_tab, c_comp
        )
        Ir2, Irr2, Itt2, Irt2 = evaluate_layer_integrals(
            d2, ee2, ur, ut, 0.0, 0.0, self.config.m, n_p, 2, self.beta_tab, c_comp
        )
        return np.array([Ir1, Irr1, Itt1, Irt1, Ir2, Irr2, Itt2, Irt2])

    def _merged_rhs(
        self,
        r: float,
        y_vec: np.ndarray,
    ) -> np.ndarray:
        """
        RHS derivative vector for the Merged Downstream Zone (6 coupled ODEs):
        y = [delta1, e1, e2, Ur, Ut, p]
        """
        cfg = self.config
        b, db_dr = cfg.get_width(r)

        d1, ee1, ee2, ur, ut, p_val = y_vec
        d2 = b - d1
        if d1 <= 1e-6 or d2 <= 1e-6:
            return np.zeros(6)

        c_comp = 0.0
        rho_curr = cfg.rho
        mach_curr = 0.0
        if cfg.compressible:
            U_tot = np.sqrt(ur**2 + ut**2)
            cp = cfg.cp
            T_curr = max(cfg.T0 - 0.5 * U_tot**2 / cp, 100.0)
            rho_curr = max(p_val / (cfg.R_gas * T_curr), 0.1)
            mach_curr = U_tot / np.sqrt(cfg.gamma * cfg.R_gas * T_curr)
            c_comp = cfg.zeta * 0.5 * U_tot**2 / (cp * T_curr)

        y5 = y_vec[:5]
        ints = self._calc_merged_integrals(d1, ee1, ee2, ur, ut, b, c_comp)
        Ir1, Irr1, Itt1, Irt1, Ir2, Irr2, Itt2, Irt2 = ints

        J_ints = np.zeros((8, 5))
        for j in range(5):
            h = 1e-8 * max(abs(y5[j]), 1e-4)
            yp = y5.copy()
            yp[j] += h
            ym = y5.copy()
            ym[j] -= h
            J_ints[:, j] = (
                self._calc_merged_integrals(
                    yp[0], yp[1], yp[2], yp[3], yp[4], b, c_comp
                )
                - self._calc_merged_integrals(
                    ym[0], ym[1], ym[2], ym[3], ym[4], b, c_comp
                )
            ) / (2.0 * h)

        tau0r_1, tau0t_1 = wall_shear_stresses(
            ur, ut, d1, ee1, rho_curr, cfg.nu, cfg.Kf, mach_curr, cfg.compressible
        )
        tau0r_2, tau0t_2 = wall_shear_stresses(
            ur, ut, d2, ee2, rho_curr, cfg.nu, cfg.Kf, mach_curr, cfg.compressible
        )

        row_r1 = J_ints[1, :] - ur * J_ints[0, :]
        rhs_r1 = - (Irr1 - Itt1 - ur * Ir1) / r - tau0r_1 / rho_curr

        row_t1 = J_ints[3, :] - ut * J_ints[0, :]
        rhs_t1 = - (2.0 * Irt1 - ut * Ir1) / r - tau0t_1 / rho_curr

        row_r2 = J_ints[5, :] - ur * J_ints[4, :]
        rhs_r2 = - (Irr2 - Itt2 - ur * Ir2) / r - tau0r_2 / rho_curr
        if abs(db_dr) > 1e-12:
            rhs_r2 += (J_ints[5, 0] - ur * J_ints[4, 0]) * db_dr

        row_t2 = J_ints[7, :] - ut * J_ints[4, :]
        rhs_t2 = - (2.0 * Irt2 - ut * Ir2) / r - tau0t_2 / rho_curr
        if abs(db_dr) > 1e-12:
            rhs_t2 += (J_ints[7, 0] - ut * J_ints[4, 0]) * db_dr

        A = np.zeros((6, 6))
        b_vec = np.zeros(6)

        # 0: Total radial momentum = eq_r1 + eq_r2
        A[0, :5] = row_r1 + row_r2
        A[0, 5] = (d1 + d2) / rho_curr
        b_vec[0] = rhs_r1 + rhs_r2

        # 1: Radial asymmetry = eq_r1 - eq_r2 (governs delta1 position)
        A[1, :5] = row_r1 - row_r2
        A[1, 5] = (d1 - d2) / rho_curr
        b_vec[1] = rhs_r1 - rhs_r2
        if abs(ee1 - ee2) < 1e-5 and abs(d1 - d2) < 1e-5:
            A[1, :] = 0.0
            A[1, 0] = 1.0
            b_vec[1] = 0.5 * db_dr

        # 2: Total tangential momentum = eq_t1 + eq_t2
        A[2, :5] = row_t1 + row_t2
        b_vec[2] = rhs_t1 + rhs_t2

        # 3: Tangential asymmetry = eq_t1 - eq_t2 (governs e1 - e2)
        A[3, :5] = row_t1 - row_t2
        b_vec[3] = rhs_t1 - rhs_t2
        if abs(ee1 - ee2) < 1e-5 and abs(d1 - d2) < 1e-5:
            A[3, :] = 0.0
            A[3, 1] = 1.0
            A[3, 2] = -1.0
            b_vec[3] = 0.0

        # 4: Continuity across passage b
        A[4, :5] = J_ints[0, :] + J_ints[4, :]
        b_vec[4] = - (Ir1 + Ir2) / r
        if abs(db_dr) > 1e-12:
            b_vec[4] += J_ints[4, 0] * db_dr

        # 5: Maximum velocity streamline condition (Johnston polar hodograph closure)
        tan_beta_val = (
            cfg.tan_beta
            if (cfg.tan_beta is not None and cfg.tan_beta > 0.0)
            else (3.6 * (ur / ut) if ut > 1e-6 else 1.0)
        )
        A[5, 3] = ur
        A[5, 4] = - tan_beta_val * ur
        A[5, 5] = 1.0 / rho_curr
        b_vec[5] = ut**2 / r + tan_beta_val * ur * ut / r

        try:
            ydot = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            ydot = np.linalg.lstsq(A, b_vec, rcond=None)[0]

        return ydot

    def solve_ode(
        self,
        nr: Optional[int] = None,
        nspan: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Solves the full 3D boundary layer marching ODEs from r1 to r2.
        Returns dictionary containing raw step arrays, 1D profiles, and reconstructed 2D field.
        """
        cfg = self.config
        nr = nr if nr is not None else cfg.nr
        nspan = nspan if nspan is not None else cfg.nspan

        # Compute initial core velocities from operating flow angle
        # In Senoo boundary layer formulation:
        # Ur_mean and Ut_mean represent radial and tangential velocities
        Ur_mean = cfg.Cr1
        Ut_mean = cfg.Ct1

        ur1_0 = Ur_mean * (1.0 - 0.5 * cfg.distortion_ur)
        ur2_0 = Ur_mean * (1.0 + 0.5 * cfg.distortion_ur)
        ut1_0 = Ut_mean * (1.0 - 0.5 * cfg.distortion_utheta)
        ut2_0 = Ut_mean * (1.0 + 0.5 * cfg.distortion_utheta)

        d1_0 = cfg.delta1_hub
        d2_0 = cfg.delta1_shroud
        e1_0 = cfg.e1_hub
        e2_0 = cfg.e1_shroud
        p_0 = cfg.p1

        y_curr = np.array([d1_0, e1_0, d2_0, e2_0, ur1_0, ut1_0, ur2_0, ut2_0, p_0], dtype=float)
        r_curr = cfg.r1
        dr = cfg.dr_max

        r_steps: List[float] = [r_curr]
        y_steps: List[np.ndarray] = [y_curr.copy()]
        zones: List[int] = [0]

        r_merge: Optional[float] = None
        merged_mode = False
        step = 0

        # RK4 Integration Loop
        while r_curr < cfg.r2 - 1e-9:
            b_local, _ = cfg.get_width(r_curr)
            step_dr = min(dr, cfg.r2 - r_curr)

            if not merged_mode:
                if (y_curr[0] + y_curr[2]) >= 0.98 * b_local:
                    merged_mode = True
                    r_merge = r_curr
                    d1_m = min(max(y_curr[0], 0.1 * b_local), 0.9 * b_local)
                    e1_m = y_curr[1]
                    e2_m = y_curr[3]
                    ur_m = 0.5 * (y_curr[4] + y_curr[6])
                    ut_m = 0.5 * (y_curr[5] + y_curr[7])
                    p_m = y_curr[8]
                    y_curr = np.array([d1_m, e1_m, e2_m, ur_m, ut_m, p_m], dtype=float)

            if not merged_mode:
                w_choice = 1 if (step % 2 == 0) else 2
                k1 = self._entry_rhs(r_curr, y_curr, w_choice)
                k2 = self._entry_rhs(r_curr + 0.5 * step_dr, y_curr + 0.5 * step_dr * k1, w_choice)
                k3 = self._entry_rhs(r_curr + 0.5 * step_dr, y_curr + 0.5 * step_dr * k2, w_choice)
                k4 = self._entry_rhs(r_curr + step_dr, y_curr + step_dr * k3, w_choice)

                y_curr = y_curr + (step_dr / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                y_curr[0] = max(y_curr[0], 1e-5)
                y_curr[2] = max(y_curr[2], 1e-5)
            else:
                k1 = self._merged_rhs(r_curr, y_curr)
                k2 = self._merged_rhs(r_curr + 0.5 * step_dr, y_curr + 0.5 * step_dr * k1)
                k3 = self._merged_rhs(r_curr + 0.5 * step_dr, y_curr + 0.5 * step_dr * k2)
                k4 = self._merged_rhs(r_curr + step_dr, y_curr + step_dr * k3)

                y_curr = y_curr + (step_dr / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                y_curr[0] = min(max(y_curr[0], 0.05 * b_local), 0.95 * b_local)

            r_curr += step_dr
            step += 1
            r_steps.append(r_curr)
            y_steps.append(y_curr.copy())
            zones.append(1 if merged_mode else 0)

        # Regular output grids
        r_grid = np.linspace(cfg.r1, cfg.r2, nr)
        span_grid = np.linspace(0.0, 1.0, nspan)
        b_grid = np.array([cfg.get_width(rv)[0] for rv in r_grid])

        R_2d, SPAN_2d = np.meshgrid(r_grid, span_grid, indexing='ij')
        Y_2d = SPAN_2d * b_grid[:, np.newaxis]

        # Unpack raw steps
        r_arr = np.array(r_steps)
        raw_d1 = np.array([y[0] for y in y_steps])
        raw_e1 = np.array([y[1] for y in y_steps])
        raw_p = np.array([y[-1] for y in y_steps])
        raw_d2 = np.zeros(len(y_steps))
        raw_e2 = np.zeros(len(y_steps))
        raw_ur1 = np.zeros(len(y_steps))
        raw_ut1 = np.zeros(len(y_steps))
        raw_ur2 = np.zeros(len(y_steps))
        raw_ut2 = np.zeros(len(y_steps))

        for k, y in enumerate(y_steps):
            b_loc, _ = cfg.get_width(r_steps[k])
            if zones[k] == 0:
                raw_d2[k] = y[2]
                raw_e2[k] = y[3]
                raw_ur1[k] = y[4]
                raw_ut1[k] = y[5]
                raw_ur2[k] = y[6]
                raw_ut2[k] = y[7]
            else:
                raw_d2[k] = b_loc - y[0]
                raw_e2[k] = y[2]
                raw_ur1[k] = y[3]
                raw_ut1[k] = y[4]
                raw_ur2[k] = y[3]
                raw_ut2[k] = y[4]

        # Interpolate 1D profiles
        delta_hub_1d = interp1d(r_arr, raw_d1, fill_value='extrapolate')(r_grid)
        delta_shroud_1d = interp1d(r_arr, raw_d2, fill_value='extrapolate')(r_grid)
        e_hub_1d = interp1d(r_arr, raw_e1, fill_value='extrapolate')(r_grid)
        e_shroud_1d = interp1d(r_arr, raw_e2, fill_value='extrapolate')(r_grid)
        Ur_hub_1d = interp1d(r_arr, raw_ur1, fill_value='extrapolate')(r_grid)
        Ut_hub_1d = interp1d(r_arr, raw_ut1, fill_value='extrapolate')(r_grid)
        Ur_shroud_1d = interp1d(r_arr, raw_ur2, fill_value='extrapolate')(r_grid)
        Ut_shroud_1d = interp1d(r_arr, raw_ut2, fill_value='extrapolate')(r_grid)
        p_1d = interp1d(r_arr, raw_p, fill_value='extrapolate')(r_grid)
        zone_1d = np.array([1 if (r_merge is not None and rv >= r_merge) else 0 for rv in r_grid])

        # 2D velocity and flow field reconstruction
        Vr_2d = np.zeros((nr, nspan))
        Vt_2d = np.zeros((nr, nspan))
        Rho_2d = np.zeros((nr, nspan))
        Mach_2d = np.zeros((nr, nspan))

        n_p = cfg.n_power if cfg.n_power is not None else 7.0
        m_p = cfg.m

        has_rev_hub = np.zeros(nr, dtype=bool)
        has_rev_shroud = np.zeros(nr, dtype=bool)

        for i in range(nr):
            b_loc = b_grid[i]
            d1 = min(max(delta_hub_1d[i], 1e-6), 0.99 * b_loc)
            d2 = min(max(delta_shroud_1d[i], 1e-6), 0.99 * b_loc)
            e1 = e_hub_1d[i]
            e2 = e_shroud_1d[i]
            ur1 = Ur_hub_1d[i]
            ut1 = Ut_hub_1d[i]
            ur2 = Ur_shroud_1d[i]
            ut2 = Ut_shroud_1d[i]
            p_loc = p_1d[i]

            hc = max(b_loc - d1 - d2, 0.0)
            Gr = (ur2 - ur1) / hc if hc > 1e-6 else 0.0
            Gt = (ut2 - ut1) / hc if hc > 1e-6 else 0.0

            # Wall reverse flow criterion (Senoo 1977 Paper 2 Eq. 14):
            # Reverse flow occurs if 1 - e * cot(alpha_core) <= 0
            cot_a1 = ut1 / ur1 if abs(ur1) > 1e-9 else 0.0
            cot_a2 = ut2 / ur2 if abs(ur2) > 1e-9 else 0.0
            if (1.0 - e1 * cot_a1) <= 0.0:
                has_rev_hub[i] = True
            if (1.0 - e2 * cot_a2) <= 0.0:
                has_rev_shroud[i] = True

            U_ref = np.sqrt(0.5 * (ur1**2 + ur2**2 + ut1**2 + ut2**2))
            cp = cfg.cp
            T_loc = (
                max(cfg.T0 - 0.5 * U_ref**2 / cp, 100.0)
                if cfg.compressible
                else (cfg.p1 / (cfg.rho * cfg.R_gas))
            )
            rho_star = p_loc / (cfg.R_gas * T_loc) if cfg.compressible else cfg.rho

            for j in range(nspan):
                y = Y_2d[i, j]
                c_fac = 0.0

                if y <= d1:
                    eta = max(min(y / d1, 1.0), 1e-6)
                    eta_pow = eta ** (1.0 / n_p)
                    om_eta_m = (1.0 - eta) ** m_p
                    term_r1 = ur1 - e1 * ut1 * om_eta_m
                    term_r2 = 1.0 - Gr * (d1 / ur1) * (1.0 - eta) if abs(ur1) > 1e-9 else 1.0
                    term_t1 = ut1 + e1 * ur1 * om_eta_m
                    term_t2 = 1.0 - Gt * (d1 / ut1) * (1.0 - eta) if abs(ut1) > 1e-9 else 1.0
                    vr = eta_pow * term_r1 * term_r2
                    vt = eta_pow * term_t1 * term_t2
                    c_fac = 1.0 - eta**2
                elif y >= b_loc - d2:
                    eta = max(min((b_loc - y) / d2, 1.0), 1e-6)
                    eta_pow = eta ** (1.0 / n_p)
                    om_eta_m = (1.0 - eta) ** m_p
                    term_r1 = ur2 - e2 * ut2 * om_eta_m
                    term_r2 = 1.0 + Gr * (d2 / ur2) * (1.0 - eta) if abs(ur2) > 1e-9 else 1.0
                    term_t1 = ut2 + e2 * ur2 * om_eta_m
                    term_t2 = 1.0 + Gt * (d2 / ut2) * (1.0 - eta) if abs(ut2) > 1e-9 else 1.0
                    vr = eta_pow * term_r1 * term_r2
                    vt = eta_pow * term_t1 * term_t2
                    c_fac = 1.0 - eta**2
                else:
                    vr = ur1 + Gr * (y - d1)
                    vt = ut1 + Gt * (y - d1)
                    c_fac = 0.0

                Vr_2d[i, j] = vr
                Vt_2d[i, j] = vt

                if cfg.compressible:
                    c_val = cfg.zeta * 0.5 * (vr**2 + vt**2) / (cp * T_loc)
                    Rho_2d[i, j] = max(rho_star * (1.0 - c_val * c_fac), 0.1)
                    a_sound = np.sqrt(cfg.gamma * cfg.R_gas * T_loc)
                    Mach_2d[i, j] = np.sqrt(vr**2 + vt**2) / a_sound
                else:
                    Rho_2d[i, j] = cfg.rho

        # Direct flow reversal detection on velocity field
        for i in range(nr):
            if np.any(Vr_2d[i, : nspan // 4] < -1e-5):
                has_rev_hub[i] = True
            if np.any(Vr_2d[i, 3 * nspan // 4 :] < -1e-5):
                has_rev_shroud[i] = True

        V_2d = np.sqrt(Vr_2d**2 + Vt_2d**2)
        # alpha measured from radial direction: arctan2(vt, vr)
        Alpha_2d = np.degrees(np.arctan2(Vt_2d, Vr_2d))

        q_dyn_inlet = 0.5 * cfg.rho * cfg.V1_mean**2
        Cp_1d = (p_1d - cfg.p1) / q_dyn_inlet if q_dyn_inlet > 0 else np.zeros_like(p_1d)

        has_reverse = bool(np.any(has_rev_hub) or np.any(has_rev_shroud))
        r_stall_hub = float(r_grid[np.where(has_rev_hub)[0][0]]) if np.any(has_rev_hub) else None
        r_stall_shroud = float(r_grid[np.where(has_rev_shroud)[0][0]]) if np.any(has_rev_shroud) else None

        reversal_radius = None
        if r_stall_hub is not None and r_stall_shroud is not None:
            reversal_radius = min(r_stall_hub, r_stall_shroud)
        elif r_stall_hub is not None:
            reversal_radius = r_stall_hub
        elif r_stall_shroud is not None:
            reversal_radius = r_stall_shroud

        # Check if reverse flow reaches diffuser exit (surge criterion)
        reversal_at_exit = bool(has_rev_hub[-1] or has_rev_shroud[-1])

        return {
            'r': r_grid,
            'span': span_grid,
            'b_r': b_grid,
            'R': R_2d,
            'SPAN': SPAN_2d,
            'Y': Y_2d,
            'vr': Vr_2d,
            'vt': Vt_2d,
            'V': V_2d,
            'alpha_2d': Alpha_2d,
            'p': p_1d,
            'Cp': Cp_1d,
            'rho_2d': Rho_2d,
            'mach_2d': Mach_2d if cfg.compressible else None,
            'delta_hub': delta_hub_1d,
            'delta_shroud': delta_shroud_1d,
            'e_hub': e_hub_1d,
            'e_shroud': e_shroud_1d,
            'Ur_hub': Ur_hub_1d,
            'Ut_hub': Ut_hub_1d,
            'Ur_shroud': Ur_shroud_1d,
            'Ut_shroud': Ut_shroud_1d,
            'has_reverse_flow_hub': has_rev_hub,
            'has_reverse_flow_shroud': has_rev_shroud,
            'zone': zone_1d,
            'has_reverse_flow': has_reverse,
            'r_stall_hub': r_stall_hub,
            'r_stall_shroud': r_stall_shroud,
            'reversal_radius': reversal_radius,
            'reversal_at_exit': reversal_at_exit,
            'r_merge': r_merge,
        }

    # -------------------------------------------------------------------------
    # 3. Critical Flow Angle Bisection Search
    # -------------------------------------------------------------------------
    def find_critical_flow_angle(
        self,
        tol: float = 0.05,
        max_iter: int = 25,
        convention: Optional[str] = None,
    ) -> float:
        """
        Determines the critical inlet flow angle for incipient wall reverse flow.

        Parameters
        ----------
        tol : float
            Convergence tolerance [deg].
        max_iter : int
            Maximum number of bisection iterations.
        convention : str or None
            'radial' to return alpha_cr (angle from radial direction, typically 75 - 85 deg).
            'tangential' to return beta_cr (angle from tangential direction, typically 7 - 16 deg).
            None (default): auto-detected based on input config. If input was given with
            alpha1_mean <= 45 deg (tangential), returns tangential angle for backwards compatibility;
            otherwise returns radial angle.
        """
        orig_alpha = self.config.alpha1_mean
        orig_beta = self.config.beta1_mean
        orig_cr1 = self.config.Cr1
        orig_ct1 = self.config.Ct1

        # Check if caller used tangential angle convention (< 45 deg)
        is_tangential_input = (orig_alpha <= 45.0)

        if self.config.solver_mode == 'map':
            # Fast solution from digitized Senoo map
            b_ratio = self.config.b1 / self.config.r1
            r_ratio = self.config.r2 / self.config.r1
            alpha_rev, beta_rev, _ = senoo_solution_map(b_ratio, r_ratio)
            if convention == 'tangential' or (convention is None and is_tangential_input):
                return beta_rev
            return alpha_rev

        # Differential ODE bisection search
        # Search range in tangential angle beta [deg]:
        # beta_high = unstalled (steep / large beta), beta_low = stalled (flatter / small beta)
        b_high = 35.0
        b_low = 3.0

        # Verify bounds
        self.config.alpha1_mean = 90.0 - b_high
        self.config.beta1_mean = b_high
        self.config.Cr1 = self.config.V1_mean * np.sin(np.radians(b_high))
        self.config.Ct1 = self.config.V1_mean * np.cos(np.radians(b_high))
        sol_high = self.solve_ode(nr=35, nspan=20)
        if sol_high['has_reverse_flow']:
            b_high = 50.0

        self.config.alpha1_mean = 90.0 - b_low
        self.config.beta1_mean = b_low
        self.config.Cr1 = self.config.V1_mean * np.sin(np.radians(b_low))
        self.config.Ct1 = self.config.V1_mean * np.cos(np.radians(b_low))
        sol_low = self.solve_ode(nr=35, nspan=20)
        if not sol_low['has_reverse_flow']:
            b_low = 1.0

        beta_cr = 0.5 * (b_low + b_high)
        for _ in range(max_iter):
            if abs(b_high - b_low) < tol:
                break
            b_mid = 0.5 * (b_low + b_high)
            self.config.alpha1_mean = 90.0 - b_mid
            self.config.beta1_mean = b_mid
            self.config.Cr1 = self.config.V1_mean * np.sin(np.radians(b_mid))
            self.config.Ct1 = self.config.V1_mean * np.cos(np.radians(b_mid))

            sol_mid = self.solve_ode(nr=45, nspan=20)
            if sol_mid['has_reverse_flow']:
                # Flow reversed -> need steeper angle to stabilize
                b_low = b_mid
            else:
                b_high = b_mid
            beta_cr = 0.5 * (b_low + b_high)

        # Restore original config
        self.config.alpha1_mean = orig_alpha
        self.config.beta1_mean = orig_beta
        self.config.Cr1 = orig_cr1
        self.config.Ct1 = orig_ct1

        alpha_cr = 90.0 - beta_cr

        if convention == 'tangential' or (convention is None and is_tangential_input):
            return float(beta_cr)
        return float(alpha_cr)

    def find_exit_reversal_angle(
        self,
        tol: float = 0.05,
        max_iter: int = 25
    ) -> float:
        """
        Determines the critical inlet flow angle where wall reverse flow
        extends all the way to the diffuser exit (r = r2), representing
        the complete diffuser stall / surge boundary.
        Returns beta_exit_rev [deg] (tangential convention).
        """
        orig_alpha = self.config.alpha1_mean
        orig_beta = self.config.beta1_mean
        orig_cr1 = self.config.Cr1
        orig_ct1 = self.config.Ct1

        b_high = 25.0
        b_low = 2.0

        beta_exit = 0.5 * (b_low + b_high)
        for _ in range(max_iter):
            if abs(b_high - b_low) < tol:
                break
            b_mid = 0.5 * (b_low + b_high)
            self.config.alpha1_mean = 90.0 - b_mid
            self.config.beta1_mean = b_mid
            self.config.Cr1 = self.config.V1_mean * np.sin(np.radians(b_mid))
            self.config.Ct1 = self.config.V1_mean * np.cos(np.radians(b_mid))

            sol = self.solve_ode(nr=40, nspan=20)
            if sol['reversal_at_exit']:
                b_low = b_mid
            else:
                b_high = b_mid
            beta_exit = 0.5 * (b_low + b_high)

        self.config.alpha1_mean = orig_alpha
        self.config.beta1_mean = orig_beta
        self.config.Cr1 = orig_cr1
        self.config.Ct1 = orig_ct1

        return float(beta_exit)

    # -------------------------------------------------------------------------
    # 4. Comprehensive Stability Limits & Angle Calculation
    # -------------------------------------------------------------------------
    def compute_stability_limits(self) -> Dict[str, Any]:
        """
        Calculates all key flow and stability angles:
        1. Operating flow angles (alpha and beta)
        2. Incompressible reverse flow angle
        3. Compressibility correction (1D flow with friction)
        4. Corrected rotating stall angle: beta_RS = 0.88 * beta_rev_corr
        5. Surge angle: reverse flow reaching exit (or empirical ratio)
        """
        cfg = self.config

        # 1. Operating Angles
        if cfg.alpha1_mean <= 45.0:
            alpha_op = 90.0 - cfg.alpha1_mean
            beta_op = cfg.alpha1_mean
        else:
            alpha_op = cfg.alpha1_mean
            beta_op = 90.0 - alpha_op

        # 2. Incompressible Reverse Flow Angle
        if cfg.solver_mode == 'map':
            b_ratio = cfg.b1 / cfg.r1
            r_ratio = cfg.r2 / cfg.r1
            alpha_rev_incomp, beta_rev_incomp, r_rev_frac = senoo_solution_map(b_ratio, r_ratio)
            r_rev = cfg.r1 + r_rev_frac * (cfg.r2 - cfg.r1)
            beta_exit_incomp = 0.85 * beta_rev_incomp
        else:
            beta_rev_incomp = self.find_critical_flow_angle(convention='tangential')
            alpha_rev_incomp = 90.0 - beta_rev_incomp
            # Solve once at critical angle to determine reversal location
            orig_a = cfg.alpha1_mean
            orig_b = cfg.beta1_mean
            cfg.alpha1_mean = alpha_rev_incomp
            cfg.beta1_mean = beta_rev_incomp
            cfg.Cr1 = cfg.V1_mean * np.sin(np.radians(beta_rev_incomp))
            cfg.Ct1 = cfg.V1_mean * np.cos(np.radians(beta_rev_incomp))
            sol_crit = self.solve_ode(nr=40, nspan=20)
            r_rev = sol_crit['reversal_radius'] or (0.5 * (cfg.r1 + cfg.r2))

            if cfg.surge_method == 'diffuser_exit':
                beta_exit_incomp = self.find_exit_reversal_angle()
            else:
                beta_exit_incomp = cfg.surge_ratio * cfg.rs_factor * beta_rev_incomp

            cfg.alpha1_mean = orig_a
            cfg.beta1_mean = orig_b
            cfg.Cr1 = cfg.V1_mean * np.sin(np.radians(orig_b))
            cfg.Ct1 = cfg.V1_mean * np.cos(np.radians(orig_b))

        # 3. Compressibility Corrections via 1D Flow-With-Friction
        delta_beta_c_rs = 0.0
        delta_beta_c_surge = 0.0
        rho_ratio_crit = 1.0
        rho_ratio_exit = 1.0

        if cfg.compressible:
            sol_1d = self.solve_1d_compressible(alpha_inlet=alpha_rev_incomp)
            r_1d = sol_1d['r']
            d_beta_1d = sol_1d['delta_beta']
            d_rho_1d = sol_1d['density_ratio']

            # Interpolate at reversal onset radius r* = r_rev
            idx_crit = int(np.argmin(np.abs(r_1d - r_rev)))
            rho_ratio_crit = float(d_rho_1d[idx_crit])
            d_beta_star_rs = float(d_beta_1d[idx_crit])
            # Senoo's formula: Delta beta_i,c = (rho* / rho_i) * Delta beta*
            delta_beta_c_rs = rho_ratio_crit * max(d_beta_star_rs, 0.0)

            # At diffuser exit r* = r2 for overall diffuser stall / surge
            rho_ratio_exit = float(d_rho_1d[-1])
            d_beta_star_exit = float(d_beta_1d[-1])
            delta_beta_c_surge = rho_ratio_exit * max(d_beta_star_exit, 0.0)

        # 4. Corrected Reverse Flow Angle
        beta_rev_corr = beta_rev_incomp + delta_beta_c_rs
        alpha_rev_corr = 90.0 - beta_rev_corr

        # 5. Rotating Stall Angle (Senoo 0.88 rule on tangential angle)
        # beta_RS = 0.88 * beta_rev_corr => alpha_RS = 90 - beta_RS
        beta_RS = cfg.rs_factor * beta_rev_corr
        alpha_RS = 90.0 - beta_RS

        # 6. Surge Angle
        if cfg.surge_method == 'diffuser_exit':
            beta_exit_corr = beta_exit_incomp + delta_beta_c_surge
            # Surge angle cannot be higher than rotating stall angle
            beta_surge = min(beta_exit_corr, beta_RS * cfg.surge_ratio)
        else:
            beta_surge = cfg.surge_ratio * beta_RS
        alpha_surge = 90.0 - beta_surge

        # 7. Inception Status & Margins
        is_stalled = bool(alpha_op >= alpha_RS)
        is_surging = bool(alpha_op >= alpha_surge)
        has_rev = bool(alpha_op >= alpha_rev_corr)

        stall_margin = alpha_RS - alpha_op
        surge_margin = alpha_surge - alpha_op

        return {
            'alpha': alpha_op,
            'beta': beta_op,
            'alpha_rotating_stall': alpha_RS,
            'beta_rotating_stall': beta_RS,
            'alpha_surge': alpha_surge,
            'beta_surge': beta_surge,
            'alpha_reverse_flow': alpha_rev_corr,
            'beta_reverse_flow': beta_rev_corr,
            'alpha_rev_incomp': alpha_rev_incomp,
            'beta_rev_incomp': beta_rev_incomp,
            'delta_beta_c': delta_beta_c_rs,
            'delta_alpha_c': -delta_beta_c_rs,
            'density_ratio_crit': rho_ratio_crit,
            'density_ratio_exit': rho_ratio_exit,
            'reversal_radius': r_rev,
            'is_stalled': is_stalled,
            'is_surging': is_surging,
            'has_reversed_flow': has_rev,
            'stall_margin': stall_margin,
            'surge_margin': surge_margin,
        }

    # -------------------------------------------------------------------------
    # 5. Full Simulation Solve Entry Point
    # -------------------------------------------------------------------------
    def solve(
        self,
        nr: Optional[int] = None,
        nspan: Optional[int] = None
    ) -> SenooResults:
        """
        Solves the diffuser flow field and calculates the stability limits
        (operating flow angle, rotating stall angle, surge angle, with compressibility corrections).
        Returns a SenooResults object (also aliased as SenooFlowField).
        """
        cfg = self.config
        nr = nr if nr is not None else cfg.nr
        nspan = nspan if nspan is not None else cfg.nspan

        # Compute stability limits first
        limits = self.compute_stability_limits()

        # Solve ODE flow field
        ode_sol = self.solve_ode(nr=nr, nspan=nspan)

        # 1D distributions
        sol_1d = self.solve_1d_compressible(r_eval=ode_sol['r'])
        rho_1d = sol_1d['rho']
        mach_1d = sol_1d['mach']
        alpha_1d = sol_1d['alpha']

        # Determine overall reversal from either ODE or threshold
        has_reversed = bool(ode_sol['has_reverse_flow'] or limits['has_reversed_flow'])
        is_stalled = bool(limits['is_stalled'] or ode_sol['has_reverse_flow'])
        is_surging = bool(limits['is_surging'] or ode_sol['reversal_at_exit'])

        results = SenooResults(
            alpha_op=limits['alpha'],
            beta_op=limits['beta'],
            alpha_rotating_stall=limits['alpha_rotating_stall'],
            beta_rotating_stall=limits['beta_rotating_stall'],
            alpha_surge=limits['alpha_surge'],
            beta_surge=limits['beta_surge'],
            alpha_reverse_flow=limits['alpha_reverse_flow'],
            beta_reverse_flow=limits['beta_reverse_flow'],
            is_stalled=is_stalled,
            is_surging=is_surging,
            has_reversed_flow=has_reversed,
            stall_margin=limits['stall_margin'],
            surge_margin=limits['surge_margin'],
            compressible_mode=cfg.compressible,
            mach_inlet=cfg.mach1,
            density_ratio_crit=limits['density_ratio_crit'],
            density_ratio_exit=limits['density_ratio_exit'],
            delta_beta_c=limits['delta_beta_c'],
            delta_alpha_c=limits['delta_alpha_c'],
            reversal_radius=ode_sol['reversal_radius'] or limits['reversal_radius'],
            r_stall_hub=ode_sol['r_stall_hub'],
            r_stall_shroud=ode_sol['r_stall_shroud'],
            r_merge=ode_sol['r_merge'],
            r=ode_sol['r'],
            b_r=ode_sol['b_r'],
            p=ode_sol['p'],
            Cp=ode_sol['Cp'],
            rho_1d=rho_1d,
            mach_1d=mach_1d,
            alpha_1d=alpha_1d,
            delta_hub=ode_sol['delta_hub'],
            delta_shroud=ode_sol['delta_shroud'],
            e_hub=ode_sol['e_hub'],
            e_shroud=ode_sol['e_shroud'],
            Ur_hub=ode_sol['Ur_hub'],
            Ut_hub=ode_sol['Ut_hub'],
            Ur_shroud=ode_sol['Ur_shroud'],
            Ut_shroud=ode_sol['Ut_shroud'],
            has_reverse_flow_hub=ode_sol['has_reverse_flow_hub'],
            has_reverse_flow_shroud=ode_sol['has_reverse_flow_shroud'],
            zone=ode_sol['zone'],
            span=ode_sol['span'],
            R=ode_sol['R'],
            SPAN=ode_sol['SPAN'],
            Y=ode_sol['Y'],
            vr=ode_sol['vr'],
            vt=ode_sol['vt'],
            V=ode_sol['V'],
            alpha_2d=ode_sol['alpha_2d'],
            rho=ode_sol['rho_2d'],
            mach=ode_sol['mach_2d'],
        )

        return results
