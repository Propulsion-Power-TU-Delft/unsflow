from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from scipy.interpolate import interp1d
from unsflow.utils.formatting import print_banner, total_chars, total_chars_mid
from .config import SenooConfig
from .profiles import (
    compute_beta_table,
    evaluate_layer_integrals,
    evaluate_core_integrals,
    wall_shear_stresses,
    width_at_radius
)
from .results import SenooFlowField


class SenooDiffuser:
    """
    Implements the Senoo vaneless diffuser model for turbomachinery instability
    and reverse flow analysis.

    References:
    - Senoo, Y., Kinoshita, Y., and Ishida, M., "Asymmetric Flow in Vaneless
      Diffusers of Centrifugal Blowers," ASME J. Fluids Eng., 1977.
    - Senoo, Y., and Kinoshita, Y., "Influence of Inlet Flow Conditions and
      Geometries of Centrifugal Vaneless Diffusers on Critical Flow Angle for
      Reverse Flow," ASME J. Fluids Eng., 1977.
    """

    def __init__(self, config: Optional[SenooConfig] = None, **kwargs):
        """
        Initialize the Senoo diffuser solver with a SenooConfig or keyword arguments.
        """
        if config is not None:
            self.config = config
        else:
            self.config = SenooConfig(**kwargs)

        # Precompute Beta function lookup table for fast analytical profile integration
        n_p = self.config.n_power if self.config.n_power is not None else 7.0
        self.beta_tab = compute_beta_table(n_p, self.config.m)

    def print_info(self):
        """Print component information banner."""
        self.config.print_info()

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
        c_comp: float = 0.0
    ) -> np.ndarray:
        """Helper to evaluate all 12 entry integrals."""
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
        wall_choice: int
    ) -> np.ndarray:
        """
        RHS derivative vector for the Entry Zone (9 ODEs):
        y = [delta1, e1, delta2, e2, Ur1, Ut1, Ur2, Ut2, p]
        """
        cfg = self.config
        b, db_dr = width_at_radius(r, cfg.r1, cfg.r2, cfg.b1, cfg.b2)

        d1, ee1, d2, ee2, ur1, ut1, ur2, ut2, p_val = y_vec
        hc = b - d1 - d2
        if d1 <= 1e-6 or d2 <= 1e-6 or hc <= 1e-6:
            return np.zeros(9)

        # Compressibility parameters
        c_comp = 0.0
        rho_curr = cfg.rho
        mach_curr = 0.0
        if cfg.compressible:
            U_avg = np.sqrt(0.5 * (ur1**2 + ur2**2 + ut1**2 + ut2**2))
            cp = cfg.gamma * cfg.R_gas / (cfg.gamma - 1.0)
            T_curr = max(cfg.T0 - 0.5 * U_avg**2 / cp, 100.0)
            rho_curr = max(p_val / (cfg.R_gas * T_curr), 0.1)
            mach_curr = U_avg / np.sqrt(cfg.gamma * cfg.R_gas * T_curr)
            c_comp = cfg.zeta * 0.5 * U_avg**2 / (cp * T_curr)

        y8 = y_vec[:8]
        ints = self._calc_entry_integrals(d1, ee1, d2, ee2, ur1, ut1, ur2, ut2, b, c_comp)
        Ir1, Irr1, Itt1, Irt1, Ir2, Irr2, Itt2, Irt2, Jr, Jrr, Jtt, Jrt = ints

        # Jacobian of integrals w.r.t y8
        J_ints = np.zeros((12, 8))
        for j in range(8):
            h = 1e-8 * max(abs(y8[j]), 1e-4)
            yp = y8.copy(); yp[j] += h
            ym = y8.copy(); ym[j] -= h
            J_ints[:, j] = (
                self._calc_entry_integrals(yp[0], yp[1], yp[2], yp[3], yp[4], yp[5], yp[6], yp[7], b, c_comp)
                - self._calc_entry_integrals(ym[0], ym[1], ym[2], ym[3], ym[4], ym[5], ym[6], ym[7], b, c_comp)
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
        c_comp: float = 0.0
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
        y_vec: np.ndarray
    ) -> np.ndarray:
        """
        RHS derivative vector for the Merged Downstream Zone (6 ODEs):
        y = [delta1, e1, e2, Ur, Ut, p]
        Uses decoupled sum/difference formulation for full numerical stability
        and Johnston's polar hodograph closure at maximum velocity line.
        """
        cfg = self.config
        b, db_dr = width_at_radius(r, cfg.r1, cfg.r2, cfg.b1, cfg.b2)

        d1, ee1, ee2, ur, ut, p_val = y_vec
        d2 = b - d1
        if d1 <= 1e-6 or d2 <= 1e-6:
            return np.zeros(6)

        c_comp = 0.0
        rho_curr = cfg.rho
        mach_curr = 0.0
        if cfg.compressible:
            U_tot = np.sqrt(ur**2 + ut**2)
            cp = cfg.gamma * cfg.R_gas / (cfg.gamma - 1.0)
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
            yp = y5.copy(); yp[j] += h
            ym = y5.copy(); ym[j] -= h
            J_ints[:, j] = (
                self._calc_merged_integrals(yp[0], yp[1], yp[2], yp[3], yp[4], b, c_comp)
                - self._calc_merged_integrals(ym[0], ym[1], ym[2], ym[3], ym[4], b, c_comp)
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

        # 5: Eq. 35 at maximum velocity: Johnston polar hodograph closure
        # tan_beta = (Ur / Ut) / (1 - Ub / Ut) with Ub / Ut ~ 0.72 (Johnston 1960)
        tan_beta_val = cfg.tan_beta if (cfg.tan_beta is not None and cfg.tan_beta > 0.0) else (3.6 * (ur / ut) if ut > 1e-6 else 1.0)
        A[5, 3] = ur
        A[5, 4] = - tan_beta_val * ur
        A[5, 5] = 1.0 / rho_curr
        b_vec[5] = ut**2 / r + tan_beta_val * ur * ut / r

        try:
            ydot = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            ydot = np.linalg.lstsq(A, b_vec, rcond=None)[0]

        return ydot

    def solve(
        self,
        nr: Optional[int] = None,
        nspan: Optional[int] = None
    ) -> SenooFlowField:
        """
        Solves the Senoo equations from diffuser inlet to outlet.
        Returns a SenooFlowField object containing the complete 2D and 1D flow field.
        """
        cfg = self.config
        nr = nr if nr is not None else cfg.nr
        nspan = nspan if nspan is not None else cfg.nspan

        # Compute inlet core velocities
        alpha_rad = np.radians(cfg.alpha1_mean)
        Ur_mean = cfg.V1_mean * np.sin(alpha_rad)
        Ut_mean = cfg.V1_mean * np.cos(alpha_rad)

        ur1_0 = Ur_mean * (1.0 - 0.5 * cfg.distortion_ur)
        ur2_0 = Ur_mean * (1.0 + 0.5 * cfg.distortion_ur)
        ut1_0 = Ut_mean * (1.0 - 0.5 * cfg.distortion_utheta)
        ut2_0 = Ut_mean * (1.0 + 0.5 * cfg.distortion_utheta)

        d1_0 = cfg.delta1_hub
        d2_0 = cfg.delta1_shroud
        e1_0 = cfg.e1_hub
        e2_0 = cfg.e1_shroud
        p_0 = cfg.p1

        # Initial state in Entry Zone
        y_curr = np.array([d1_0, e1_0, d2_0, e2_0, ur1_0, ut1_0, ur2_0, ut2_0, p_0], dtype=float)
        r_curr = cfg.r1
        dr = cfg.dr_max

        r_steps: List[float] = [r_curr]
        y_steps: List[np.ndarray] = [y_curr.copy()]
        zones: List[int] = [0] # 0 for Entry, 1 for Merged

        r_merge: Optional[float] = None
        merged_mode = False
        step = 0

        # Run integration loop
        while r_curr < cfg.r2 - 1e-9:
            b_local, _ = width_at_radius(r_curr, cfg.r1, cfg.r2, cfg.b1, cfg.b2)
            step_dr = min(dr, cfg.r2 - r_curr)

            if not merged_mode:
                # Check for boundary layer merging
                if (y_curr[0] + y_curr[2]) >= 0.98 * b_local:
                    merged_mode = True
                    r_merge = r_curr
                    # Transition to 6-variable state: [delta1, e1, e2, Ur, Ut, p]
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
                # Keep physical constraints
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

        # Build regular output radial grid
        r_grid = np.linspace(cfg.r1, cfg.r2, nr)
        span_grid = np.linspace(0.0, 1.0, nspan)
        b_grid = np.array([width_at_radius(r_val, cfg.r1, cfg.r2, cfg.b1, cfg.b2)[0] for r_val in r_grid])

        R_2d, SPAN_2d = np.meshgrid(r_grid, span_grid, indexing='ij')
        Y_2d = SPAN_2d * b_grid[:, np.newaxis]

        # Interpolate state variables onto r_grid
        r_arr = np.array(r_steps)
        p_1d = np.zeros(nr)
        delta_hub_1d = np.zeros(nr)
        delta_shroud_1d = np.zeros(nr)
        e_hub_1d = np.zeros(nr)
        e_shroud_1d = np.zeros(nr)
        Ur_hub_1d = np.zeros(nr)
        Ut_hub_1d = np.zeros(nr)
        Ur_shroud_1d = np.zeros(nr)
        Ut_shroud_1d = np.zeros(nr)
        zone_1d = np.zeros(nr, dtype=int)

        # Unpack raw step arrays
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
            b_loc, _ = width_at_radius(r_steps[k], cfg.r1, cfg.r2, cfg.b1, cfg.b2)
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
        zone_1d = np.array([1 if (r_merge is not None and r_val >= r_merge) else 0 for r_val in r_grid])

        # 2D flow field fields
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

            # Reverse flow detection criterion (Paper 2, Eq. 14 / Paper 1)
            # Wall 1 reverse flow if 1 - e1 * cot(alpha1) <= 0  <=>  e1 >= tan(alpha1)
            cot_a1 = ut1 / ur1 if abs(ur1) > 1e-9 else 0.0
            cot_a2 = ut2 / ur2 if abs(ur2) > 1e-9 else 0.0
            if (1.0 - e1 * cot_a1) <= 0.0:
                has_rev_hub[i] = True
            if (1.0 - e2 * cot_a2) <= 0.0:
                has_rev_shroud[i] = True

            # Thermodynamic state for density calculation
            U_ref = np.sqrt(0.5 * (ur1**2 + ur2**2 + ut1**2 + ut2**2))
            cp = cfg.gamma * cfg.R_gas / (cfg.gamma - 1.0)
            T_loc = max(cfg.T0 - 0.5 * U_ref**2 / cp, 100.0) if cfg.compressible else (cfg.p1 / (cfg.rho * cfg.R_gas))
            rho_star = p_loc / (cfg.R_gas * T_loc) if cfg.compressible else cfg.rho

            for j in range(nspan):
                y = Y_2d[i, j]
                c_fac = 0.0

                if y <= d1:
                    eta = max(min(y / d1, 1.0), 1e-6)
                    eta_pow = eta**(1.0 / n_p)
                    om_eta_m = (1.0 - eta)**m_p
                    term_r1 = ur1 - e1 * ut1 * om_eta_m
                    term_r2 = 1.0 - Gr * (d1 / ur1) * (1.0 - eta) if abs(ur1) > 1e-9 else 1.0
                    term_t1 = ut1 + e1 * ur1 * om_eta_m
                    term_t2 = 1.0 - Gt * (d1 / ut1) * (1.0 - eta) if abs(ut1) > 1e-9 else 1.0
                    vr = eta_pow * term_r1 * term_r2
                    vt = eta_pow * term_t1 * term_t2
                    c_fac = (1.0 - eta**2)
                elif y >= b_loc - d2:
                    eta = max(min((b_loc - y) / d2, 1.0), 1e-6)
                    eta_pow = eta**(1.0 / n_p)
                    om_eta_m = (1.0 - eta)**m_p
                    term_r1 = ur2 - e2 * ut2 * om_eta_m
                    term_r2 = 1.0 + Gr * (d2 / ur2) * (1.0 - eta) if abs(ur2) > 1e-9 else 1.0
                    term_t1 = ut2 + e2 * ur2 * om_eta_m
                    term_t2 = 1.0 + Gt * (d2 / ut2) * (1.0 - eta) if abs(ut2) > 1e-9 else 1.0
                    vr = eta_pow * term_r1 * term_r2
                    vt = eta_pow * term_t1 * term_t2
                    c_fac = (1.0 - eta**2)
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
            if np.any(Vr_2d[i, :nspan//4] < -1e-5):
                has_rev_hub[i] = True
            if np.any(Vr_2d[i, 3*nspan//4:] < -1e-5):
                has_rev_shroud[i] = True

        V_2d = np.sqrt(Vr_2d**2 + Vt_2d**2)
        Alpha_2d = np.degrees(np.arctan2(Vr_2d, Vt_2d))

        q_dyn_inlet = 0.5 * cfg.rho * cfg.V1_mean**2
        Cp_1d = (p_1d - cfg.p1) / q_dyn_inlet if q_dyn_inlet > 0 else np.zeros_like(p_1d)

        # Summary stall properties
        is_stalled = bool(np.any(has_rev_hub) or np.any(has_rev_shroud))
        r_stall_hub = float(r_grid[np.where(has_rev_hub)[0][0]]) if np.any(has_rev_hub) else None
        r_stall_shroud = float(r_grid[np.where(has_rev_shroud)[0][0]]) if np.any(has_rev_shroud) else None

        return SenooFlowField(
            r=r_grid,
            span=span_grid,
            b_r=b_grid,
            R=R_2d,
            SPAN=SPAN_2d,
            Y=Y_2d,
            vr=Vr_2d,
            vt=Vt_2d,
            V=V_2d,
            alpha=Alpha_2d,
            rho=Rho_2d,
            mach=Mach_2d if cfg.compressible else None,
            p=p_1d,
            Cp=Cp_1d,
            delta_hub=delta_hub_1d,
            delta_shroud=delta_shroud_1d,
            e_hub=e_hub_1d,
            e_shroud=e_shroud_1d,
            Ur_hub=Ur_hub_1d,
            Ut_hub=Ut_hub_1d,
            Ur_shroud=Ur_shroud_1d,
            Ut_shroud=Ut_shroud_1d,
            has_reverse_flow_hub=has_rev_hub,
            has_reverse_flow_shroud=has_rev_shroud,
            zone=zone_1d,
            is_stalled=is_stalled,
            r_stall_hub=r_stall_hub,
            r_stall_shroud=r_stall_shroud,
            r_merge=r_merge
        )

    def find_critical_flow_angle(
        self,
        tol: float = 0.05,
        max_iter: int = 25
    ) -> float:
        """
        Determines the critical inlet flow angle alpha_cr [deg] for reverse flow onset
        using bisection search across flow angles.
        Matches the critical flow angle analyses of Senoo & Kinoshita (1977).
        """
        # Save original angle
        orig_angle = self.config.alpha1_mean

        # Search bounds: large angle (unstalled) and small angle (stalled)
        a_high = 35.0
        a_low = 3.0

        # Check that high is unstalled
        self.config.alpha1_mean = a_high
        sol_high = self.solve(nr=40, nspan=20)
        if sol_high.is_stalled:
            # Expand upper bound
            a_high = 50.0

        # Check that low is stalled
        self.config.alpha1_mean = a_low
        sol_low = self.solve(nr=40, nspan=20)
        if not sol_low.is_stalled:
            a_low = 1.0

        # Bisection search
        alpha_cr = 0.5 * (a_low + a_high)
        for _ in range(max_iter):
            if abs(a_high - a_low) < tol:
                break
            a_mid = 0.5 * (a_low + a_high)
            self.config.alpha1_mean = a_mid
            sol_mid = self.solve(nr=50, nspan=25)

            if sol_mid.is_stalled:
                # Flow is reversed -> critical angle is higher
                a_low = a_mid
            else:
                # Flow is forward -> critical angle is lower
                a_high = a_mid
            alpha_cr = 0.5 * (a_low + a_high)

        # Restore original config
        self.config.alpha1_mean = orig_angle
        return float(alpha_cr)
