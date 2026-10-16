"""
unsflow.senoo.results
=====================
Data structures and visualization methods for Senoo vaneless diffuser solutions.
Stores operating, rotating stall, and surge angles in the radial-reference
convention (alpha) while maintaining full access to the complementary
tangential convention (beta = 90 - alpha).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np

from unsflow.utils.formatting import print_banner, total_chars, total_chars_mid


@dataclass
class SenooResults:
    """
    Container for Senoo vaneless diffuser stability analysis and flow field results.

    Flow Angle Conventions:
    - alpha: flow angle measured from the radial direction [deg] (0 = pure radial, 90 = pure tangential)
    - beta: flow angle measured from the tangential direction [deg] (beta = 90 - alpha)

    Stability Criteria:
    - Rotating Stall occurs when alpha >= alpha_rotating_stall (or beta <= beta_rotating_stall)
    - Surge occurs when alpha >= alpha_surge (or beta <= beta_surge)
    """

    # --- Scalar Operating Angles ---
    alpha_op: float = 0.0                   # Operating mean radial flow angle [deg]
    beta_op: float = 0.0                    # Operating mean tangential flow angle [deg] (90 - alpha_op)

    # --- Stability Angles (Radial Convention, alpha) ---
    alpha_rotating_stall: float = 0.0       # Critical angle for rotating stall onset [deg]
    alpha_surge: float = 0.0                # Critical angle for surge onset [deg]
    alpha_reverse_flow: float = 0.0         # Critical angle for wall reverse flow onset [deg]

    # --- Complementary Stability Angles (Tangential Convention, beta) ---
    beta_rotating_stall: float = 0.0        # Rotating stall angle [deg] (90 - alpha_rotating_stall)
    beta_surge: float = 0.0                 # Surge angle [deg] (90 - alpha_surge)
    beta_reverse_flow: float = 0.0          # Reverse flow angle [deg] (90 - alpha_reverse_flow)

    # --- Stability Flags and Margins ---
    is_stalled: bool = False                # True if operating point is in rotating stall (alpha >= alpha_RS)
    is_surging: bool = False                # True if operating point is in surge (alpha >= alpha_surge)
    has_reversed_flow: bool = False         # True if wall reverse flow is present anywhere in diffuser
    stall_margin: float = 0.0               # alpha_rotating_stall - alpha [deg] (> 0 means stable margin)
    surge_margin: float = 0.0               # alpha_surge - alpha [deg] (> 0 means stable margin)

    # --- Compressibility Information ---
    compressible_mode: bool = False         # Whether compressibility correction was applied
    mach_inlet: float = 0.0                 # Inlet Mach number
    density_ratio_crit: float = 1.0         # rho* / rho_i at reversal onset location
    density_ratio_exit: float = 1.0         # rho_exit / rho_i at diffuser exit
    delta_beta_c: float = 0.0               # Compressibility shift on tangential angle [deg]
    delta_alpha_c: float = 0.0              # Compressibility shift on radial angle [deg] (-delta_beta_c)

    # --- Reversal Locations ---
    reversal_radius: Optional[float] = None # Radius where wall reverse flow first initiates [m]
    r_stall_hub: Optional[float] = None     # Radius of hub reverse flow onset [m]
    r_stall_shroud: Optional[float] = None  # Radius of shroud reverse flow onset [m]
    r_merge: Optional[float] = None         # Radius where boundary layers merge [m]

    # --- 1D Radial Distributions ---
    r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b_r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    p: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Cp: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rho_1d: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mach_1d: np.ndarray = field(default_factory=lambda: np.zeros(0))
    alpha_1d: np.ndarray = field(default_factory=lambda: np.zeros(0))
    delta_hub: np.ndarray = field(default_factory=lambda: np.zeros(0))
    delta_shroud: np.ndarray = field(default_factory=lambda: np.zeros(0))
    e_hub: np.ndarray = field(default_factory=lambda: np.zeros(0))
    e_shroud: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Ur_hub: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Ut_hub: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Ur_shroud: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Ut_shroud: np.ndarray = field(default_factory=lambda: np.zeros(0))
    has_reverse_flow_hub: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    has_reverse_flow_shroud: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    zone: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    # --- 2D Mesh and Flow Field Arrays ---
    span: np.ndarray = field(default_factory=lambda: np.zeros(0))
    R: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    SPAN: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Y: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    vr: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    vt: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    V: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    alpha_2d: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    rho: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mach: Optional[np.ndarray] = None

    def __post_init__(self):
        # Synchronize angles
        if self.beta_op == 0.0 and self.alpha_op != 0.0:
            self.beta_op = 90.0 - self.alpha_op
        elif self.alpha_op == 0.0 and self.beta_op != 0.0:
            self.alpha_op = 90.0 - self.beta_op

        if self.beta_rotating_stall == 0.0 and self.alpha_rotating_stall != 0.0:
            self.beta_rotating_stall = 90.0 - self.alpha_rotating_stall
        elif self.alpha_rotating_stall == 0.0 and self.beta_rotating_stall != 0.0:
            self.alpha_rotating_stall = 90.0 - self.beta_rotating_stall

        if self.beta_surge == 0.0 and self.alpha_surge != 0.0:
            self.beta_surge = 90.0 - self.alpha_surge
        elif self.alpha_surge == 0.0 and self.beta_surge != 0.0:
            self.alpha_surge = 90.0 - self.beta_surge

        if self.beta_reverse_flow == 0.0 and self.alpha_reverse_flow != 0.0:
            self.beta_reverse_flow = 90.0 - self.alpha_reverse_flow
        elif self.alpha_reverse_flow == 0.0 and self.beta_reverse_flow != 0.0:
            self.alpha_reverse_flow = 90.0 - self.beta_reverse_flow

        self.stall_margin = self.alpha_rotating_stall - self.alpha_op
        self.surge_margin = self.alpha_surge - self.alpha_op

    @property
    def cp(self) -> np.ndarray:
        """Alias for Cp."""
        return self.Cp

    @property
    def alpha(self) -> np.ndarray:
        """
        Returns the 2D array of radial flow angles across (r, span) [deg].
        For the scalar operating angle, use `alpha_operating` or `alpha_op`.
        """
        return self.alpha_2d

    @property
    def alpha_operating(self) -> float:
        """Operating radial flow angle [deg]."""
        return self.alpha_op

    @property
    def beta_operating(self) -> float:
        """Operating tangential flow angle [deg]."""
        return self.beta_op

    @property
    def tangential_angle(self) -> float:
        """Operating tangential flow angle beta [deg]."""
        return self.beta_op

    @property
    def rotating_stall_angle(self) -> float:
        """Critical rotating stall angle alpha_RS [deg] (radial convention)."""
        return self.alpha_rotating_stall

    @property
    def surge_angle(self) -> float:
        """Critical surge angle alpha_surge [deg] (radial convention)."""
        return self.alpha_surge

    @property
    def reverse_flow_angle(self) -> float:
        """Critical reverse flow angle alpha_rev [deg] (radial convention)."""
        return self.alpha_reverse_flow

    def print_summary(self):
        """Print structured summary banner of stability limits and flow state."""
        print_banner("SENOO DIFFUSER STABILITY & FLOW SUMMARY")
        print(f"{'Operating Radial Flow Angle alpha [deg]:':<{total_chars_mid}}{self.alpha_op:>{total_chars_mid}.2f}")
        print(f"{'Operating Tangential Angle beta [deg]:':<{total_chars_mid}}{self.beta_op:>{total_chars_mid}.2f}")
        print(f"{'Reverse Flow Angle alpha_rev [deg]:':<{total_chars_mid}}{self.alpha_reverse_flow:>{total_chars_mid}.2f}")
        print(f"{'Rotating Stall Angle alpha_RS [deg]:':<{total_chars_mid}}{self.alpha_rotating_stall:>{total_chars_mid}.2f}")
        print(f"{'Surge Angle alpha_surge [deg]:':<{total_chars_mid}}{self.alpha_surge:>{total_chars_mid}.2f}")
        print(f"{'Rotating Stall Margin Delta_alpha [deg]:':<{total_chars_mid}}{self.stall_margin:>{total_chars_mid}.2f}")
        print(f"{'Surge Margin Delta_alpha [deg]:':<{total_chars_mid}}{self.surge_margin:>{total_chars_mid}.2f}")
        print(f"{'Rotating Stall Inception Status:':<{total_chars_mid}}{('STALLED' if self.is_stalled else 'STABLE'):>{total_chars_mid}}")
        print(f"{'Surge Inception Status:':<{total_chars_mid}}{('SURGING' if self.is_surging else 'STABLE'):>{total_chars_mid}}")
        print(f"{'Wall Reverse Flow Detected:':<{total_chars_mid}}{str(self.has_reversed_flow):>{total_chars_mid}}")

        if self.reversal_radius is not None:
            print(f"{'Reverse Flow Onset Radius r_rev [m]:':<{total_chars_mid}}{self.reversal_radius:>{total_chars_mid}.4f}")
        if self.r_merge is not None:
            print(f"{'Boundary Layer Merge Radius [m]:':<{total_chars_mid}}{self.r_merge:>{total_chars_mid}.4f}")
        if len(self.Cp) > 0:
            print(f"{'Static Pressure Recovery Cp,exit [-]:':<{total_chars_mid}}{self.Cp[-1]:>{total_chars_mid}.4f}")
        if self.compressible_mode:
            print(f"{'Inlet Mach Number M1 [-]:':<{total_chars_mid}}{self.mach_inlet:>{total_chars_mid}.3f}")
            print(f"{'Exit Density Ratio rho_exit/rho_in [-]:':<{total_chars_mid}}{self.density_ratio_exit:>{total_chars_mid}.4f}")
            print(f"{'Compressibility Angle Shift Delta_alpha [deg]:':<{total_chars_mid}}{self.delta_alpha_c:>{total_chars_mid}.2f}")
        print_banner()

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
        d = {
            'alpha': self.alpha_2d,
            'alpha_operating': self.alpha_op,
            'beta_operating': self.beta_op,
            'alpha_rotating_stall': self.alpha_rotating_stall,
            'alpha_surge': self.alpha_surge,
            'alpha_reverse_flow': self.alpha_reverse_flow,
            'beta_rotating_stall': self.beta_rotating_stall,
            'beta_surge': self.beta_surge,
            'beta_reverse_flow': self.beta_reverse_flow,
            'is_stalled': self.is_stalled,
            'is_surging': self.is_surging,
            'has_reversed_flow': self.has_reversed_flow,
            'stall_margin': self.stall_margin,
            'surge_margin': self.surge_margin,
            'r': self.r,
            'p': self.p,
            'Cp': self.Cp,
            'vr': self.vr,
            'vt': self.vt,
            'V': self.V,
        }
        if self.mach is not None:
            d['mach'] = self.mach
        return d


# Alias for backwards compatibility with earlier codebase
SenooFlowField = SenooResults
