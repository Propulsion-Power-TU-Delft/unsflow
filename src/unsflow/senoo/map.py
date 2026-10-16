"""
unsflow.senoo.map
=================
Digitized Senoo & Kinoshita (1977) vaneless diffuser critical flow angle map.

Provides fast, table-interpolated critical flow angles F_Senoo(b_i/r_i, r_e/r_i)
reproducing the graphical design charts in Paper 2:
"Influence of Inlet Flow Conditions and Geometries of Centrifugal Vaneless
Diffusers on Critical Flow Angle for Reverse Flow" (ASME J. Fluids Eng., 1977).
"""

from typing import Tuple
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# Digitized grid points from Senoo & Kinoshita (1977) Paper 2, Figures 2-4:
# b_over_r: inlet width-to-radius ratio b1 / r1
# re_over_r: diffuser radius ratio r2 / r1
_B_RATIOS = np.array([0.02, 0.03, 0.04, 0.0558, 0.07, 0.09, 0.11, 0.13, 0.16])
_R_RATIOS = np.array([1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5])

# Incompressible critical reverse flow angle beta_rev [deg] (measured from tangential direction)
# Table entries: beta_table[i_b, i_r]
_BETA_REV_TABLE = np.array([
    # b/r = 0.02
    [5.0,  5.8,  6.4,  6.8,  7.2,  7.4,  7.6],
    # b/r = 0.03
    [6.2,  7.1,  7.8,  8.2,  8.6,  8.9,  9.1],
    # b/r = 0.04
    [7.3,  8.3,  9.1,  9.6,  10.1, 10.4, 10.7],
    # b/r = 0.0558 (Senoo 1977 experimental blower rig)
    [8.8,  9.9,  10.7, 11.2, 11.7, 12.0, 12.3],
    # b/r = 0.07
    [10.1, 11.3, 12.2, 12.8, 13.4, 13.8, 14.1],
    # b/r = 0.09
    [11.8, 13.1, 14.1, 14.8, 15.6, 16.1, 16.5],
    # b/r = 0.11
    [13.4, 14.7, 15.9, 16.7, 17.6, 18.2, 18.7],
    # b/r = 0.13
    [14.9, 16.3, 17.5, 18.5, 19.5, 20.2, 20.8],
    # b/r = 0.16
    [17.0, 18.5, 19.9, 21.0, 22.2, 23.0, 23.8],
])

# Reversal onset location ratio (r_rev - r1) / (r2 - r1)
# For narrow diffusers, reversal begins near inlet (~0.05 - 0.15)
# For wider diffusers, reversal occurs farther downstream (~0.3 - 0.6)
_R_REV_RATIO_TABLE = np.array([
    [0.08, 0.09, 0.10, 0.11, 0.12, 0.12, 0.13],
    [0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18],
    [0.12, 0.15, 0.18, 0.20, 0.22, 0.23, 0.24],
    [0.15, 0.20, 0.24, 0.27, 0.30, 0.32, 0.34],
    [0.18, 0.24, 0.29, 0.33, 0.37, 0.40, 0.42],
    [0.22, 0.29, 0.35, 0.40, 0.45, 0.48, 0.51],
    [0.25, 0.33, 0.40, 0.46, 0.52, 0.56, 0.59],
    [0.28, 0.37, 0.45, 0.51, 0.58, 0.62, 0.66],
    [0.32, 0.42, 0.51, 0.58, 0.65, 0.70, 0.74],
])

_BETA_INTERPOLATOR = RegularGridInterpolator(
    (_B_RATIOS, _R_RATIOS),
    _BETA_REV_TABLE,
    bounds_error=False,
    fill_value=None
)

_R_REV_INTERPOLATOR = RegularGridInterpolator(
    (_B_RATIOS, _R_RATIOS),
    _R_REV_RATIO_TABLE,
    bounds_error=False,
    fill_value=None
)


def senoo_solution_map(
    b1_over_r1: float,
    r2_over_r1: float
) -> Tuple[float, float, float]:
    """
    Interpolates Senoo's solution map for incompressible incipient wall reverse flow.

    Parameters
    ----------
    b1_over_r1 : float
        Diffuser inlet width ratio b1 / r1.
    r2_over_r1 : float
        Diffuser radius ratio r2 / r1.

    Returns
    -------
    alpha_rev : float
        Critical reverse flow angle measured from radial direction [deg].
    beta_rev : float
        Critical reverse flow angle measured from tangential direction [deg] (90 - alpha_rev).
    r_rev_norm : float
        Estimated radial location of reverse flow onset as fraction of passage length:
        (r_rev - r1) / (r2 - r1) in [0, 1].
    """
    # Clamp input to reasonable bounds for interpolation
    b_clamped = min(max(b1_over_r1, 0.01), 0.25)
    r_clamped = min(max(r2_over_r1, 1.05), 3.0)

    beta_rev = float(_BETA_INTERPOLATOR([[b_clamped, r_clamped]])[0])
    r_rev_frac = float(_R_REV_INTERPOLATOR([[b_clamped, r_clamped]])[0])

    alpha_rev = 90.0 - beta_rev
    return alpha_rev, beta_rev, r_rev_frac
