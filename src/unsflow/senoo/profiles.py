"""
unsflow.senoo.profiles
======================
Analytical velocity profiles, integral moments via Beta functions,
and turbulent wall shear stress closures for the Senoo vaneless diffuser model.

References:
- Senoo, Y., Kinoshita, Y., and Ishida, M., "Asymmetric Flow in Vaneless
  Diffusers of Centrifugal Blowers," ASME J. Fluids Eng., 99(1), 104-111, 1977.
- Senoo, Y., and Kinoshita, Y., "Influence of Inlet Flow Conditions and
  Geometries of Centrifugal Vaneless Diffusers on Critical Flow Angle for
  Reverse Flow," ASME J. Fluids Eng., 99(1), 98-103, 1977.
"""

from typing import Dict, Tuple, Optional, Callable
import numpy as np
from scipy.special import beta


def compute_beta_table(n: float, m: float) -> Dict[int, Dict[int, float]]:
    """
    Computes lookup table of exact integrals:
        J(k, l) = \\int_0^1 \\eta^{k/n} (1 - \\eta)^l d\\eta = B(k/n + 1, l + 1)
    for k in {1, 2} and l in {0, ..., 2*m + 2}.
    This closed-form formulation replaces numerical quadrature with exact Beta functions.
    """
    table: Dict[int, Dict[int, float]] = {}
    for k in [1, 2]:
        p = k / n + 1.0
        table[k] = {}
        for l in range(2 * int(m) + 3):
            table[k][l] = float(beta(p, l + 1.0))
    return table


def evaluate_layer_integrals(
    delta: float,
    e: float,
    Ur: float,
    Ut: float,
    Gr: float,
    Gt: float,
    m: float,
    n: float,
    wall: int,
    beta_tab: Dict[int, Dict[int, float]],
    c_comp: float = 0.0,
) -> Tuple[float, float, float, float]:
    """
    Evaluates profile integrals across a boundary layer:
        Ir  = \\int_0^\\delta v_r dy = \\delta \\int_0^1 v_r d\\eta
        Irr = \\int_0^\\delta v_r^2 dy = \\delta \\int_0^1 v_r^2 d\\eta
        Itt = \\int_0^\\delta v_\\theta^2 dy = \\delta \\int_0^1 v_\\theta^2 d\\eta
        Irt = \\int_0^\\delta v_r v_\\theta dy = \\delta \\int_0^1 v_r v_\\theta d\\eta

    Parameters
    ----------
    delta : float
        Boundary layer thickness [m].
    e : float
        Skew parameter (e = \\tan(\\gamma)).
    Ur, Ut : float
        Core radial and tangential velocities at boundary layer edge [m/s].
    Gr, Gt : float
        Radial and tangential velocity gradients across core flow [1/s].
    m : float
        Crossflow velocity profile exponent (typically 3.0).
    n : float
        Streamwise velocity power law exponent (typically 7.0).
    wall : int
        1 for hub wall (negative gradient sign), 2 for shroud wall (positive gradient sign).
    beta_tab : dict
        Precomputed Beta function lookup table from `compute_beta_table`.
    c_comp : float
        Compressibility correction factor c = \\zeta * U^2 / (2 cp T).

    Returns
    -------
    Ir, Irr, Itt, Irt : float
        Integral moments of the boundary layer.
    """
    s = -1.0 if wall == 1 else 1.0

    Cr = s * Gr * delta / Ur if abs(Ur) > 1e-12 else 0.0
    Ct = s * Gt * delta / Ut if abs(Ut) > 1e-12 else 0.0

    Ar = Ur
    Br = -e * Ut
    At = Ut
    Bt = e * Ur

    m_int = int(m)
    tab1 = beta_tab[1]
    tab2 = beta_tab[2]

    # vr d_eta = [Ar + Br (1-eta)^m] * [1 + Cr (1-eta)]
    int_vr = (
        Ar * tab1[0]
        + Ar * Cr * tab1[1]
        + Br * tab1[m_int]
        + Br * Cr * tab1[m_int + 1]
    )

    # vt d_eta = [At + Bt (1-eta)^m] * [1 + Ct (1-eta)]
    int_vt = (
        At * tab1[0]
        + At * Ct * tab1[1]
        + Bt * tab1[m_int]
        + Bt * Ct * tab1[m_int + 1]
    )

    # vr^2 d_eta
    P_r = np.zeros(2 * m_int + 3)
    C_poly = np.array([1.0, 2.0 * Cr, Cr**2])
    P_r[0:3] += Ar**2 * C_poly
    P_r[m_int : m_int + 3] += 2.0 * Ar * Br * C_poly
    P_r[2 * m_int : 2 * m_int + 3] += Br**2 * C_poly
    int_vrr = sum(P_r[l] * tab2[l] for l in range(len(P_r)))

    # vt^2 d_eta
    P_t = np.zeros(2 * m_int + 3)
    C_poly_t = np.array([1.0, 2.0 * Ct, Ct**2])
    P_t[0:3] += At**2 * C_poly_t
    P_t[m_int : m_int + 3] += 2.0 * At * Bt * C_poly_t
    P_t[2 * m_int : 2 * m_int + 3] += Bt**2 * C_poly_t
    int_vtt = sum(P_t[l] * tab2[l] for l in range(len(P_t)))

    # vr * vt d_eta
    P_rt = np.zeros(2 * m_int + 3)
    C_cross = np.array([1.0, Cr + Ct, Cr * Ct])
    P_rt[0:3] += (Ar * At) * C_cross
    P_rt[m_int : m_int + 3] += (Ar * Bt + At * Br) * C_cross
    P_rt[2 * m_int : 2 * m_int + 3] += (Br * Bt) * C_cross
    int_vrt = sum(P_rt[l] * tab2[l] for l in range(len(P_rt)))

    # Density correction factor if c_comp > 0:
    # rho(eta)/rho* = 1 - c*(1 - eta^2) => mean factor ~ (1 - 0.5 * c_comp)
    if c_comp > 0.0:
        rho_factor = 1.0 - 0.5 * c_comp
        int_vr *= rho_factor
        int_vrr *= rho_factor
        int_vtt *= rho_factor
        int_vrt *= rho_factor

    Ir = delta * int_vr
    Irr = delta * int_vrr
    Itt = delta * int_vtt
    Irt = delta * int_vrt

    return Ir, Irr, Itt, Irt


def evaluate_core_integrals(
    hc: float,
    Ur1: float,
    Ur2: float,
    Ut1: float,
    Ut2: float,
) -> Tuple[float, float, float, float]:
    """
    Evaluates exact polynomial integrals across the inviscid core flow:
        Jr  = \\int_0^{hc} Ur(xi) dxi
        Jrr = \\int_0^{hc} Ur^2(xi) dxi
        Jtt = \\int_0^{hc} Ut^2(xi) dxi
        Jrt = \\int_0^{hc} Ur(xi) Ut(xi) dxi
    where velocities vary linearly across hc between wall 1 and wall 2.
    """
    Jr = hc * 0.5 * (Ur1 + Ur2)
    Jrr = hc * (Ur1**2 + Ur1 * Ur2 + Ur2**2) / 3.0
    Jtt = hc * (Ut1**2 + Ut1 * Ut2 + Ut2**2) / 3.0
    Jrt = hc * (Ur1 * Ut1 / 3.0 + (Ur1 * Ut2 + Ur2 * Ut1) / 6.0 + Ur2 * Ut2 / 3.0)
    return Jr, Jrr, Jtt, Jrt


def wall_shear_stresses(
    Ur: float,
    Ut: float,
    delta: float,
    e: float,
    rho: float,
    nu: float,
    Kf: float = 0.045,
    mach: float = 0.0,
    compressible: bool = False,
) -> Tuple[float, float]:
    """
    Computes wall shear stress components tau_0r and tau_0theta.
    From Senoo & Kinoshita (1977) Paper 2 Eqs. (18)-(22) and Paper 1 Eqs. (6)-(7):
        tau_0r = 0.5 * Kf * rho * U^2 * R_delta^(-1/4) * (sin(alpha) - e * cos(alpha)) * lambda(M)
        tau_0theta = 0.5 * Kf * rho * U^2 * R_delta^(-1/4) * (cos(alpha) + e * sin(alpha)) * lambda(M)
    """
    U = np.sqrt(Ur**2 + Ut**2)
    if U < 1e-9:
        return 0.0, 0.0

    R_delta = max(U * delta / nu, 10.0)
    Cfu = Kf * (R_delta ** (-0.25))

    # Mager compressibility correction factor (Paper 1, Eq. 7)
    if compressible and mach > 0.05:
        lam = 1.0 - 0.0695 * mach + 0.0857 * mach**2 - 0.017 * mach**3
        Cfu *= max(lam, 0.5)

    sin_a = Ur / U
    cos_a = Ut / U

    tau_0r = 0.5 * rho * U**2 * Cfu * (sin_a - e * cos_a)
    tau_0theta = 0.5 * rho * U**2 * Cfu * (cos_a + e * sin_a)

    return tau_0r, tau_0theta


def width_at_radius(
    r: float,
    r1: float,
    r2: float,
    b1: float,
    b2: float,
    b_func: Optional[Callable[[float], float]] = None,
    db_dr_func: Optional[Callable[[float], float]] = None,
) -> Tuple[float, float]:
    """
    Returns local diffuser passage width b(r) [m] and derivative db/dr [-].

    Parameters
    ----------
    r : float
        Current radial station [m].
    r1, r2 : float
        Diffuser inlet and exit radii [m].
    b1, b2 : float
        Diffuser inlet and exit passage widths [m].
    b_func : callable or None, optional
        Custom function b(r) [m]. If provided, evaluated instead of linear taper.
    db_dr_func : callable or None, optional
        Custom derivative function db/dr [-]. If None and b_func is provided,
        approximated using central differences.

    Returns
    -------
    b : float
        Local passage width [m].
    db_dr : float
        Local passage width derivative with respect to radius [-].
    """
    if b_func is not None:
        b = float(b_func(r))
        if db_dr_func is not None:
            db_dr = float(db_dr_func(r))
        else:
            hr = 1e-6 * max(abs(r2 - r1), 0.01)
            b_plus = float(b_func(r + hr))
            b_minus = float(b_func(r - hr))
            db_dr = (b_plus - b_minus) / (2.0 * hr)
        return b, db_dr

    if abs(r2 - r1) < 1e-12:
        return b1, 0.0
    db_dr = (b2 - b1) / (r2 - r1)
    b = b1 + db_dr * (r - r1)
    return b, db_dr
