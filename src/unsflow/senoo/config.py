"""
unsflow.senoo.config
====================
Configuration dataclass and factory methods for the Senoo vaneless diffuser model.
Supports both dimensional and non-dimensional inputs, ideal gases, and real fluid
properties via CoolProp.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Callable, Tuple
import os
import configparser
import numpy as np
from scipy.interpolate import PchipInterpolator

try:
    from CoolProp.CoolProp import PropsSI, PhaseSI
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False

from unsflow.utils.formatting import print_banner, total_chars, total_chars_mid
from unsflow.senoo.profiles import width_at_radius


@dataclass
class SenooConfig:
    """
    Configuration for Senoo & Kinoshita vaneless diffuser rotating stall
    and critical flow angle calculations.

    Angle Conventions:
    - alpha: angle measured from the radial direction [deg] (0 deg = pure radial, 90 deg = pure tangential)
    - beta: angle measured from the tangential direction [deg] (beta = 90 deg - alpha)
    """

    # --- Geometry ---
    r1: float = 0.25                    # Diffuser inlet radius [m]
    r2: float = 0.50                    # Diffuser exit radius [m]
    b1: float = 0.014                   # Diffuser inlet width [m]
    b2: Optional[float] = None          # Diffuser exit width [m] (defaults to b1, parallel walls)
    pinch_ratio: Optional[float] = None # Ratio b2 / b1
    b_func: Optional[Callable[[float], float]] = None # Custom function b(r) [m]
    db_dr_func: Optional[Callable[[float], float]] = None # Custom derivative function db/dr [-]
    r_geom: Optional[Any] = None        # Discrete radius stations for tabular width [m]
    b_geom: Optional[Any] = None        # Discrete passage widths for tabular width [m]

    # --- Fluid & Thermodynamics ---
    fluid: Optional[str] = None         # Fluid name (e.g., 'Air', 'Argon', 'R1233zd(E)'), queried via CoolProp
    gamma: float = 1.4                  # Specific heat ratio (cp / cv)
    R_gas: float = 287.05               # Specific gas constant [J/(kg K)]
    p1: float = 101325.0                # Inlet static pressure [Pa]
    T1: float = 288.15                  # Inlet static temperature [K]
    rho: float = 1.2                    # Fluid density [kg/m^3]
    nu: float = 1.5e-5                  # Kinematic viscosity [m^2/s]
    mu: Optional[float] = None          # Dynamic viscosity [Pa s]
    a_sound: Optional[float] = None     # Speed of sound [m/s]
    cp: Optional[float] = None          # Specific heat at constant pressure [J/(kg K)]
    T0: Optional[float] = None          # Inlet total temperature [K]
    zeta: float = 0.7                   # Thermal wall factor (0.7 adiabatic, 0.0 isothermal)
    compressible: bool = False          # Whether compressibility effects should be modeled

    # --- Flow Conditions ---
    alpha1_mean: float = 73.0           # Mean inlet flow angle from radial direction [deg]
    beta1_mean: Optional[float] = None  # Mean inlet flow angle from tangential direction [deg] (90 - alpha1)
    V1_mean: Optional[float] = None     # Mean inlet velocity magnitude [m/s] (computed from mach1 if None)
    Cr1: Optional[float] = None         # Radial velocity at inlet [m/s]
    Ct1: Optional[float] = None         # Tangential velocity at inlet [m/s]
    mdot: Optional[float] = None        # Mass flow rate [kg/s]
    mach1: float = 0.0                  # Inlet Mach number [-]
    rpm: Optional[float] = None         # Impeller rotational speed [rpm]
    omega: Optional[float] = None       # Impeller angular velocity [rad/s]

    # --- Boundary Layer & Skew Initial Conditions ---
    delta_b_hub: float = 0.05           # Inlet hub boundary layer thickness ratio delta_1 / b1
    delta_b_shroud: float = 0.05        # Inlet shroud boundary layer thickness ratio delta_2 / b1
    delta1_hub: Optional[float] = None  # Physical hub boundary layer thickness [m]
    delta1_shroud: Optional[float] = None # Physical shroud boundary layer thickness [m]
    e1_hub: float = 0.0                 # Hub wall skew parameter e = tan(gamma)
    e1_shroud: float = 0.0              # Shroud wall skew parameter e = tan(gamma)
    distortion_ur: float = 0.0          # Radial velocity distortion across span (Delta Ur / Ur_mean)
    distortion_utheta: float = 0.0      # Tangential velocity distortion across span (Delta Ut / Ut_mean)

    # --- Empirical & Model Parameters ---
    Kf: float = 0.045                   # Skin friction factor constant (0.045 Blasius)
    m: float = 3.0                      # Crossflow profile exponent (3.0 Senoo)
    n_power: Optional[float] = None     # Streamwise power law exponent (defaults to 7.0)
    tan_beta: Optional[float] = None    # Hodograph parameter in merged zone
    rs_factor: float = 0.88             # Rotating stall onset factor on tangential angle (beta_RS = 0.88 * beta_rev)
    surge_method: str = 'diffuser_exit' # 'diffuser_exit' (reverse flow reaches r2) or 'empirical_ratio'
    surge_ratio: float = 0.88           # Empirical surge factor on tangential angle (beta_surge = surge_ratio * beta_RS)
    cf_1d: float = 0.005                # Mean friction factor for 1D flow-with-friction model

    # --- Solver Settings ---
    solver_mode: str = 'ode'            # 'ode' (differential boundary layer marching) or 'map' (digitized Senoo map)
    nr: int = 80                        # Number of radial mesh stations
    nspan: int = 40                     # Number of spanwise mesh stations
    dr_max: float = 0.002               # Maximum radial integration step [m]
    rtol: float = 1e-5                  # ODE relative tolerance
    atol: float = 1e-7                  # ODE absolute tolerance

    def __post_init__(self):
        # 1. Diffuser Geometry
        if self.r_geom is not None and self.b_geom is not None:
            r_arr = np.asarray(self.r_geom, dtype=float)
            b_arr = np.asarray(self.b_geom, dtype=float)
            if len(r_arr) >= 2:
                spline = PchipInterpolator(r_arr, b_arr, extrapolate=True)
                self.b_func = lambda r: float(spline(r))
                self.db_dr_func = lambda r: float(spline.derivative()(r))
                self.b1 = float(spline(self.r1))
                self.b2 = float(spline(self.r2))
                self.pinch_ratio = self.b2 / self.b1 if self.b1 > 0 else 1.0

        elif self.b_func is not None:
            self.b1 = float(self.b_func(self.r1))
            if self.b2 is None:
                self.b2 = float(self.b_func(self.r2))
            self.pinch_ratio = self.b2 / self.b1 if self.b1 > 0 else 1.0

        else:
            if self.b2 is None:
                if self.pinch_ratio is not None:
                    self.b2 = self.b1 * self.pinch_ratio
                else:
                    self.b2 = self.b1
            if self.pinch_ratio is None and self.b1 > 0:
                self.pinch_ratio = self.b2 / self.b1

        # 2. Fluid Thermophysical Properties via CoolProp if fluid name specified
        if self.fluid is not None and COOLPROP_AVAILABLE:
            try:
                # Diffusers operate in the gas/vapor phase
                try:
                    phase = PhaseSI('P', self.p1, 'T', self.T1, self.fluid)
                except Exception:
                    phase = 'gas'

                if phase in ['liquid', 'twophase']:
                    # Subcooled or two-phase condition provided for vapor compressor; evaluate at saturated vapor
                    cp_cp = float(PropsSI('CPMASS', 'Q', 1.0, 'T', self.T1, self.fluid))
                    cv_cp = float(PropsSI('CVMASS', 'Q', 1.0, 'T', self.T1, self.fluid))
                    rho_cp = float(PropsSI('D', 'Q', 1.0, 'T', self.T1, self.fluid))
                    mu_cp = float(PropsSI('V', 'Q', 1.0, 'T', self.T1, self.fluid))
                    a_cp = float(PropsSI('A', 'Q', 1.0, 'T', self.T1, self.fluid))
                    self.p1 = float(PropsSI('P', 'Q', 1.0, 'T', self.T1, self.fluid))
                else:
                    cp_cp = float(PropsSI('CPMASS', 'P', self.p1, 'T', self.T1, self.fluid))
                    cv_cp = float(PropsSI('CVMASS', 'P', self.p1, 'T', self.T1, self.fluid))
                    rho_cp = float(PropsSI('D', 'P', self.p1, 'T', self.T1, self.fluid))
                    mu_cp = float(PropsSI('V', 'P', self.p1, 'T', self.T1, self.fluid))
                    a_cp = float(PropsSI('A', 'P', self.p1, 'T', self.T1, self.fluid))

                gamma_cp = cp_cp / cv_cp
                molar_mass = float(PropsSI('MOLARMASS', self.fluid)) # kg/mol
                R_gas_cp = 8.31446261815324 / molar_mass

                self.rho = rho_cp
                self.gamma = gamma_cp
                self.cp = cp_cp
                self.mu = mu_cp
                self.nu = mu_cp / rho_cp
                self.a_sound = a_cp
                self.R_gas = R_gas_cp
            except Exception:
                # Fallback to ideal gas calculation if CoolProp fails
                pass

        if self.cp is None:
            self.cp = self.gamma * self.R_gas / (self.gamma - 1.0)
        if self.a_sound is None:
            self.a_sound = np.sqrt(self.gamma * self.R_gas * self.T1)
        if self.mu is None:
            self.mu = self.nu * self.rho

        # Resolve inlet velocity magnitude
        if self.V1_mean is None:
            if self.mach1 > 0.0 and self.a_sound is not None:
                self.V1_mean = self.mach1 * self.a_sound
            else:
                self.V1_mean = 50.0

        # 3. Angle synchronization (maintain both alpha and beta)
        # In Senoo (1977), flow angle was defined from tangential (beta, typically 5 - 35 deg).
        # In modern turbomachinery and the Word document, alpha is defined from radial (90 - beta).
        if self.beta1_mean is not None:
            # Explicit beta provided
            beta_val = self.beta1_mean
            alpha_rad_tangential = np.radians(beta_val)
            if self.Cr1 is None or self.Ct1 is None:
                self.Cr1 = self.V1_mean * np.sin(alpha_rad_tangential)
                self.Ct1 = self.V1_mean * np.cos(alpha_rad_tangential)
        elif self.alpha1_mean <= 45.0:
            # Caller passed tangential angle in alpha1_mean (Senoo 1977 convention)
            self.beta1_mean = self.alpha1_mean
            beta_val = self.alpha1_mean
            alpha_rad_tangential = np.radians(beta_val)
            if self.Cr1 is None or self.Ct1 is None:
                self.Cr1 = self.V1_mean * np.sin(alpha_rad_tangential)
                self.Ct1 = self.V1_mean * np.cos(alpha_rad_tangential)
        else:
            # Caller passed radial angle in alpha1_mean (radial convention)
            self.beta1_mean = 90.0 - self.alpha1_mean
            alpha_rad = np.radians(self.alpha1_mean)
            if self.Cr1 is None or self.Ct1 is None:
                self.Cr1 = self.V1_mean * np.cos(alpha_rad)
                self.Ct1 = self.V1_mean * np.sin(alpha_rad)

        # 4. Resolve velocities and mass flow
        if self.Cr1 is not None and self.Ct1 is not None:
            self.V1_mean = np.sqrt(self.Cr1**2 + self.Ct1**2)
        elif self.mdot is not None:
            area1 = 2.0 * np.pi * self.r1 * self.b1
            self.Cr1 = self.mdot / (self.rho * area1)
            # Use beta_mean to compute Ct1
            beta_rad = np.radians(self.beta1_mean)
            self.Ct1 = self.Cr1 / np.tan(beta_rad) if np.tan(beta_rad) > 1e-6 else self.V1_mean
            self.V1_mean = np.sqrt(self.Cr1**2 + self.Ct1**2)

        area1 = 2.0 * np.pi * self.r1 * self.b1
        self.mdot = self.rho * area1 * self.Cr1

        # 5. Mach number and compressibility
        if self.mach1 <= 0.0 and self.a_sound > 0.0:
            self.mach1 = self.V1_mean / self.a_sound

        if self.mach1 > 0.05:
            self.compressible = True

        # 6. Total temperature
        if self.T0 is None:
            self.T0 = self.T1 + 0.5 * self.V1_mean**2 / self.cp

        # 7. Boundary layer physical thicknesses
        if self.delta1_hub is None:
            self.delta1_hub = self.delta_b_hub * self.b1
        else:
            self.delta_b_hub = self.delta1_hub / self.b1

        if self.delta1_shroud is None:
            self.delta1_shroud = self.delta_b_shroud * self.b1
        else:
            self.delta_b_shroud = self.delta1_shroud / self.b1

    def get_width(self, r: float) -> Tuple[float, float]:
        """
        Returns local diffuser passage width b(r) [m] and derivative db/dr [-] at radius r.
        Supports:
        - Constant width (parallel walls)
        - Linear taper / pinched diffuser (b1 to b2)
        - Custom analytical callable b_func(r)
        - Discrete tabular CAD/contour coordinates (r_geom, b_geom)
        """
        return width_at_radius(
            r,
            self.r1,
            self.r2,
            self.b1,
            self.b2,
            b_func=self.b_func,
            db_dr_func=self.db_dr_func,
        )

    @classmethod
    def from_dimensional(
        cls,
        r1: float,
        r2: float,
        b1: float,
        alpha1_mean: float,
        V1_mean: Optional[float] = None,
        b2: Optional[float] = None,
        fluid: Optional[str] = None,
        b_func: Optional[Callable[[float], float]] = None,
        db_dr_func: Optional[Callable[[float], float]] = None,
        r_geom: Optional[Any] = None,
        b_geom: Optional[Any] = None,
        **kwargs
    ) -> 'SenooConfig':
        """
        Create configuration from dimensional parameters:
        - r1, r2: inlet and exit radii [m]
        - b1, b2: inlet and exit passage widths [m]
        - alpha1_mean: mean flow angle measured from radial direction [deg]
        - V1_mean: mean inlet velocity magnitude [m/s]
        - fluid: optional CoolProp fluid name (e.g. 'Air', 'Argon', 'R1233zd(E)')
        - b_func: optional custom function b(r) [m] for arbitrary variable width
        - db_dr_func: optional custom derivative function db/dr [-]
        - r_geom, b_geom: optional discrete radius and passage width coordinates for tabular contour
        """
        return cls(
            r1=r1,
            r2=r2,
            b1=b1,
            b2=b2,
            alpha1_mean=alpha1_mean,
            V1_mean=V1_mean,
            fluid=fluid,
            b_func=b_func,
            db_dr_func=db_dr_func,
            r_geom=r_geom,
            b_geom=b_geom,
            **kwargs
        )

    @classmethod
    def from_nondimensional(
        cls,
        b_over_r0: float,
        re_over_r0: float,
        Rb: float,
        alpha1_mean: float = 73.0,
        mach1: float = 0.0,
        distortion_ur: float = 0.0,
        distortion_utheta: float = 0.0,
        delta_b_hub: float = 0.05,
        delta_b_shroud: float = 0.05,
        pinch_ratio: float = 1.0,
        r0: float = 1.0,
        rho: float = 1.2,
        **kwargs
    ) -> 'SenooConfig':
        """
        Create configuration from non-dimensional ratios matching Senoo's papers:
        - b_over_r0: b1 / r1 (width-to-radius ratio)
        - re_over_r0: r2 / r1 (radius ratio)
        - Rb: Reynolds number V1 * b1 / nu
        - alpha1_mean: inlet mean flow angle [deg]
        - mach1: inlet Mach number
        - distortion_ur, distortion_utheta: velocity distortion across span
        - pinch_ratio: b2 / b1
        """
        r1 = r0
        r2 = r0 * re_over_r0
        b1 = r0 * b_over_r0
        b2 = b1 * pinch_ratio

        V1_mean = 50.0
        nu = (V1_mean * b1) / Rb if Rb > 0 else 1.5e-5
        compressible = mach1 > 0.05

        return cls(
            r1=r1,
            r2=r2,
            b1=b1,
            b2=b2,
            pinch_ratio=pinch_ratio,
            alpha1_mean=alpha1_mean,
            V1_mean=V1_mean,
            delta_b_hub=delta_b_hub,
            delta_b_shroud=delta_b_shroud,
            distortion_ur=distortion_ur,
            distortion_utheta=distortion_utheta,
            rho=rho,
            nu=nu,
            mach1=mach1,
            compressible=compressible,
            **kwargs
        )

    @classmethod
    def from_operating_point(
        cls,
        r1: float,
        r2: float,
        b1: float,
        mdot: float,
        rpm: float,
        slip_factor: float = 0.85,
        b2: Optional[float] = None,
        p1: float = 101325.0,
        T1: float = 288.15,
        fluid: str = 'Air',
        **kwargs
    ) -> 'SenooConfig':
        """
        Create configuration directly from compressor design and operating condition:
        - r1, r2: diffuser inlet and exit radii [m]
        - b1, b2: passage widths [m]
        - mdot: mass flow rate [kg/s]
        - rpm: impeller rotational speed [rpm]
        - slip_factor: Stanitz/Wiesner slip factor sigma = Ct1 / U2
        - p1, T1: inlet static conditions [Pa, K]
        - fluid: CoolProp fluid name (e.g. 'Air', 'Argon', 'R1233zd(E)')
        """
        omega = 2.0 * np.pi * rpm / 60.0
        U2 = omega * r1
        Ct1 = slip_factor * U2

        # Create temporary instance to resolve fluid density
        cfg_temp = cls(r1=r1, r2=r2, b1=b1, b2=b2, p1=p1, T1=T1, fluid=fluid)
        area1 = 2.0 * np.pi * r1 * b1
        Cr1 = mdot / (cfg_temp.rho * area1)
        V1 = np.sqrt(Cr1**2 + Ct1**2)
        alpha1 = float(np.degrees(np.arctan2(Ct1, Cr1)))

        return cls(
            r1=r1,
            r2=r2,
            b1=b1,
            b2=b2,
            p1=p1,
            T1=T1,
            fluid=fluid,
            mdot=mdot,
            rpm=rpm,
            omega=omega,
            Cr1=Cr1,
            Ct1=Ct1,
            V1_mean=V1,
            alpha1_mean=alpha1,
            **kwargs
        )

    @classmethod
    def from_ini(cls, config_file: str, section: str = 'SENOO') -> 'SenooConfig':
        """Load configuration from an INI file."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        parser = configparser.ConfigParser()
        parser.read(config_file)

        if not parser.has_section(section):
            raise ValueError(f"Section [{section}] not found in {config_file}")

        params: Dict[str, Any] = {}
        for key, val in parser.items(section):
            key_lower = key.lower()
            if val.lower() in ('true', 'yes', 'on'):
                params[key_lower] = True
            elif val.lower() in ('false', 'no', 'off'):
                params[key_lower] = False
            else:
                try:
                    params[key_lower] = int(val)
                except ValueError:
                    try:
                        params[key_lower] = float(val)
                    except ValueError:
                        params[key_lower] = val

        return cls(**params)

    def print_info(self):
        """Print formatted configuration summary banner."""
        print_banner('SENOO DIFFUSER CONFIGURATION')
        print(f"{'Inlet Radius r1 [m]:':<{total_chars_mid}}{self.r1:>{total_chars_mid}.4f}")
        print(f"{'Outlet Radius r2 [m]:':<{total_chars_mid}}{self.r2:>{total_chars_mid}.4f}")
        print(f"{'Radius Ratio r2/r1 [-]:':<{total_chars_mid}}{(self.r2/self.r1):>{total_chars_mid}.4f}")
        print(f"{'Inlet Width b1 [m]:':<{total_chars_mid}}{self.b1:>{total_chars_mid}.4f}")
        print(f"{'Outlet Width b2 [m]:':<{total_chars_mid}}{self.b2:>{total_chars_mid}.4f}")
        print(f"{'Width-to-Radius b1/r1 [-]:':<{total_chars_mid}}{(self.b1/self.r1):>{total_chars_mid}.4f}")
        print(f"{'Fluid:':<{total_chars_mid}}{str(self.fluid or 'Ideal Gas'):>{total_chars_mid}}")
        print(f"{'Specific Heat Ratio gamma [-]:':<{total_chars_mid}}{self.gamma:>{total_chars_mid}.4f}")
        print(f"{'Fluid Density rho [kg/m^3]:':<{total_chars_mid}}{self.rho:>{total_chars_mid}.4f}")
        print(f"{'Kinematic Viscosity nu [m^2/s]:':<{total_chars_mid}}{self.nu:>{total_chars_mid}.2e}")
        print(f"{'Operating Radial Angle alpha1 [deg]:':<{total_chars_mid}}{self.alpha1_mean:>{total_chars_mid}.2f}")
        print(f"{'Operating Tangential Angle beta1 [deg]:':<{total_chars_mid}}{self.beta1_mean:>{total_chars_mid}.2f}")
        print(f"{'Mean Velocity V1 [m/s]:':<{total_chars_mid}}{self.V1_mean:>{total_chars_mid}.2f}")
        print(f"{'Inlet Mach Number M1 [-]:':<{total_chars_mid}}{self.mach1:>{total_chars_mid}.3f}")
        print(f"{'Compressibility Correction:':<{total_chars_mid}}{str(self.compressible):>{total_chars_mid}}")
        print(f"{'Solver Mode:':<{total_chars_mid}}{self.solver_mode.upper():>{total_chars_mid}}")
        print(f"{'Surge Prediction Method:':<{total_chars_mid}}{self.surge_method:>{total_chars_mid}}")
        print_banner()
