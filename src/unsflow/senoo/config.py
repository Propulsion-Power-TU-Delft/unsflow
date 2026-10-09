from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import configparser
import os
from unsflow.utils.formatting import print_banner, total_chars, total_chars_mid


@dataclass
class SenooConfig:
    """
    Configuration parameters for Senoo vaneless diffuser model.
    Supports both non-dimensional ratios and physical dimensional inputs,
    as well as compressible and incompressible flow regimes.
    """
    # Geometry
    r1: float = 1.0                    # Inlet radius [m]
    r2: float = 1.5                    # Outlet radius [m]
    b1: float = 0.05                   # Inlet passage width [m]
    b2: Optional[float] = None         # Outlet passage width [m] (defaults to b1 for parallel walls)
    pinch_ratio: Optional[float] = None # b2 / b1 (pinch ratio, optional alternative to b2)

    # Inflow conditions
    alpha1_mean: float = 15.0          # Inlet mean flow angle [deg] (measured from tangential/circumferential)
    V1_mean: float = 50.0              # Inlet mean velocity magnitude [m/s]
    delta1_hub: Optional[float] = None # Inlet boundary layer thickness at hub [m]
    delta1_shroud: Optional[float] = None # Inlet boundary layer thickness at shroud [m]
    delta_b_hub: float = 0.05          # Inlet hub boundary layer thickness ratio delta_1 / b1
    delta_b_shroud: float = 0.05       # Inlet shroud boundary layer thickness ratio delta_2 / b1
    e1_hub: float = 0.0                # Inlet wall crossflow skewness e = tan(gamma) at hub
    e1_shroud: float = 0.0             # Inlet wall crossflow skewness e = tan(gamma) at shroud
    distortion_ur: float = 0.0         # Radial velocity distortion Delta Ur / Ur_mean = (Ur2 - Ur1) / Ur_mean
    distortion_utheta: float = 0.0     # Tangential velocity distortion Delta Ut / Ut_mean = (Ut2 - Ut1) / Ut_mean

    # Fluid & Thermodynamics
    rho: float = 1.2                   # Fluid density [kg/m^3]
    nu: float = 1.5e-5                 # Fluid kinematic viscosity [m^2/s]
    p1: float = 101325.0               # Inlet static pressure [Pa]
    compressible: bool = False         # Whether to enable compressibility corrections
    mach1: float = 0.0                 # Inlet Mach number
    gamma: float = 1.4                 # Specific heat ratio (k)
    R_gas: float = 287.05              # Specific gas constant [J/(kg K)]
    T0: Optional[float] = None         # Total temperature [K]
    zeta: float = 0.7                  # Thermal wall condition factor (0.7 adiabatic, 0.0 isothermal)

    # Empirical & Boundary Layer Closure Parameters
    Kf: float = 0.045                  # Skin friction factor constant (0.045 Blasius, 0.051 Clauser)
    m: float = 3.0                     # Crossflow profile exponent (typically 3.0, or 2.0 for Mager)
    n_power: Optional[float] = None    # Streamwise velocity power law exponent (None -> 2.667 * R_delta^(1/8))
    tan_beta: Optional[float] = None   # Johnston outer hodograph parameter (None -> dynamic closure)

    # Numerical Solver Settings
    nr: int = 100                      # Number of radial stations in reconstructed flow field
    nspan: int = 50                    # Number of spanwise stations across diffuser passage
    dr_max: float = 0.002              # Maximum integration step size [m]
    rtol: float = 1e-5                 # Relative ODE tolerance
    atol: float = 1e-7                 # Absolute ODE tolerance

    def __post_init__(self):
        # Handle b2 and pinch_ratio
        if self.b2 is None:
            if self.pinch_ratio is not None:
                self.b2 = self.b1 * self.pinch_ratio
            else:
                self.b2 = self.b1
        elif self.pinch_ratio is None and self.b1 > 0:
            self.pinch_ratio = self.b2 / self.b1

        # Automatically enable compressibility if Mach number is substantial
        if self.mach1 > 0.1:
            self.compressible = True

        # Resolve boundary layer physical thicknesses from ratios if not explicitly given
        if self.delta1_hub is None:
            self.delta1_hub = self.delta_b_hub * self.b1
        else:
            self.delta_b_hub = self.delta1_hub / self.b1

        if self.delta1_shroud is None:
            self.delta1_shroud = self.delta_b_shroud * self.b1
        else:
            self.delta_b_shroud = self.delta1_shroud / self.b1

        # Total temperature
        if self.T0 is None:
            # Estimate static temperature from ideal gas law
            T1 = self.p1 / (self.rho * self.R_gas)
            cp = self.gamma * self.R_gas / (self.gamma - 1.0)
            self.T0 = T1 + 0.5 * self.V1_mean**2 / cp

    @classmethod
    def from_nondimensional(
        cls,
        b_over_r0: float,
        re_over_r0: float,
        Rb: float,
        alpha1_mean: float = 15.0,
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
        Create configuration from non-dimensional parameters matching Senoo's papers:
        - b_over_r0: b1 / r0 (width-to-radius ratio)
        - re_over_r0: r2 / r0 (radius ratio)
        - Rb: Reynolds number U * b / nu
        - alpha1_mean: Inlet mean flow angle [deg]
        - mach1: Inlet Mach number
        - distortion_ur: Delta Ur / Ur_mean
        - distortion_utheta: Delta Ut / Ut_mean
        - delta_b_hub, delta_b_shroud: delta / b1
        - pinch_ratio: b2 / b1
        """
        r1 = r0
        r2 = r0 * re_over_r0
        b1 = r0 * b_over_r0
        b2 = b1 * pinch_ratio

        # Pick reference velocity and viscosity consistent with Rb = V1 * b1 / nu
        V1_mean = 50.0
        nu = (V1_mean * b1) / Rb if Rb > 0 else 1.5e-5

        compressible = mach1 > 0.1

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
    def from_dimensional(
        cls,
        r1: float,
        r2: float,
        b1: float,
        alpha1_mean: float,
        V1_mean: float = 50.0,
        b2: Optional[float] = None,
        **kwargs
    ) -> 'SenooConfig':
        """Create configuration from dimensional parameters."""
        return cls(
            r1=r1,
            r2=r2,
            b1=b1,
            b2=b2,
            alpha1_mean=alpha1_mean,
            V1_mean=V1_mean,
            **kwargs
        )

    @classmethod
    def from_ini(cls, config_file: str, section: str = 'SENOO') -> 'SenooConfig':
        """Load configuration from an INI file."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        parser = configparser.ConfigParser()
        parser.read(config_file)

        params: Dict[str, Any] = {}
        if not parser.has_section(section):
            raise ValueError(f"Section [{section}] not found in {config_file}")

        for key, val in parser.items(section):
            key_lower = key.lower()
            # Try parsing boolean, float, int, or string
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
        """Print formatted configuration banner."""
        print_banner('SENOO DIFFUSER CONFIGURATION')
        print(f"{'Inlet Radius r1 [m]:':<{total_chars_mid}}{self.r1:>{total_chars_mid}.4f}")
        print(f"{'Outlet Radius r2 [m]:':<{total_chars_mid}}{self.r2:>{total_chars_mid}.4f}")
        print(f"{'Radius Ratio r2/r1 [-]:':<{total_chars_mid}}{(self.r2/self.r1):>{total_chars_mid}.4f}")
        print(f"{'Inlet Width b1 [m]:':<{total_chars_mid}}{self.b1:>{total_chars_mid}.4f}")
        print(f"{'Width-to-Radius b1/r1 [-]:':<{total_chars_mid}}{(self.b1/self.r1):>{total_chars_mid}.4f}")
        print(f"{'Pinch Ratio b2/b1 [-]:':<{total_chars_mid}}{(self.b2/self.b1):>{total_chars_mid}.4f}")
        print(f"{'Mean Flow Angle alpha1 [deg]:':<{total_chars_mid}}{self.alpha1_mean:>{total_chars_mid}.2f}")
        print(f"{'Mean Velocity V1 [m/s]:':<{total_chars_mid}}{self.V1_mean:>{total_chars_mid}.2f}")
        print(f"{'Inlet Mach Number M1 [-]:':<{total_chars_mid}}{self.mach1:>{total_chars_mid}.3f}")
        print(f"{'Compressible Mode:':<{total_chars_mid}}{str(self.compressible):>{total_chars_mid}}")
        print(f"{'Reynolds Number Rb [-]:':<{total_chars_mid}}{(self.V1_mean*self.b1/self.nu):>{total_chars_mid}.2e}")
        print(f"{'Hub BL Ratio delta_1/b1 [-]:':<{total_chars_mid}}{self.delta_b_hub:>{total_chars_mid}.4f}")
        print(f"{'Shroud BL Ratio delta_2/b1 [-]:':<{total_chars_mid}}{self.delta_b_shroud:>{total_chars_mid}.4f}")
        print(f"{'Radial Distortion dUr/Ur [-]:':<{total_chars_mid}}{self.distortion_ur:>{total_chars_mid}.4f}")
        print(f"{'Tangential Distortion dUt/Ut [-]:':<{total_chars_mid}}{self.distortion_utheta:>{total_chars_mid}.4f}")
        print_banner()
