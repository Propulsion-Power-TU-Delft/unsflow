from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from unsflow.utils.formatting import print_banner, total_chars, total_chars_mid


@dataclass
class SenooFlowField:
    """
    Container for the full flow field computed by the Senoo vaneless diffuser model.
    Contains both 2D mesh distributions across (radius, span) and 1D boundary layer
    distributions along radius.
    """
    # 1D Coordinates
    r: np.ndarray                      # Radial stations [m] (length Nr)
    span: np.ndarray                   # Normalized span y/b in [0, 1] (length Nspan)
    b_r: np.ndarray                    # Diffuser width b(r) [m] (length Nr)

    # 2D Mesh Arrays (Nr x Nspan)
    R: np.ndarray                      # 2D grid of radius [m]
    SPAN: np.ndarray                   # 2D grid of normalized span [-]
    Y: np.ndarray                      # 2D grid of physical spanwise position y [m]

    # 2D Flow Field Variables (Nr x Nspan)
    vr: np.ndarray                     # Radial velocity [m/s]
    vt: np.ndarray                     # Circumferential / tangential velocity [m/s]
    V: np.ndarray                      # Resultant velocity magnitude [m/s]
    alpha: np.ndarray                  # Flow angle [deg] (measured from tangential)
    rho: np.ndarray                    # Fluid density [kg/m^3]
    mach: Optional[np.ndarray] = None  # Mach number distribution (if compressible)

    # 1D Radial Distributions (length Nr)
    p: np.ndarray = field(default_factory=lambda: np.array([]))         # Static pressure [Pa]
    Cp: np.ndarray = field(default_factory=lambda: np.array([]))        # Static pressure coefficient (p - p1)/(0.5*rho*V1^2)
    delta_hub: np.ndarray = field(default_factory=lambda: np.array([])) # Hub boundary layer thickness [m]
    delta_shroud: np.ndarray = field(default_factory=lambda: np.array([])) # Shroud boundary layer thickness [m]
    e_hub: np.ndarray = field(default_factory=lambda: np.array([]))     # Hub crossflow parameter e1 = tan(gamma1)
    e_shroud: np.ndarray = field(default_factory=lambda: np.array([]))  # Shroud crossflow parameter e2 = tan(gamma2)
    Ur_hub: np.ndarray = field(default_factory=lambda: np.array([]))    # Core / boundary edge radial velocity at hub [m/s]
    Ut_hub: np.ndarray = field(default_factory=lambda: np.array([]))    # Core / boundary edge tangential velocity at hub [m/s]
    Ur_shroud: np.ndarray = field(default_factory=lambda: np.array([])) # Core / boundary edge radial velocity at shroud [m/s]
    Ut_shroud: np.ndarray = field(default_factory=lambda: np.array([])) # Core / boundary edge tangential velocity at shroud [m/s]
    has_reverse_flow_hub: np.ndarray = field(default_factory=lambda: np.array([]))    # Boolean array of hub reverse flow
    has_reverse_flow_shroud: np.ndarray = field(default_factory=lambda: np.array([])) # Boolean array of shroud reverse flow
    zone: np.ndarray = field(default_factory=lambda: np.array([]))      # Flow zone (0 = Entry Zone, 1 = Merged Downstream Zone)

    # Summary Metrics
    is_stalled: bool = False
    r_stall_hub: Optional[float] = None
    r_stall_shroud: Optional[float] = None
    r_merge: Optional[float] = None
    critical_flow_angle: Optional[float] = None

    def print_summary(self):
        """Print a formatted summary of diffuser flow results and stability."""
        print_banner('SENOO FLOW FIELD SUMMARY')
        print(f"{'Inlet Radius [m]:':<{total_chars_mid}}{self.r[0]:>{total_chars_mid}.4f}")
        print(f"{'Outlet Radius [m]:':<{total_chars_mid}}{self.r[-1]:>{total_chars_mid}.4f}")
        print(f"{'Inlet Mean Angle [deg]:':<{total_chars_mid}}{np.mean(self.alpha[0, :]):>{total_chars_mid}.2f}")
        print(f"{'Outlet Mean Angle [deg]:':<{total_chars_mid}}{np.mean(self.alpha[-1, :]):>{total_chars_mid}.2f}")
        print(f"{'Inlet Mean Velocity [m/s]:':<{total_chars_mid}}{np.mean(self.V[0, :]):>{total_chars_mid}.2f}")
        print(f"{'Outlet Mean Velocity [m/s]:':<{total_chars_mid}}{np.mean(self.V[-1, :]):>{total_chars_mid}.2f}")
        if len(self.p) > 0:
            print(f"{'Total Static Pressure Rise [Pa]:':<{total_chars_mid}}{(self.p[-1] - self.p[0]):>{total_chars_mid}.2f}")
            print(f"{'Outlet Cp [-]:':<{total_chars_mid}}{self.Cp[-1]:>{total_chars_mid}.4f}")
        if self.r_merge is not None:
            print(f"{'Boundary Layer Merge Radius [m]:':<{total_chars_mid}}{self.r_merge:>{total_chars_mid}.4f}")
        else:
            print(f"{'Boundary Layer Merged:':<{total_chars_mid}}{'No (Wide Diffuser)':>{total_chars_mid}}")
        print(f"{'Reverse Flow / Stall Detected:':<{total_chars_mid}}{str(self.is_stalled):>{total_chars_mid}}")
        if self.r_stall_hub is not None:
            print(f"{'Hub Reverse Flow Onset Radius [m]:':<{total_chars_mid}}{self.r_stall_hub:>{total_chars_mid}.4f}")
        if self.r_stall_shroud is not None:
            print(f"{'Shroud Reverse Flow Onset Radius [m]:':<{total_chars_mid}}{self.r_stall_shroud:>{total_chars_mid}.4f}")
        if self.critical_flow_angle is not None:
            print(f"{'Critical Flow Angle [deg]:':<{total_chars_mid}}{self.critical_flow_angle:>{total_chars_mid}.2f}")
        print_banner()

    def plot_flow_field(self, field_name: str = 'vr', cmap: str = 'viridis', ax: Optional[plt.Axes] = None):
        """
        Plot 2D contour map of flow field variable across (radius, normalized span).
        field_name: 'vr', 'vt', 'V', 'alpha', or 'rho'
        """
        data_dict = {
            'vr': (self.vr, r'Radial Velocity $v_r$ [m/s]'),
            'vt': (self.vt, r'Tangential Velocity $v_\theta$ [m/s]'),
            'V': (self.V, r'Velocity Magnitude $V$ [m/s]'),
            'alpha': (self.alpha, r'Flow Angle $\alpha$ [deg]'),
            'rho': (self.rho, r'Density $\rho$ [kg/m$^3$]')
        }
        if field_name not in data_dict:
            raise ValueError(f"Unknown field: {field_name}. Choose from {list(data_dict.keys())}")

        val, label = data_dict[field_name]
        show = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            show = True

        cf = ax.contourf(self.R, self.SPAN, val, levels=40, cmap=cmap)
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(label)
        ax.set_xlabel(r'Radius $r$ [m]')
        ax.set_ylabel(r'Normalized Span $y/b$ [-]')
        ax.set_title(f'Senoo Diffuser Flow Field: {label}')
        ax.grid(True, linestyle='--', alpha=0.3)

        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_profiles(self, r_stations: Optional[List[float]] = None):
        """
        Plot velocity profiles vr(y) and vt(y) at selected radial stations,
        matching Figures 7 and 8 from Senoo et al. (1977).
        """
        if r_stations is None:
            r_stations = [
                self.r[0],
                self.r[len(self.r)//4],
                self.r[len(self.r)//2],
                self.r[3*len(self.r)//4],
                self.r[-1]
            ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

        for r_target in r_stations:
            idx = int(np.argmin(np.abs(self.r - r_target)))
            r_actual = self.r[idx]
            lbl = f"r/r1 = {r_actual / self.r[0]:.2f}"
            ax1.plot(self.vr[idx, :], self.span, label=lbl)
            ax2.plot(self.vt[idx, :], self.span, label=lbl)

        ax1.axvline(0.0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel(r'$v_r$ [m/s]')
        ax1.set_ylabel(r'Normalized Span $y/b$')
        ax1.set_title('Radial Velocity Profiles')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(fontsize=8)

        ax2.set_xlabel(r'$v_\theta$ [m/s]')
        ax2.set_title('Tangential Velocity Profiles')
        ax2.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()

    def to_dict(self) -> Dict[str, Any]:
        """Convert flow field results to dictionary format."""
        return {
            'r': self.r,
            'span': self.span,
            'b_r': self.b_r,
            'vr': self.vr,
            'vt': self.vt,
            'V': self.V,
            'alpha': self.alpha,
            'rho': self.rho,
            'p': self.p,
            'Cp': self.Cp,
            'delta_hub': self.delta_hub,
            'delta_shroud': self.delta_shroud,
            'e_hub': self.e_hub,
            'e_shroud': self.e_shroud,
            'is_stalled': self.is_stalled,
            'r_stall_hub': self.r_stall_hub,
            'r_stall_shroud': self.r_stall_shroud,
            'r_merge': self.r_merge,
            'critical_flow_angle': self.critical_flow_angle
        }
