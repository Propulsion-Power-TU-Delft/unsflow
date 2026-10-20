"""
HECC Vaneless Diffuser Operation with Air
=========================================
Operating conditions:
    - Geometry: b1 = b2 = 0.010 m, r1 = 0.216 m, r2 = 0.280 m (parallel walls)
    - Fluid: Air (gamma ~ 1.40, Molar mass ~ 28.96 g/mol)
    - Inlet Flow Angle: alpha1 = 75.0 deg (radial reference, beta1 = 15.0 deg)
    - Inlet Mach Number: M1 = 1.20 (V1 ~ 408.5 m/s)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from unsflow.senoo import SenooConfig, SenooDiffuser, SenooResults
from unsflow.utils.thesis_plots import *
set_thesis_style()

r = np.array([0.216, 0.224, 0.232, 0.241, 0.249, 0.259, 0.269, 0.280])
b = np.array([0.015, 0.014, 0.012, 0.011, 0.010, 0.010, 0.010, 0.010])
alpha_inlet = 70
M_inlet = 1.01
nr = 100
nspan = 50

def run_hecc_air():
    pic_dir = os.path.join(os.path.dirname(__file__), "pics")
    os.makedirs(pic_dir, exist_ok=True)

    print("=" * 80)
    print(" HECC VANELESS DIFFUSER OPERATION: AIR (M1 = 1.20, alpha1 = 75 deg)")
    print("=" * 80)

    cfg = SenooConfig.from_dimensional(
        r1 = r[0],
        r2 = r[-1],
        b1 = b[0],
        r_geom=r,
        b_geom=b,
        alpha1_mean=alpha_inlet,
        mach1=M_inlet,
        fluid='Air',
        compressible=True,
        nr=nr,
        nspan=nspan
    )

    diffuser = SenooDiffuser(cfg)
    diffuser.print_info()

    # Solve flow field and calculate stability limits
    field: SenooResults = diffuser.solve()
    field.print_summary()

    # Plot local flowfield distributions
    fig, axes = create_figure(1, 1.618, (2,2))

    # (a) Flow Angles: alpha and beta
    ax1 = axes[0, 0]
    ax1.plot(field.r * 1000, field.alpha_1d, 'b-', label=r"$\alpha(r)$")
    ax1.axhline(field.rotating_stall_angle, color='r', linestyle='--', label=f"$\\alpha_{{\\rm stall}} = {field.rotating_stall_angle:.1f}^\\circ$")
    ax1.axhline(field.surge_angle, color='darkred', linestyle=':', label=f"$\\alpha_{{\\rm surge}} = {field.surge_angle:.1f}^\\circ$")
    ax1.set_xlabel("$r$ [mm]")
    ax1.set_ylabel(r"$\alpha$ [deg]")
    ax1.grid(alpha=.3)
    ax1.legend(loc="best")

    # (b) Velocity components
    ax2 = axes[0, 1]
    ax2.plot(field.r * 1000, field.Ur_hub, 'b-',  label=r"$C_r$")
    ax2.plot(field.r * 1000, field.Ut_hub, 'r-',  label=r"$C_\theta$")
    ax2.set_xlabel(r"$r$ [mm]")
    ax2.set_ylabel(r"$C$ [m/s]")
    ax2.grid(alpha=.3)
    ax2.legend(loc="best")

    # (c) Static Pressure Recovery Cp and Mach Number
    ax3 = axes[1, 0]
    ax3.plot(field.r * 1000, field.Cp, 'g-')
    ax3.set_xlabel(r"$r$ [mm]")
    ax3.set_ylabel(r"$C_{\rm p}$", color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3.grid(alpha=.3)
    ax3_m = ax3.twinx()
    ax3_m.plot(field.r * 1000, field.mach_1d, 'k--')
    ax3_m.set_ylabel(r"$M$", color='k')
    ax3_m.tick_params(axis='y', labelcolor='k')


    # (d) Boundary Layer Thickness and Wall Skew
    ax4 = axes[1, 1]
    ax4.plot(field.r * 1000, field.delta_hub / field.b_r, 'b-',  label=r"$\delta_{\mathrm{hub}} / b$")
    ax4.plot(field.r * 1000, field.delta_shroud / field.b_r, 'r--',  label=r"$\delta_{\mathrm{shroud}} / b$")
    ax4.set_xlabel(r"$r$ [mm]")
    ax4.set_ylabel(r"$\delta / b$")
    ax4.grid(alpha=.3)
    ax4.legend(loc="best")

    # fig.suptitle(f"HECC Vaneless Diffuser - Air (M1 = {M_inlet}, alpha1 = {alpha_inlet} deg)", y=0.98)
    fpath = os.path.join(pic_dir, "hecc_air_fields.pdf")
    fig.savefig(fpath)
    print(f"Saved figure: {fpath}")

    return cfg, field


if __name__ == '__main__':
    run_hecc_air()
    plt.show()