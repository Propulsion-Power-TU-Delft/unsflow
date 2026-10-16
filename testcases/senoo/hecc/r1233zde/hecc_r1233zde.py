"""
HECC Vaneless Diffuser Operation with R1233zd(E)
================================================
Operating conditions:
    - Geometry: b1 = b2 = 0.010 m, r1 = 0.216 m, r2 = 0.280 m (parallel walls)
    - Fluid: R1233zd(E) (gamma ~ 1.10, Molar mass ~ 130.5 g/mol, heavy refrigerant)
    - Inlet Flow Angle: alpha1 = 75.0 deg (radial reference, beta1 = 15.0 deg)
    - Inlet Mach Number: M1 = 1.20 (V1 ~ 164.2 m/s)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from unsflow.senoo import SenooConfig, SenooDiffuser, SenooResults


def run_hecc_r1233zde():
    pic_dir = os.path.join(os.path.dirname(__file__), "pics")
    os.makedirs(pic_dir, exist_ok=True)

    print("=" * 80)
    print(" HECC VANELESS DIFFUSER OPERATION: R1233zd(E) (M1 = 1.20, alpha1 = 75 deg)")
    print("=" * 80)

    cfg = SenooConfig.from_dimensional(
        r1=0.216,
        r2=0.280,
        b1=0.010,
        b2=0.010,
        alpha1_mean=75.0,
        mach1=1.20,
        fluid='R1233zd(E)',
        compressible=True,
        nr=60,
        nspan=30
    )

    diffuser = SenooDiffuser(cfg)
    diffuser.print_info()

    # Solve flow field and calculate stability limits
    field: SenooResults = diffuser.solve()
    field.print_summary()

    # Plot local flowfield distributions
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # (a) Flow Angles: alpha and beta
    ax1 = axes[0, 0]
    ax1.plot(field.r * 1000, field.alpha_1d, 'b-', lw=2.2, label=r"Radial Angle $\alpha(r)$")
    ax1.axhline(field.rotating_stall_angle, color='r', linestyle='--', label=f"Rotating Stall $\\alpha_{{RS}} = {field.rotating_stall_angle:.1f}^\\circ$")
    ax1.axhline(field.surge_angle, color='darkred', linestyle=':', label=f"Surge $\\alpha_{{surge}} = {field.surge_angle:.1f}^\\circ$")
    ax1.set_xlabel("Radius $r$ [mm]", fontsize=11)
    ax1.set_ylabel(r"Flow Angle $\alpha$ [deg]", fontsize=11)
    ax1.set_title(r"(a) Flow Angle Distribution $\alpha(r)$", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", fontsize=9)

    # (b) Velocity components
    ax2 = axes[0, 1]
    ax2.plot(field.r * 1000, field.Ur_hub, 'b-', lw=2.0, label=r"Hub Core $C_r$")
    ax2.plot(field.r * 1000, field.Ut_hub, 'r-', lw=2.0, label=r"Hub Core $C_\theta$")
    ax2.set_xlabel("Radius $r$ [mm]", fontsize=11)
    ax2.set_ylabel("Velocity [m/s]", fontsize=11)
    ax2.set_title("(b) Core Velocity Components", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", fontsize=9)

    # (c) Static Pressure Recovery Cp and Mach Number
    ax3 = axes[1, 0]
    ax3.plot(field.r * 1000, field.Cp, 'g-', lw=2.2, label=r"Pressure Recovery $C_p$")
    ax3.set_xlabel("Radius $r$ [mm]", fontsize=11)
    ax3.set_ylabel(r"Static Pressure Recovery $C_p$ [-]", color='g', fontsize=11)
    ax3.tick_params(axis='y', labelcolor='g')
    ax3.grid(True, linestyle=":", alpha=0.6)

    ax3_m = ax3.twinx()
    ax3_m.plot(field.r * 1000, field.mach_1d, 'k--', lw=1.8, label="Mach Number $M$")
    ax3_m.set_ylabel("Mach Number $M$ [-]", color='k', fontsize=11)
    ax3_m.tick_params(axis='y', labelcolor='k')
    ax3.set_title(r"(c) Pressure Recovery $C_p$ and Mach Number $M$", fontsize=12)

    # (d) Boundary Layer Thickness and Wall Skew
    ax4 = axes[1, 1]
    ax4.plot(field.r * 1000, field.delta_hub / field.b_r, 'b-', lw=2.0, label=r"$\delta_{\mathrm{hub}} / b$")
    ax4.plot(field.r * 1000, field.delta_shroud / field.b_r, 'r--', lw=2.0, label=r"$\delta_{\mathrm{shroud}} / b$")
    ax4.set_xlabel("Radius $r$ [mm]", fontsize=11)
    ax4.set_ylabel(r"Boundary Layer Ratio $\delta / b$ [-]", fontsize=11)
    ax4.set_title("(d) Boundary Layer Growth Across Radius", fontsize=12)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="best", fontsize=9)

    fig.suptitle(f"HECC Vaneless Diffuser - R1233zd(E) (M1 = 1.20, alpha1 = 75 deg)", fontsize=14, y=0.98)
    fig.tight_layout()
    fpath = os.path.join(pic_dir, "hecc_r1233zde_fields.png")
    fig.savefig(fpath, dpi=200)
    print(f"Saved figure: {fpath}")

    return cfg, field


if __name__ == '__main__':
    run_hecc_r1233zde()
