"""
Replication of Senoo, Kinoshita & Ishida (1977) Experimental Validation
========================================================================

Reference:
    "Asymmetric Flow in Vaneless Diffusers of Centrifugal Blowers"
    Y. Senoo, Y. Kinoshita, M. Ishida
    ASME Journal of Fluids Engineering, Vol. 99, Issue 1, 1977.

Experimental Rig & Operating Conditions:
    - Impeller exit / diffuser inlet diameter: 502 mm  => r1 = 0.251 m
    - Diffuser exit diameter:                 1000 mm => r2 = 0.502 m  (r2/r1 = 2.0)
    - Diffuser passage width (parallel walls): 14 mm  => b1 = 0.014 m  (b/r1 = 0.0558)
    - Mean inlet velocity magnitude:          V1 ~ 45 m/s
    - Fluid:                                  Air (rho = 1.2 kg/m^3, nu = 1.5e-5 m^2/s)
    - Reynolds number:                        Rb = V1 * b / nu ~ 4.2e4

Cases Analyzed:
    1. Experiment 1 (Normal / Unstalled flow):
       - Flow coefficient phi = 0.21, inlet mean flow angle alpha1 = 17 deg.
       - Replicates Fig. 7 (velocity profiles) & Fig. 9 (Cp, flow angle, and wall angle).
    2. Experiment 2 (Low flow / Reverse flow stall):
       - Flow coefficient phi = 0.11, inlet mean flow angle alpha1 = 8 deg.
       - Replicates Fig. 8 (reverse flow at walls) & Fig. 10.
    3. Critical flow angle search:
       - Uses bisection search to determine the critical angle alpha_cr where
         radial wall reverse flow first initiates.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from unsflow.senoo import SenooConfig, SenooDiffuser, SenooFlowField


def run_senoo_replication():
    # Output directory for saved plots
    pic_dir = os.path.join(os.path.dirname(__file__), "pics")
    os.makedirs(pic_dir, exist_ok=True)

    print("=" * 80)
    print(" REPLICATING SENOO ET AL. (1977) VANELESS DIFFUSER EXPERIMENT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # CASE 1: Experiment 1 - Normal / Unstalled Flow (alpha = 17 deg)
    # -------------------------------------------------------------------------
    print("\n[1/3] Running Case 1: Unstalled Operating Point (alpha1 = 17.0 deg)...")

    cfg_exp1 = SenooConfig.from_dimensional(
        r1=0.251,           # Inlet radius [m]
        r2=0.502,           # Outlet radius [m]
        b1=0.014,           # Passage width [m]
        alpha1_mean=17.0,   # Mean flow angle [deg]
        V1_mean=45.0,       # Mean velocity magnitude [m/s]
        rho=1.2,            # Air density [kg/m^3]
        nu=1.5e-5,          # Kinematic viscosity [m^2/s]
        nr=80,              # Radial grid resolution
        nspan=40            # Spanwise grid resolution
    )

    diffuser1 = SenooDiffuser(cfg_exp1)
    diffuser1.print_info()

    field1: SenooFlowField = diffuser1.solve()
    field1.print_summary()

    # -------------------------------------------------------------------------
    # CASE 2: Experiment 2 - Stalled / Reverse Flow (alpha = 8 deg)
    # -------------------------------------------------------------------------
    print("\n[2/3] Running Case 2: Low-Flow Stall / Reverse Flow (alpha1 = 8.0 deg)...")

    cfg_exp2 = SenooConfig.from_dimensional(
        r1=0.251,
        r2=0.502,
        b1=0.014,
        alpha1_mean=8.0,    # Low flow angle -> reverse flow onset
        V1_mean=45.0,
        rho=1.2,
        nu=1.5e-5,
        nr=80,
        nspan=40
    )

    diffuser2 = SenooDiffuser(cfg_exp2)
    field2: SenooFlowField = diffuser2.solve()
    field2.print_summary()

    # -------------------------------------------------------------------------
    # CASE 3: Critical Flow Angle Search
    # -------------------------------------------------------------------------
    print("\n[3/3] Calculating Critical Flow Angle for Reverse Flow Onset...")
    alpha_cr = diffuser1.find_critical_flow_angle(tol=0.1, max_iter=20)
    print(f"  >>> Critical Flow Angle alpha_cr = {alpha_cr:.2f} deg")

    # -------------------------------------------------------------------------
    # PLOTTING: Replicate Reference Paper Figures
    # -------------------------------------------------------------------------
    print("\nGenerating publication-quality figures matching Senoo (1977)...")

    # Target radial stations matching Figure 7 & Figure 8: r/r0 = [1.014, 1.1, 1.2, 1.3, 1.6]
    r0 = cfg_exp1.r1
    radius_ratios = [1.014, 1.10, 1.20, 1.30, 1.60]
    target_radii = [rr * r0 for rr in radius_ratios]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # -------------------------------------------------------------------------
    # FIGURE 1: Velocity Profiles across Span (Replication of Figure 7)
    # -------------------------------------------------------------------------
    fig1, (ax_vr, ax_vt) = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)

    for rr, r_target, color in zip(radius_ratios, target_radii, colors):
        idx = int(np.argmin(np.abs(field1.r - r_target)))
        r_actual = field1.r[idx]
        lbl = f"$r/r_0 = {r_actual / r0:.3f}$"

        ax_vr.plot(field1.vr[idx, :], field1.span, label=lbl, color=color, linewidth=2.0)
        ax_vt.plot(field1.vt[idx, :], field1.span, label=lbl, color=color, linewidth=2.0)

    ax_vr.axvline(0.0, color='black', linestyle='--', alpha=0.6, linewidth=1.0)
    ax_vr.set_xlabel(r"Radial Velocity $v_r$ [m/s]", fontsize=12)
    ax_vr.set_ylabel(r"Normalized Span $y/b$ (Hub to Shroud)", fontsize=12)
    ax_vr.set_title(r"(a) Radial Velocity Profiles $v_r(y)$", fontsize=13)
    ax_vr.grid(True, linestyle=":", alpha=0.5)
    ax_vr.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax_vr.set_ylim([0.0, 1.0])

    ax_vt.set_xlabel(r"Tangential Velocity $v_\theta$ [m/s]", fontsize=12)
    ax_vt.set_title(r"(b) Tangential Velocity Profiles $v_\theta(y)$", fontsize=13)
    ax_vt.grid(True, linestyle=":", alpha=0.5)
    ax_vt.legend(loc="upper right", framealpha=0.9, fontsize=9)

    fig1.suptitle(r"Replication of Senoo et al. (1977) - Figure 7 ($\alpha_1 = 17^\circ$, $\phi = 0.21$)", fontsize=14, y=0.98)
    fig1.tight_layout()
    f1_path = os.path.join(pic_dir, "senoo_fig7_velocity_profiles.png")
    fig1.savefig(f1_path, dpi=200)
    print(f"  Saved Figure 7 replication: {f1_path}")

    # -------------------------------------------------------------------------
    # FIGURE 2: Radial Distributions of Cp, BL Thickness, and Flow Angles (Fig. 9)
    # -------------------------------------------------------------------------
    fig2, (ax_cp, ax_ang) = plt.subplots(1, 2, figsize=(11, 4.8))
    r_over_r0 = field1.r / r0

    # (a) Pressure Recovery Cp and BL Thickness
    ax_cp.plot(r_over_r0, field1.Cp, 'b-', linewidth=2.2, label=r"Pressure Recovery $C_p$ (Senoo Model)")
    ax_cp.set_xlabel(r"Radius Ratio $r/r_0$ [-]", fontsize=12)
    ax_cp.set_ylabel(r"Static Pressure Recovery $C_p$ [-]", color='b', fontsize=12)
    ax_cp.tick_params(axis='y', labelcolor='b')
    ax_cp.set_ylim([0.0, 0.7])
    ax_cp.grid(True, linestyle=":", alpha=0.5)

    ax_bl = ax_cp.twinx()
    ax_bl.plot(r_over_r0, field1.delta_hub / field1.b_r, 'r--', linewidth=1.8, label=r"$\delta_1 / b$ (Hub BL)")
    ax_bl.plot(r_over_r0, field1.delta_shroud / field1.b_r, 'm:', linewidth=1.8, label=r"$\delta_2 / b$ (Shroud BL)")
    ax_bl.axvline(field1.r_merge / r0, color='gray', linestyle='-.', alpha=0.7, label=f"BL Merge ($r/r_0 = {field1.r_merge/r0:.2f}$)")
    ax_bl.set_ylabel(r"Boundary Layer Ratio $\delta/b$ [-]", color='r', fontsize=12)
    ax_bl.tick_params(axis='y', labelcolor='r')
    ax_bl.set_ylim([0.0, 0.6])

    # Combine legends for ax_cp and ax_bl
    lines1, labels1 = ax_cp.get_legend_handles_labels()
    lines2, labels2 = ax_bl.get_legend_handles_labels()
    ax_cp.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8.5, framealpha=0.9)
    ax_cp.set_title(r"(a) Pressure Recovery $C_p$ & Layer Merging", fontsize=13)

    # (b) Flow Angle alpha and Wall Streamline Angle (alpha - gamma)
    alpha_core = np.degrees(np.arctan2(field1.vr[:, len(field1.span)//2], field1.vt[:, len(field1.span)//2]))
    # Wall streamline angle alpha - gamma = arctan(1/cot(alpha - gamma)) = arctan(tan(alpha) - e)
    gamma_hub = np.degrees(np.arctan(field1.e_hub))
    wall_angle_hub = alpha_core - gamma_hub

    ax_ang.plot(r_over_r0, alpha_core, 'k-', linewidth=2.0, label=r"Core Flow Angle $\alpha$ [deg]")
    ax_ang.plot(r_over_r0, wall_angle_hub, 'r--', linewidth=2.0, label=r"Wall Streamline Angle $\alpha - \gamma$ [deg]")
    ax_ang.axhline(0.0, color='red', linestyle=':', alpha=0.6, label="Reverse Flow Limit (0 deg)")
    ax_ang.set_xlabel(r"Radius Ratio $r/r_0$ [-]", fontsize=12)
    ax_ang.set_ylabel(r"Angle [deg]", fontsize=12)
    ax_ang.set_title(r"(b) Flow Angle & Wall Streamline Angle", fontsize=13)
    ax_ang.set_ylim([-5.0, 35.0])
    ax_ang.grid(True, linestyle=":", alpha=0.5)
    ax_ang.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    fig2.suptitle(r"Replication of Senoo et al. (1977) - Figure 9 ($\alpha_1 = 17^\circ$)", fontsize=14, y=0.98)
    fig2.tight_layout()
    f2_path = os.path.join(pic_dir, "senoo_fig9_pressure_and_angles.png")
    fig2.savefig(f2_path, dpi=200)
    print(f"  Saved Figure 9 replication: {f2_path}")

    # -------------------------------------------------------------------------
    # FIGURE 3: Reverse Flow Stall Profiles (Replication of Figure 8)
    # -------------------------------------------------------------------------
    fig3, (ax_st1, ax_st2) = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)

    # Plot unstalled Case 1 vs stalled Case 2 at r/r0 = 1.10 and 1.30
    for r_t, color in zip([1.10 * r0, 1.30 * r0], ['#1f77b4', '#d62728']):
        idx1 = int(np.argmin(np.abs(field1.r - r_t)))
        idx2 = int(np.argmin(np.abs(field2.r - r_t)))
        r_lbl = f"$r/r_0 = {field1.r[idx1]/r0:.2f}$"

        ax_st1.plot(field1.vr[idx1, :], field1.span, color=color, linewidth=2.0, label=f"Unstalled 17 deg ({r_lbl})")
        ax_st2.plot(field2.vr[idx2, :], field2.span, color=color, linewidth=2.0, label=f"Stalled 8 deg ({r_lbl})")

    ax_st1.axvline(0.0, color='black', linestyle='--', alpha=0.6)
    ax_st1.set_xlabel(r"Radial Velocity $v_r$ [m/s]", fontsize=12)
    ax_st1.set_ylabel(r"Normalized Span $y/b$", fontsize=12)
    ax_st1.set_title(r"Normal Flow: $\alpha_1 = 17^\circ$ (No Reverse Flow)", fontsize=12)
    ax_st1.grid(True, linestyle=":", alpha=0.5)
    ax_st1.legend(loc="upper right", fontsize=9)
    ax_st1.set_ylim([0.0, 1.0])

    ax_st2.axvline(0.0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label="Zero Radial Velocity")
    # Highlight reverse flow region in red
    ax_st2.axvspan(-6.0, 0.0, color='red', alpha=0.15, label="Reverse Flow Zone ($v_r < 0$)")
    ax_st2.set_xlabel(r"Radial Velocity $v_r$ [m/s]", fontsize=12)
    ax_st2.set_title(r"Stalled Flow: $\alpha_1 = 8^\circ$ (Reverse Flow at Wall)", fontsize=12)
    ax_st2.grid(True, linestyle=":", alpha=0.5)
    ax_st2.legend(loc="lower right", fontsize=9)

    fig3.suptitle(r"Replication of Senoo et al. (1977) - Figure 8 (Reverse Flow Stall Comparison)", fontsize=14, y=0.98)
    fig3.tight_layout()
    f3_path = os.path.join(pic_dir, "senoo_fig8_reverse_flow_comparison.png")
    fig3.savefig(f3_path, dpi=200)
    print(f"  Saved Figure 8 replication: {f3_path}")

    # -------------------------------------------------------------------------
    # FIGURE 4: Complete 2D Radial Velocity Contour Field (vr & vt)
    # -------------------------------------------------------------------------
    fig4, (ax_c1, ax_c2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

    cf1 = ax_c1.contourf(field1.R / r0, field1.SPAN, field1.vr, levels=35, cmap='viridis')
    cbar1 = plt.colorbar(cf1, ax=ax_c1)
    cbar1.set_label(r"$v_r$ [m/s]", fontsize=11)
    ax_c1.set_ylabel(r"Span $y/b$ [-]", fontsize=11)
    ax_c1.set_title(r"2D Radial Velocity Field $v_r(r, y)$ across Diffuser", fontsize=12)

    cf2 = ax_c2.contourf(field1.R / r0, field1.SPAN, field1.vt, levels=35, cmap='plasma')
    cbar2 = plt.colorbar(cf2, ax=ax_c2)
    cbar2.set_label(r"$v_\theta$ [m/s]", fontsize=11)
    ax_c2.set_xlabel(r"Radius Ratio $r/r_0$ [-]", fontsize=11)
    ax_c2.set_ylabel(r"Span $y/b$ [-]", fontsize=11)
    ax_c2.set_title(r"2D Tangential Velocity Field $v_\theta(r, y)$ across Diffuser", fontsize=12)

    fig4.suptitle(r"Full 2D Field Reconstruction ($\alpha_1 = 17^\circ$, $r_2/r_1 = 2.0$)", fontsize=14, y=0.98)
    fig4.tight_layout()
    f4_path = os.path.join(pic_dir, "senoo_2d_contour_fields.png")
    fig4.savefig(f4_path, dpi=200)
    print(f"  Saved 2D contour field: {f4_path}")

    print("\nAll replications successfully completed and saved to 'pics/'!")
    return field1, field2


if __name__ == "__main__":
    run_senoo_replication()
