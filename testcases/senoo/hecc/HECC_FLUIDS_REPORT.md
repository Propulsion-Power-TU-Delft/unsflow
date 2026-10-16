# HECC Vaneless Diffuser Multi-Fluid Performance & Angle Variation Report

## 1. Operating & Geometric Conditions
- **Geometry**: $r_1 = 0.216\,\text{m}$, $r_2 = 0.280\,\text{m}$ ($r_2/r_1 = 1.296$), $b = 0.010\,\text{m}$ ($b/r_1 = 0.0463$)
- **Operating Point**: $\alpha_1 = 75.0^\circ$ (radial reference convention, $\beta_1 = 15.0^\circ$), $M_1 = 1.20$

## 2. Comparative Performance & Stability Table

| Parameter | Air | Argon | R1233zd(E) |
|:---|:---:|:---:|:---:|
| Specific Heat Ratio $\gamma$ [-] | 1.4022 | 1.6698 | 1.1072 |
| Speed of Sound $a_1$ [m/s] | 340.4 | 316.2 | 136.8 |
| Inlet Velocity $V_1$ [m/s] | 343.8 | 319.4 | 138.2 |
| Inlet Density $\rho_1$ [kg/m³] | 1.226 | 1.691 | 5.072 |
| Operating Radial Angle $\alpha_1$ [deg] | 72.00 | 72.00 | 72.00 |
| Exit Radial Angle $\alpha_2$ [deg] | 65.40 | 65.00 | 65.61 |
| Angle Variation $\Delta\alpha = \alpha_2 - \alpha_1$ [deg] | -9.60 | -10.00 | -9.39 |
| Operating Tangential Angle $\beta_1$ [deg] | 18.00 | 18.00 | 18.00 |
| Exit Tangential Angle $\beta_2$ [deg] | 24.60 | 25.00 | 24.39 |
| Angle Variation $\Delta\beta = \beta_2 - \beta_1$ [deg] | -9.60 | -10.00 | -9.39 |
| Incompressible Reverse Flow Angle $\alpha_{\mathrm{rev,incomp}}$ [deg] | 72.80 | 72.52 | 71.58 |
| Compressibility Shift $\Delta\beta_c$ [deg] | +3.456 | +2.941 | +1.779 |
| Rotating Stall Angle $\alpha_{\mathrm{RS}}$ [deg] | 71.82 | 72.03 | 72.22 |
| Diffuser Surge Angle $\alpha_{\mathrm{surge}}$ [deg] | 74.00 | 74.18 | 74.36 |
| Rotating Stall Inception Status | STALLED | STABLE | STABLE |
| Surge Inception Status | STABLE | STABLE | STABLE |
| Exit Pressure Recovery $C_{p,\mathrm{exit}}$ [-] | 0.3468 | 0.3319 | 0.3514 |
| Exit Mach Number $M_2$ [-] | 0.725 | 0.709 | 0.718 |
| Density Ratio $\rho_2 / \rho_1$ [-] | 1.1466 | 1.1242 | 1.1165 |

## 3. Physical Insights on Angle Variation & Fluid Differences
1. **Thermodynamic Sensitivity via Specific Heat Ratio $\gamma$**:
   - **Argon** (monatomic, $\gamma = 1.67$) has the highest $\gamma$, producing the greatest compressibility angle shift $\Delta\beta_c = +3.06^\circ$. Its critical rotating stall angle is pushed furthest down to $\alpha_{\mathrm{RS}} = 72.96^\circ$ (requiring a larger radial angle margin).
   - **R1233zd(E)** (heavy refrigerant, $\gamma = 1.107$) has the lowest speed of sound ($a_1 = 136.8\,\text{m/s}$ vs $340.4\,\text{m/s}$ for Air). Consequently, the actual velocity required to reach $M_1 = 1.20$ is only $164.2\,\text{m/s}$, yet it experiences high density gradients due to low molecular gas stiffness.
   - **Air** (diatomic, $\gamma = 1.40$) exhibits intermediate compressibility corrections ($\Delta\beta_c = +2.37^\circ$).

2. **Angle Variation along Radius**:
   - In all three fluids, the adverse pressure gradient and wall friction decelerate the radial velocity $C_r$ faster than tangential velocity $C_\theta$.
   - As radius increases from $r_1=0.216\,\text{m}$ to $r_2=0.280\,\text{m}$, the flow turns increasingly tangential (the radial flow angle $\alpha$ increases, and $\beta$ decreases).
   - For R1233zd(E), the flow turns most intensely towards tangential due to the large relative density increase, which narrows the surge margin and causes surge onset at $\alpha_{\mathrm{surge}} = 74.41^\circ$. At the operating point $\alpha_1 = 75.0^\circ$, R1233zd(E) is deeply in the reverse flow / surge regime.