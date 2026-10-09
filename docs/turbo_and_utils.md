# `unsflow.turbo` & `unsflow.utils`: Aerothermodynamics and Plotting Utilities

This document details the supporting modules `unsflow.turbo` and `unsflow.utils`, providing thermodynamic performance calculations and publication-quality visualization tools.

---

## 1. `unsflow.turbo`: Turbomachinery Aerothermodynamics

The `unsflow.turbo` module implements core nondimensional aerothermodynamic performance formulas for turbomachinery stages, supporting both ideal gas formulations and real gas thermophysical properties via **CoolProp** [1].

Implemented in [`src/unsflow/turbo/turbo_performance_functions.py`](../src/unsflow/turbo/turbo_performance_functions.py).

### 1.1 Available Functions

| Function | Definition / Equation | Inputs |
|---|---|---|
| `compute_flow_coefficient` | $\Phi = \frac{\dot{m}}{\rho D_{ref}^2 U_{ref}}$ | `mflow`, `density`, `r_ref`, `omega` or `rpm` |
| `compute_work_coefficient_temperature_ratio` | $\Psi = \frac{c_p (T_{t2}/T_{t1} - 1) T_{t1}}{U_{ref}^2}$ | `tratio`, `r_ref`, `omega` or `rpm`, `cp`, `Tt1` |
| `compute_work_coefficient_temperature_ratio_real` | $\Psi = \frac{\Delta h_t}{U_{ref}^2}$ | `delta_ht` (real enthalpy rise), `r_ref`, `omega` or `rpm` |
| `compute_tip_mach_number` | $M_{tip} = \frac{U_{ref}}{\sqrt{\gamma R T_{t1}}}$ | `omega` or `rpm`, `r_ref`, `gamma`, `Tt1`, `Rgas` |
| `compute_tip_mach_number_real` | $M_{tip} = \frac{U_{ref}}{a_{sound}}$ | `r_ref`, `soundspeed`, `omega` or `rpm` |
| `compute_isentropic_efficiency` | $\eta_{is} = \frac{\Pi^{(\gamma-1)/\gamma} - 1}{\tau - 1}$ | `pratio` ($\Pi$), `tratio` ($\tau$), `gamma` |
| `compute_temperature_ratio` | $\tau = 1 + \frac{\Pi^{(\gamma-1)/\gamma} - 1}{\eta_{is}}$ | `pratio` ($\Pi$), `efficiency` ($\eta_{is}$), `gamma` |

### 1.2 Usage Example

```python
from unsflow.turbo.turbo_performance_functions import (
    compute_flow_coefficient,
    compute_work_coefficient_temperature_ratio,
    compute_tip_mach_number,
    compute_isentropic_efficiency
)

# Operating conditions
m_dot = 20.188          # Mass flow rate [kg/s]
rho_in = 1.2            # Inlet density [kg/m^3]
r_tip = 0.25            # Blade tip radius [m]
rpm = 17188.0           # Rotational speed [rpm]
Tt_in = 288.15          # Inlet total temperature [K]
pratio = 2.10           # Total pressure ratio
tratio = 1.28           # Total temperature ratio

# Compute non-dimensional parameters
phi = compute_flow_coefficient(mflow=m_dot, density=rho_in, r_ref=r_tip, rpm=rpm)
psi = compute_work_coefficient_temperature_ratio(tratio=tratio, r_ref=r_tip, rpm=rpm, Tt1=Tt_in)
M_tip = compute_tip_mach_number(rpm=rpm, r_ref=r_tip, Tt1=Tt_in)
eta = compute_isentropic_efficiency(pratio=pratio, tratio=tratio)

print(f"Flow Coefficient (Phi):      {phi:.4f}")
print(f"Work Coefficient (Psi):      {psi:.4f}")
print(f"Tip Mach Number (M_tip):     {M_tip:.3f}")
print(f"Isentropic Efficiency (eta): {eta*100:.2f}%")
```

---

## 2. `unsflow.utils`: Publication Graphics and Formatting

The `unsflow.utils` module standardizes terminal output formatting and ensures all generated figures match rigorous academic publishing standards (PhD theses, ASME Journal of Turbomachinery, AIAA Journal).

Implemented in [`src/unsflow/utils/`](../src/unsflow/utils/):
- [`thesis_plots.py`](../src/unsflow/utils/thesis_plots.py): Matplotlib style sheets with LaTeX Computer Modern typography and sizing calibrated to thesis page widths.
- [`formatting.py`](../src/unsflow/utils/formatting.py): Clean terminal banners and status boxes.
- [`plot_styles.py`](../src/unsflow/utils/plot_styles.py): Consistent palette and transparency standards.

### 2.1 `thesis_plots`

Provides functions to automatically configure Matplotlib for LaTeX compilation without visual clipping:

- `set_thesis_style()`: Enables LaTeX rendering, sets serif fonts (`cm`), sets standard label and tick font sizes.
- `create_figure(fraction, aspect_ratio, subplots)`: Calculates figure dimensions as an exact fraction of the thesis text width ($418.25\,\text{pt}$ / $5.78\,\text{in}$):

```python
import matplotlib.pyplot as plt
from unsflow.utils.thesis_plots import set_thesis_style, create_figure

# 1. Activate LaTeX publication styling
set_thesis_style()

# 2. Create figure (e.g. half-page width, aspect ratio 1.2)
fig, ax = create_figure(fraction=0.5, aspect_ratio=1.2, subplots=(1, 1))

ax.plot([0, 1], [0, 1], label=r'$\Phi / \Phi_{des}$')
ax.set_xlabel(r'Mass Flow Coefficient $\Phi$')
ax.set_ylabel(r'Pressure Rise $\Psi$')
ax.legend()

# 3. Export vector PDF
fig.savefig('publishable_figure.pdf', bbox_inches='tight')
```

### 2.2 `formatting`

Provides consistent console logging:

```python
from unsflow.utils.formatting import print_banner

print_banner("SIMULATION INITIALIZED")
# Prints:
# --------------------------------------------------------------------------------
#                             SIMULATION INITIALIZED                              
# --------------------------------------------------------------------------------
```

---

## References

- **[1] Bell, I. H., Wronski, J., Quoilin, S., & Lemort, V.** (2014). *Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and the Open-Source Thermophysical Property Library CoolProp*. Industrial & Engineering Chemistry Research, 53(6), 2498–2508.
