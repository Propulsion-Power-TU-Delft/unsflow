# `unsflow.senoo`: Centrifugal Vaneless Diffuser Rotating Stall Model

The `unsflow.senoo` module provides an implementation of the boundary-layer and rotating-stall inception theory for centrifugal compressor vaneless diffusers developed by **Yasutoshi Senoo and co-workers (ASME, 1977)** [1, 2].

The model computes the asymmetric development of turbulent boundary layers on the hub and shroud walls under adverse radial pressure gradients, predicts core flow deceleration, evaluates wall shear stresses, and determines the **critical flow angle** for the inception of wall reverse flow and rotating stall.

---

## 1. Theoretical Background

In centrifugal compressor vaneless diffusers, the flow enters with high tangential swirl and decelerates as radius increases. Because of the radial pressure gradient, low-momentum fluid near the walls turns more tangentially than the core flow (cross-flow skewing). At low flow rates, this cross-flow leads to local radial reverse flow ($v_r < 0$) along one or both walls, triggering rotating stall.

```
       Shroud Wall (Boundary Layer delta_2)
   -------------------------------------------------  y = b(r)
      Core Flow (V_c, alpha_c, p) -->
   -------------------------------------------------  y = 0
       Hub Wall (Boundary Layer delta_1)
   r1 (Inlet)                                       r2 (Exit)
```

### 1.1 Velocity Profile Parameterization

The passage is divided into three zones across width $b(r)$:
1. **Hub boundary layer** ($0 \le y \le \delta_1$):
   $$\frac{v_t}{V_c} = \left(\frac{y}{\delta_1}\right)^{1/n}, \qquad \frac{v_r}{V_c} = \left(\frac{y}{\delta_1}\right)^{1/n} \left[ 1 - \left(\frac{y}{\delta_1}\right)^m \right] \tan\alpha_{w1} + \left(\frac{y}{\delta_1}\right)^{1/n+m} \tan\alpha_c$$
2. **Core flow** ($\delta_1 \le y \le b - \delta_2$):
   $$v_t(r) = V_c(r) \cos\alpha_c(r), \qquad v_r(r) = V_c(r) \sin\alpha_c(r)$$
3. **Shroud boundary layer** ($b - \delta_2 \le y \le b$):
   Symmetric or asymmetric power-law profile governed by $\delta_2(r)$ and wall angle $\alpha_{w2}(r)$.

### 1.2 Governing Momentum Integral Equations

Integrating the radial and tangential Reynolds-averaged Navier-Stokes equations along the passage width yields a system of coupled first-order ODEs:
- **Radial Momentum Integral**:
  $$\frac{d}{dr} \int_0^b \rho v_r^2 \, dy - \frac{1}{r} \int_0^b \rho v_t^2 \, dy = -b \frac{dp}{dr} - (\tau_{wr1} + \tau_{wr2})$$
- **Tangential Momentum Integral**:
  $$\frac{d}{dr} \int_0^b \rho r v_r v_t \, dy = -r (\tau_{wt1} + \tau_{wt2})$$
- **Conservation of Mass**:
  $$2\pi r \int_0^b \rho v_r \, dy = \dot{m} = \text{const}$$

### 1.3 Wall Shear Stresses & Reverse Flow Criterion

Wall shear stresses $\tau_{w1}, \tau_{w2}$ are computed using the Ludwieg–Tillmann or Blasius turbulent pipe friction formulas adapted for skewed boundary layers.

**Reverse Flow Criterion**:
Reverse flow initiates when the radial wall velocity component changes sign:
$$\tan\alpha_{w1} \le 0 \quad \text{or} \quad \tan\alpha_{w2} \le 0$$
The inlet flow angle at which reverse flow first appears anywhere between $r_1$ and $r_2$ is defined as the **critical flow angle $\alpha_{cr}$**. When the operating angle $\alpha_1 < \alpha_{cr}$, the diffuser is prone to rotating stall.

---

## 2. Module Architecture & API

Implemented in [`src/unsflow/senoo/`](../src/unsflow/senoo/):

| File | Class / Functions | Description |
|---|---|---|
| [`config.py`](../src/unsflow/senoo/config.py) | `SenooConfig` | Dataclass configuring geometry ($r_1, r_2, b_1, b_2$), inlet flow, Reynolds number, and grid resolution |
| [`senoo_diffuser.py`](../src/unsflow/senoo/senoo_diffuser.py) | `SenooDiffuser` | Main solver marching radially, integrating boundary layer equations, and evaluating stability |
| [`results.py`](../src/unsflow/senoo/results.py) | `SenooFlowField` | Result container holding 2D velocity fields, pressure coefficient, boundary layer thicknesses, and reverse flow flags |
| [`profiles.py`](../src/unsflow/senoo/profiles.py) | Analytical Integrals | Precomputes Beta function tables for fast, closed-form evaluation of boundary layer profiles |

### Configuration Factory Methods:
- `SenooConfig.from_dimensional(r1, r2, b1, alpha1_mean, V1_mean, rho, nu, ...)`: Configures the problem from physical dimensions in meters, m/s, kg/m³.
- `SenooConfig.from_nondimensional(b_over_r0, re_over_r0, Rb, alpha1_mean, ...)`: Configures the problem directly from non-dimensional ratios and Reynolds number.

---

## 3. Usage Example

### 3.1 Direct Flowfield Evaluation

```python
from unsflow.senoo import SenooConfig, SenooDiffuser

# 1. Setup configuration (dimensional)
cfg = SenooConfig.from_dimensional(
    r1=0.251,            # Inlet radius [m]
    r2=0.502,            # Exit radius [m]
    b1=0.014,            # Inlet width [m]
    b2=0.014,            # Exit width [m] (parallel walls)
    alpha1_mean=17.0,    # Mean inlet flow angle [deg]
    V1_mean=45.0,        # Mean inlet velocity [m/s]
    rho=1.2,             # Density [kg/m^3]
    nu=1.5e-5,           # Kinematic viscosity [m^2/s]
    nr=80,               # Radial mesh points
    nspan=40             # Spanwise mesh points
)

# 2. Run solver
diffuser = SenooDiffuser(cfg)
flowfield = diffuser.solve()

# 3. Check stability
print(f"Flow reversed: {flowfield.has_reversed_flow}")
if flowfield.has_reversed_flow:
    print(f"Reversal detected at radius r = {flowfield.reversal_radius:.3f} m")

# 4. Access 2D profiles and results
print(f"Exit Cp: {flowfield.cp[-1]:.4f}")
print(f"Exit hub boundary layer thickness: {flowfield.delta1[-1]*1000:.2f} mm")
```

### 3.2 Finding the Critical Flow Angle ($\alpha_{cr}$)

```python
def find_critical_angle(diffuser, low=5.0, high=30.0, tol=0.1):
    """Bisection search to locate the stall inception boundary."""
    while (high - low) > tol:
        mid = 0.5 * (low + high)
        diffuser.config.alpha1_mean = mid
        res = diffuser.solve()
        if res.has_reversed_flow:
            low = mid   # Unstable, critical angle is higher
        else:
            high = mid  # Stable, critical angle is lower
    return 0.5 * (low + high)

alpha_cr = find_critical_angle(diffuser)
print(f"Critical flow angle alpha_cr = {alpha_cr:.2f} deg")
```

---

## 4. Included Test Cases & Validation

Located in [`testcases/senoo/`](../testcases/senoo/):

1. **`replicate_senoo_1977/`**:
   - Comprehensive replication of the **Senoo et al. (1977)** experimental blower rig ($r_1 = 0.251\,\text{m}$, $r_2 = 0.502\,\text{m}$, $b = 14\,\text{mm}$, $Rb = 4.2 \times 10^4$).
   - **Experiment 1 (Unstalled flow, $\alpha_1 = 17^\circ$)**: Replicates Fig. 7 (spanwise velocity profiles) and Fig. 9 (static pressure rise $C_p$ and flow angle distribution).
   - **Experiment 2 (Stalled flow, $\alpha_1 = 8^\circ$)**: Replicates Fig. 8 (reverse flow inception at walls) and Fig. 10.
   - Run: `python replicate_senoo_1977.py`
2. **`hecc_peak_eff/`**:
   - Stability assessment of the High Efficiency Centrifugal Compressor (HECC) vaneless diffuser at peak efficiency condition.
   - Run: `python hecc_peak_eff.py`
3. **`test_senoo_model.py`**:
   - Automated pytest suite validating:
     - Dimensional & non-dimensional configuration factory methods.
     - Monotonic pressure rise in unstalled operating regime.
     - Accurate detection of wall reverse flow onset.
     - Analytical Beta-integral profile matching.

---

## References

- **[1] Senoo, Y., Kinoshita, Y., & Ishida, M.** (1977). *Asymmetric Flow in Vaneless Diffusers of Centrifugal Blowers*. ASME Journal of Fluids Engineering, 99(1), 104–111.
- **[2] Senoo, Y., & Kinoshita, Y.** (1977). *Influence of Inlet Flow Conditions and Geometries of Centrifugal Vaneless Diffusers on Critical Flow Angle for Reverse Flow*. ASME Journal of Fluids Engineering, 99(1), 98–103.
