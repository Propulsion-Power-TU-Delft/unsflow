# `unsflow.senoo`: Centrifugal Vaneless Diffuser Rotating Stall & Surge Model

The `unsflow.senoo` module provides an implementation of the boundary-layer, rotating-stall, and surge inception theory for centrifugal compressor vaneless diffusers developed by **Yasutoshi Senoo and co-workers (ASME 1977, 1978)** [1, 2, 3] and extended by **Lüdtke** [4].

Given a diffuser design geometry ($r_1, r_2, b_1, b_2$), working fluid (ideal gas or real fluids via `CoolProp`), and operating flow conditions, the package calculates:
1. **Operating flow angles**: both radial flow angle ($\alpha = \arctan(C_\theta / C_r)$) and tangential flow angle ($\beta = \arctan(C_r / C_\theta) = 90^\circ - \alpha$).
2. **Rotating stall angle**: $\alpha_{\mathrm{RS}}$ and $\beta_{\mathrm{RS}}$ (based on Senoo's rule $\beta_{\mathrm{RS}} \approx 0.88 \beta_{\mathrm{rev}}$).
3. **Surge angle**: $\alpha_{\mathrm{surge}}$ and $\beta_{\mathrm{surge}}$ (based on wall reverse flow extending to the diffuser exit $r^* = r_2$ or an empirical surge factor).
4. **Compressibility corrections**: corrections for density rise along the diffuser using Senoo's 1D flow-with-friction model mapping.

---

## 1. Theoretical Background

In centrifugal compressor vaneless diffusers, the flow enters with high tangential swirl and decelerates as radius increases. Because of the adverse radial pressure gradient, low-momentum fluid near the hub and shroud walls turns more tangentially than the core flow (cross-flow skewing). At reduced flow rates, this cross-flow leads to local radial reverse flow ($v_r < 0$) along one or both walls, triggering rotating stall and eventually surge.

```
       Shroud Wall (Boundary Layer delta_2, Wall Skew e_2)
   --------------------------------------------------------  y = b(r)
      Core Flow (V_c, alpha_c, p) -->
   --------------------------------------------------------  y = 0
       Hub Wall (Boundary Layer delta_1, Wall Skew e_1)
   r1 (Inlet)                                              r2 (Exit)
```

### 1.1 Velocity Profile Parameterization & Integral Equations

The boundary layer velocity distribution across width $b(r)$ is modeled using Johnston's triangular polar hodograph:
- **Streamwise velocity**: $\frac{u}{U} = \left(\frac{y}{\delta}\right)^{1/n}$
- **Crossflow velocity**: $\frac{w}{U} = e \left(1 - \frac{y}{\delta}\right)^m \frac{u}{U}$

The passage transitions between two zones:
1. **Entry Zone (9 coupled ODEs)**: separate hub and shroud boundary layers ($\delta_1, \delta_2$) separated by an inviscid/turbulent core ($h_c = b - \delta_1 - \delta_2$). Closed via momentum integral equations and numerical Beta-function integrals.
2. **Merged Zone (6 coupled ODEs)**: boundary layers meet at the mid-span ($\delta_1 + \delta_2 = b$).

### 1.2 Angle Conventions

In turbomachinery aerodynamics, two complementary angle conventions are used:
- **Radial reference convention ($\alpha$)**: $0^\circ$ is pure radial, $90^\circ$ is pure tangential.
  $$\tan\alpha = \frac{C_\theta}{C_r}$$
- **Tangential reference convention ($\beta$)**: $0^\circ$ is pure tangential, $90^\circ$ is pure radial.
  $$\tan\beta = \frac{C_r}{C_\theta}, \qquad \alpha + \beta = 90^\circ$$

All outputs of `unsflow.senoo` report angles primarily in the **radial convention** ($\alpha$, e.g. $\alpha_{\mathrm{op}}, \alpha_{\mathrm{RS}}, \alpha_{\mathrm{surge}}$) while maintaining accessible complementary tangential properties ($\beta_{\mathrm{op}}, \beta_{\mathrm{RS}}, \beta_{\mathrm{surge}}$).

### 1.3 Stability Criteria: Rotating Stall & Surge

1. **Wall Reverse Flow Inception ($\alpha_{\mathrm{rev}}, \beta_{\mathrm{rev}}$)**:
   Reverse flow initiates when the radial wall shear stress vanishes:
   $$1 - e \cot\alpha_{\mathrm{core}} \le 0$$
   The radius where this occurs first is the reversal onset radius $r_{\mathrm{rev}} = r^*$.

2. **Rotating Stall Angle ($\alpha_{\mathrm{RS}}, \beta_{\mathrm{RS}}$)**:
   Senoo & Kinoshita (1977, 1978) discovered that rotating stall does not set in instantaneously at the first inception of local reverse flow, but rather when the flow rate is further throttled by an empirical factor of approximately $0.88$ on the tangential flow angle:
   $$\beta_{\mathrm{RS}} \approx 0.88 \cdot \beta_{\mathrm{rev}}$$
   $$\alpha_{\mathrm{RS}} = 90^\circ - \beta_{\mathrm{RS}} = 90^\circ - 0.88 (90^\circ - \alpha_{\mathrm{rev}})$$

3. **Diffuser Surge Angle ($\alpha_{\mathrm{surge}}, \beta_{\mathrm{surge}}$)**:
   Surge occurs when the wall reverse flow layer extends across the entire radial span to the diffuser exit ($r^* = r_2$), or at an empirical mass flow ratio ($\beta_{\mathrm{surge}} = c_{\mathrm{surge}} \beta_{\mathrm{RS}}$).

### 1.4 Compressibility Correction Model

Senoo (1977 Paper 2 & ASME 78-GT-19) demonstrated that compressibility increases density along the diffuser radius ($\rho^* > \rho_i$), decelerating the radial velocity ($C_{r,\mathrm{comp}} < C_{r,\mathrm{incomp}}$) and steering the flow closer to the tangential direction.

The compressibility shift is solved via a 1D compressible swirling flow with wall friction:
$$\frac{dC_\theta}{dr} = -\frac{C_\theta}{r} - \frac{c_f C C_\theta}{b C_r}$$
$$\frac{dp}{dr} = \rho \frac{C_\theta^2}{r} - \rho C_r \frac{dC_r}{dr} - \frac{c_f \rho C C_r}{b}$$
$$\frac{dC_r}{dr} = \frac{C_r}{1 - M_r^2} \left[ -\left(\frac{1}{r} + \frac{1}{b}\frac{db}{dr}\right) - \frac{\gamma}{a^2}\left(\frac{C_\theta^2}{r} - \frac{c_f C C_r}{b}\right) - \frac{\gamma-1}{a^2} C_\theta \frac{dC_\theta}{dr}\right]$$

The incompressible critical angle is then corrected using Senoo's scaling relation:
$$\Delta \beta^* = \beta_{\mathrm{incomp}}(r^*) - \beta_{\mathrm{comp}}(r^*) > 0$$
$$\Delta \beta_{i,c} \approx \frac{\rho^*}{\rho_i} \Delta \beta^*$$
$$\beta_{\mathrm{rev},\mathrm{corr}} = \beta_{\mathrm{rev},\mathrm{incomp}} + \Delta \beta_{i,c}$$
$$\alpha_{\mathrm{rev},\mathrm{corr}} = 90^\circ - \beta_{\mathrm{rev},\mathrm{corr}}$$

---

## 2. Module Architecture

Implemented in [`src/unsflow/senoo/`](../src/unsflow/senoo/):

| File | Class / Function | Description |
|---|---|---|
| [`config.py`](../src/unsflow/senoo/config.py) | `SenooConfig` | Dataclass configuring geometry, CoolProp fluids, flow conditions, and solver modes |
| [`senoo_diffuser.py`](../src/unsflow/senoo/senoo_diffuser.py) | `SenooDiffuser` | Marching ODE solver, 1D flow-with-friction solver, and critical bisection search |
| [`results.py`](../src/unsflow/senoo/results.py) | `SenooResults` (`SenooFlowField`) | Dataclass storing stability angles ($\alpha_{\mathrm{op}}, \alpha_{\mathrm{RS}}, \alpha_{\mathrm{surge}}$), 1D profiles, and 2D flow fields |
| [`profiles.py`](../src/unsflow/senoo/profiles.py) | Analytical Integrals | Beta-function integral tables and wall shear stress closures |
| [`map.py`](../src/unsflow/senoo/map.py) | `senoo_solution_map` | 2D interpolator digitized from Senoo & Kinoshita (1977) Paper 2 design charts |

---

## 3. Usage Examples

### 3.1 Basic Usage: Calculating Stability Angles

```python
from unsflow.senoo import SenooConfig, SenooDiffuser

# Configure vaneless diffuser
cfg = SenooConfig.from_dimensional(
    r1=0.251,           # Inlet radius [m]
    r2=0.502,           # Exit radius [m]
    b1=0.014,           # Passage width [m]
    alpha1_mean=73.0,   # Operating flow angle from radial [deg]
    V1_mean=45.0,       # Mean velocity magnitude [m/s]
    fluid='Air',        # Working fluid
    solver_mode='map'   # 'map' for fast lookup, 'ode' for full differential marching
)

diffuser = SenooDiffuser(cfg)
limits = diffuser.compute_stability_limits()

print(f"Operating Radial Angle:   alpha = {limits['alpha']:.2f} deg")
print(f"Operating Tangential Angle: beta = {limits['beta']:.2f} deg")
print(f"Reverse Flow Inception:   alpha_rev = {limits['alpha_reverse_flow']:.2f} deg")
print(f"Rotating Stall Angle:     alpha_RS  = {limits['alpha_rotating_stall']:.2f} deg")
print(f"Diffuser Surge Angle:     alpha_surge = {limits['alpha_surge']:.2f} deg")
print(f"Is Stalled:               {limits['is_stalled']}")
print(f"Stall Margin:             {limits['stall_margin']:.2f} deg")
```

### 3.2 Real Gas / Multi-Fluid Modeling with CoolProp

The package supports real gases and refrigerants via `CoolProp` (e.g. `Air`, `Argon`, `R1233zd(E)`, `CO2`):

```python
cfg_refrigerant = SenooConfig.from_dimensional(
    r1=0.216,
    r2=0.460,
    b1=0.014,
    alpha1_mean=75.0,
    V1_mean=120.0,
    fluid='R1233zd(E)',
    p1=47789.0,        # Static pressure [Pa] (Chapter 6 conditions)
    T1=273.15,         # Static temperature [K]
    compressible=True  # Activate 1D compressibility correction
)

diffuser = SenooDiffuser(cfg_refrigerant)
results = diffuser.solve()
results.print_summary()
```

### 3.3 Factory from Operating Point (Mass Flow & RPM)

You can initialize a case directly from compressor stage operating variables:

```python
cfg_point = SenooConfig.from_operating_point(
    r1=0.225,
    r2=0.450,
    b1=0.015,
    mdot=4.2,             # Mass flow rate [kg/s]
    rpm=21789.0,          # Impeller speed [rpm]
    slip_factor=0.85,     # Stanitz slip factor
    fluid='Air',
    compressible=True
)

diffuser = SenooDiffuser(cfg_point)
results = diffuser.solve()
print(f"Stall Margin: {results.stall_margin:.2f} deg")
```

### 3.4 Specifying Variable Diffuser Width $b(r)$

The model supports three ways to define variable passage width along radius:

1. **Linear Taper / Pinched Diffusers**:
   Specify `b1` and `b2` (or `pinch_ratio = b2 / b1`):
   ```python
   cfg = SenooConfig.from_dimensional(
       r1=0.25, r2=0.50, b1=0.015, b2=0.010, alpha1_mean=73.0
   )
   ```

2. **Custom Non-Linear Functions (Callable $b(r)$)**:
   Pass any analytical function `b_func(r)` (and optional exact derivative `db_dr_func(r)`):
   ```python
   # Quadratic pinch contour:
   b_func = lambda r: 0.016 - (0.016 - 0.010) * ((r - 0.25) / (0.50 - 0.25))**2
   db_func = lambda r: -2.0 * (0.016 - 0.010) * (r - 0.25) / (0.25**2)

   cfg = SenooConfig.from_dimensional(
       r1=0.25, r2=0.50, b1=0.016, alpha1_mean=73.0,
       b_func=b_func, db_dr_func=db_func
   )
   ```

3. **Discrete Tabular / CAD Coordinates**:
   Pass discrete coordinates along the diffuser (`r_geom, b_geom`); the model constructs a smooth, shape-preserving PCHIP spline with analytical derivatives:
   ```python
   r_cad = [0.22, 0.26, 0.32, 0.40, 0.48]
   b_cad = [0.018, 0.017, 0.015, 0.012, 0.011]

   cfg = SenooConfig.from_dimensional(
       r1=0.22, r2=0.48, b1=0.018, alpha1_mean=73.0,
       r_geom=r_cad, b_geom=b_cad
   )
   ```

---

## 4. Test Suite and Replication Cases

- **`testcases/senoo/test_senoo_model.py`**:
  Comprehensive unit test suite covering configuration initialization, unstalled flow, reverse flow detection, bisection search, pinched diffusers, Mach 0.6 compressible regime, multi-fluid CoolProp support, compressibility shift verification, and operating point factory.
- **`testcases/senoo/replicate_senoo_1977/`**:
  Replicates the experimental results from Senoo, Kinoshita & Ishida (1977) [1], generating Fig. 7 velocity profiles, Fig. 9 pressure rise $C_p$ and flow angle distributions, Fig. 8 wall reverse flow comparison, and 2D velocity contours.
- **`testcases/senoo/hecc_peak_eff/`**:
  Stability assessment and velocity profile replication for the High Efficiency Centrifugal Compressor (HECC) vaneless diffuser at peak efficiency.

---

## References

- **[1] Senoo, Y., Kinoshita, Y., & Ishida, M.** (1977). *Asymmetric Flow in Vaneless Diffusers of Centrifugal Blowers*. ASME Journal of Fluids Engineering, Vol. 99, No. 1, pp. 104–111.
- **[2] Senoo, Y., & Kinoshita, Y.** (1977). *Influence of Inlet Flow Conditions and Geometries of Centrifugal Vaneless Diffusers on Critical Flow Angle for Reverse Flow*. ASME Journal of Fluids Engineering, Vol. 99, No. 1, pp. 98–103.
- **[3] Senoo, Y., & Kinoshita, Y.** (1978). *Limits of Rotating Stall and Stall in Vaneless Diffusers of Centrifugal Compressors*. ASME Paper No. 78-GT-19.
- **[4] Lüdtke, K. H.** (2004). *Process Centrifugal Compressors: Basics, Function, Operation, Design, Manufacture*. Springer.
