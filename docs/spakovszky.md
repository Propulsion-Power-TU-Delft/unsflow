# `unsflow.spakovszky`: Dynamic Compression System Modeling Framework

The `unsflow.spakovszky` module implements the small-perturbation stability assessment framework for axial and radial turbomachinery developed by **Zoltán S. Spakovszky (MIT, 2000)** [1, 2]. 

The model represents compression system components as transfer functions in the Laplace domain and predicts the onset of rotating stall and surge via global system eigenvalue / pole analysis.

---

## 1. Theoretical Background

Linearized compressible flow perturbations of axial velocity $\delta V_x$, tangential velocity $\delta V_\theta$, and static pressure $\delta p$ are decomposed into harmonic Fourier modes in azimuth and Laplace modes in time:

$$\delta q(x, \theta, t) = \sum_{n=-\infty}^{+\infty} \hat{q}_n(x) e^{s t + j n \theta}$$

where:
- $s = \sigma + j \omega$ is the complex Laplace eigenvalue:
  - $\sigma = \text{Re}(s)$: modal growth rate ($\sigma > 0$ denotes instability).
  - $\omega = \text{Im}(s)$: modal oscillation frequency (stall cell rotational speed $\omega / n$).
- $n \in \{1, 2, 3, \dots\}$ is the circumferential harmonic wavenumber.

### 1.1 State Vector and Wave Decomposition

In each duct or gap region, the physical perturbations relate to potential and vorticity wave amplitudes $[A_n, B_n, C_n]^T$ through the transmission matrix $T(x, s, n)$:

$$\begin{bmatrix} \delta V_x \\ \delta V_\theta \\ \delta p \end{bmatrix} = T_{ax}(x, s, n) \begin{bmatrix} A_n \\ B_n \\ C_n \end{bmatrix}$$

- $A_n$: Upstream-decaying potential perturbation ($\propto e^{n x}$)
- $B_n$: Downstream-decaying potential perturbation ($\propto e^{-n x}$)
- $C_n$: Downstream-convected vortical perturbation ($\propto e^{-(s/V_x + j n V_y / V_x) x}$)

For radial components (impellers, vaneless and vaned diffusers), an analogous transmission matrix $T_{rad}(r, s, n)$ maps perturbations as functions of radius $r$, source strength $Q$, and circulation $\Gamma$.

### 1.2 Component Transfer Functions

For each component $k$, an input-output transfer matrix $H_k(s, n)$ relates downstream to upstream state vectors:
- **Axial Blade Rows (`AxialRotor`, `AxialStator`)**:
  Modeled as semi-actuator disks incorporating:
  - Blade row inertia parameter $\lambda = \frac{c}{\cos^2\gamma}$
  - Steady-state loss slope: $\frac{d\mathcal{L}}{d(\tan\beta)}$
  - Unsteady blade response time lag $\tau$
  - Deviation / turning characteristics
- **Ducts and Gaps (`AxialDuct`, `AxialGap`)**:
  Transmission across axial distances with mean background velocities $(V_x, V_y)$.
- **Radial Components (`RadialImpeller`, `VanedDiffuser`, `VanelessDiffuser`)**:
  Formulated in polar/cylindrical coordinates accounting for radius ratio, logarithmic spiral streamlines, and non-uniform mean flows.

### 1.3 Global System Transfer Function & Root Finding

Connecting components in streamwise order yields the global system transfer matrix $M(s, n)$:

$$M(s, n) = H_N^{-1}(s, n) \cdot H_{N-1}(s, n) \cdots H_1(s, n)$$

Applying the boundary conditions (inlet upstream boundary condition $IC$ and outlet downstream boundary condition $EC$):

$$\det \Big( X_{sys}(s, n) \Big) = 0$$

Roots $s_i = \sigma_i + j \omega_i$ of the system determinant are solved using the **`Shot_Gun`** randomized root-search algorithm in the complex plane domain $[\sigma_{min}, \sigma_{max}] \times [\omega_{min}, \omega_{max}]$.

---

## 2. Module Architecture

Implemented in [`src/unsflow/spakovszky/`](../src/unsflow/spakovszky/):

| Component / File | Purpose |
|---|---|
| [`driver.py`](../src/unsflow/spakovszky/driver.py) | `Driver`: Orchestrates component assembly, boundary conditions, pole searching, and root loci |
| [`axial_duct.py`](../src/unsflow/spakovszky/axial_duct.py) | `AxialDuct`: Inlet/exit duct flow dynamics |
| [`axial_gap.py`](../src/unsflow/spakovszky/axial_gap.py) | `AxialGap`: Axial spacing between rotor and stator blade rows |
| [`axial_rotor.py`](../src/unsflow/spakovszky/axial_rotor.py) | `AxialRotor`: Rotating blade row with unsteady loss and inertia |
| [`axial_stator.py`](../src/unsflow/spakovszky/axial_stator.py) | `AxialStator`: Stationary blade row actuator disk model |
| [`radial_impeller.py`](../src/unsflow/spakovszky/radial_impeller.py) | `RadialImpeller`: Centrifugal compressor impeller dynamics |
| [`vaned_diffuser.py`](../src/unsflow/spakovszky/vaned_diffuser.py) | `VanedDiffuser`: Centrifugal vaned diffuser row |
| [`vaneless_diffuser.py`](../src/unsflow/spakovszky/vaneless_diffuser.py) | `VanelessDiffuser`: Radial swirling flow diffuser with radius ratio |
| [`swirling_flow.py`](../src/unsflow/spakovszky/swirling_flow.py) | `SwirlingFlow`: Free-vortex and swirling flow annular duct dynamics |
| [`functions.py`](../src/unsflow/spakovszky/functions.py) | Analytical matrices (`Tax_n`, `Trad_n`, `Bgap_n`) and `Shot_Gun` pole search engine |

---

## 3. Usage Example: Axial Stage Stability

```python
import numpy as np
import matplotlib.pyplot as plt
from unsflow.spakovszky.driver import Driver
from unsflow.spakovszky.axial_duct import AxialDuct
from unsflow.spakovszky.axial_rotor import AxialRotor
from unsflow.spakovszky.axial_gap import AxialGap
from unsflow.spakovszky.axial_stator import AxialStator

# 1. Define stage flow angles and velocities
Vx1 = 0.34
Vy1 = 0.0
alfa1 = 0.0
beta1 = -71.1 * np.pi / 180
beta2 = -35.0 * np.pi / 180
alfa2 = 65.7 * np.pi / 180
lambda_r = 0.33     # Rotor inertia
dLr_dTanb = -0.28   # Rotor loss slope
tau_r = 0.0         # Blade response lag

alfa3 = alfa2
beta3 = beta2
alfa4 = 0.0
beta4 = -71.1 * np.pi / 180
lambda_s = 0.25     # Stator inertia
dLs_dTana = 0.04    # Stator loss slope
tau_s = 0.0

# 2. Build component chain
inlet  = AxialDuct(Vy=Vy1, Vx=Vx1, x=0.0)
rotor  = AxialRotor(Vx1, Vy1, Vx1*np.tan(alfa2), alfa1, beta1, beta2, lambda_r, dLr_dTanb, tau_r)
gap    = AxialGap(x1=0.08, x2=0.18, Vx=Vx1, Vy=Vx1*np.tan(alfa2))
stator = AxialStator(Vx1, Vx1*np.tan(alfa2), 0.0, alfa3, alfa4, lambda_s, dLs_dTana, tau_s)
outlet = AxialDuct(Vy=0.0, Vx=Vx1, x=0.25)

# 3. Assemble system in Driver
driver = Driver('axial')
driver.add_component(inlet)
driver.add_component(rotor)
driver.add_component(gap)
driver.add_component(stator)
driver.add_component(outlet)

# 4. Set boundary conditions
driver.set_inlet_boundary_conditions()
driver.set_outlet_boundary_conditions('infinite duct length')

# 5. Search for eigenvalues (harmonics n = 1 to 5)
domain = [-2.5, 0.5, -4.5, 4.5]  # [sigma_min, sigma_max, omega_min, omega_max]
driver.set_eigenvalues_research_settings(domain=domain, grid=[1, 1], attempts=30, tol=1e-3)
driver.find_eigenvalues(harmonic_numbers=np.arange(1, 6))

# 6. Plot and export root locus
driver.plot_eigenvalues(save_filename='rotor_stator_stability')
driver.store_results_pickle(save_filename='stage_results')
plt.show()
```

---

## 4. Included Test Cases

Located in [`testcases/spakovszky/`](../testcases/spakovszky/):

1. **`axial_rotor_stator/`**:
   - Replicates **Exercise 5.4.3 of Spakovszky's PhD thesis**.
   - Investigates the effect of rotor-stator gap spacing ($\Delta x / c \in \{0.0, 0.3, 5.0\}$) on modal stability.
   - Run: `python main.py`
2. **`axial_rotor_stator_root_locus/`**:
   - Traces the root locus as compressor flow coefficient decreases towards stall inception.
   - Run: `python main.py` and `python main_post_process.py`
3. **`iris/`**:
   - Stability assessment for the **IRIS centrifugal compressor stage**.
   - Sweeps across four operating speedlines (`speedline_0`, `speedline_1`, `speedline_2`, `speedline_3`).
   - Run: `python main_analysis.py` inside each speedline directory.
4. **`notebooks/`**:
   - `jet_wake_perturbations.ipynb`: Analyzes the impact of non-uniform circumferential impeller discharge (jet-wake structure) on downstream stability.
   - `NASA_CC3.ipynb`: Dynamic modeling of the NASA CC3 centrifugal compressor test rig.

---

## References

- **[1] Spakovszky, Z. S.** (2000). *Applications of Axial and Radial Compressor Dynamic System Modeling*. Ph.D. Thesis, Massachusetts Institute of Technology.
- **[2] Spakovszky, Z. S., Weigl, H. J., Paduano, J. D., van Schalkwyk, C. M., Suder, K. L., & Bright, M. M.** (2001). *Rotating Stall Control in a High-Speed Stage With Inlet Distortion: Part I—System Modeling*. ASME Journal of Turbomachinery, 123(1), 29–38.
