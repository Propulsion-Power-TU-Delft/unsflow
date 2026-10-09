# `unsflow.greitzer`: Lumped-Parameter Compressor Stability Models

The `unsflow.greitzer` module provides implementations of the classical lumped-parameter compressor instability models developed by **Greitzer (1976)** [1] and **Moore & Greitzer (1986)** [2]. These models capture the non-linear dynamic behavior of compression systems during stall inception, rotating stall transients, and surge limit cycles.

---

## 1. Theoretical Background

A compression system consisting of an inlet duct, compressor, plenum, and throttle duct is modeled as an equivalent lumped-parameter dynamic system.

```
          Inlet Duct             Plenum Volume (Vp)          Throttle Duct
    =====[ Compressor ]================[        ]==============[ Throttle ]=====>
         (Ac, Lc, Uref)                (   Pp   )              (At, Lt, Kv)
```

### 1.1 Greitzer 1D Model (1976)

The 1D model describes the temporal evolution of three state variables:
1. $\Phi_c$: Compressor flow coefficient ($\dot{m}_c / (\rho A_c U_{ref})$)
2. $\Phi_t$: Throttle flow coefficient ($\dot{m}_t / (\rho A_c U_{ref})$)
3. $\Psi_p$: Plenum total-to-static pressure rise coefficient ($(p_p - p_{01}) / (\frac{1}{2}\rho U_{ref}^2)$)

The governing ordinary differential equations (ODEs) in non-dimensional time $\xi = t \cdot a \sqrt{\frac{A_c}{V_p L_c}}$ are:

$$\frac{d\Phi_c}{d\xi} = B \left( \Psi_c(\Phi_c) - \Psi_p \right)$$

$$\frac{d\Phi_t}{d\xi} = \frac{B}{G} \left( \Psi_p - K_{valve} \Phi_t^2 \right)$$

$$\frac{d\Psi_p}{d\xi} = \frac{1}{B} \left( \Phi_c - \Phi_t \right)$$

#### Characteristic Non-Dimensional Parameters:
- **Greitzer $B$ parameter**:
  $$B = \frac{U_{ref}}{2 a} \sqrt{\frac{V_p}{A_c L_c}}$$
  $B$ characterizes the ratio of pressure forces to inertial forces. For small $B$ ($B < B_{cr}$), the system typically develops stable rotating stall; for large $B$ ($B > B_{cr}$), large-amplitude deep surge oscillations occur.
- **Inertia ratio $G$**:
  $$G = \frac{L_t A_c}{L_c A_t}$$

### 1.2 Moore–Greitzer 2D Model (1986)

The Moore–Greitzer model couples the 1D axisymmetric system with the growth of 2D asymmetric circumferential flow perturbations (rotating stall cells).

State variables:
1. $\Psi$: Total-to-static pressure rise coefficient
2. $\Phi$: Annulus-averaged flow coefficient
3. $J$: Squared amplitude of the rotating stall cell ($J = A^2$)

The system equations are:

$$\frac{d\Psi}{d\xi} = \frac{W}{4 L_c B^2} \left[ \frac{\Phi}{W} - \frac{1}{W}\sqrt{\frac{\Psi}{K_{valve}}} \right]$$

$$\frac{d\Phi}{d\xi} = \frac{H}{L_c} \left[ -\frac{\Psi - \Psi_{c0}}{H} + 1 + \frac{3}{2}\left(\frac{\Phi}{W} - 1\right)\left(1 - \frac{J}{2}\right) - \frac{1}{2}\left(\frac{\Phi}{W} - 1\right)^3 \right]$$

$$\frac{dJ}{d\xi} = J \left[ 1 - \left(\frac{\Phi}{W} - 1\right)^2 - \frac{J}{4} \right] \frac{3 a_{lag} H}{W(1 + m a_{lag})}$$

where:
- $\Psi_{c0}, H, W$: Cubic unstalled compressor characteristic parameters:
  $$\Psi_c(\Phi) = \Psi_{c0} + H \left[ 1 + \frac{3}{2}\left(\frac{\Phi}{W} - 1\right) - \frac{1}{2}\left(\frac{\Phi}{W} - 1\right)^3 \right]$$
- $a_{lag}$: Reciprocal time lag parameter of the blade row.
- $m$: Duct geometry parameter (typically $1 \le m \le 2$).

---

## 2. Module Architecture & API

The module is implemented in [`src/unsflow/greitzer/`](../src/unsflow/greitzer/):

| File | Class / Functions | Description |
|---|---|---|
| [`config.py`](../src/unsflow/greitzer/config.py) | `Config` | Parses INI configuration files for geometry, compressor characteristic, and operating parameters |
| [`greitzer.py`](../src/unsflow/greitzer/greitzer.py) | `Greitzer` | Solves 1D and 2D ODE systems, computes stability maps, and produces trajectory plots |
| [`greitzer.py`](../src/unsflow/greitzer/greitzer.py) | `unstalled_characteristic()` | Evaluates the cubic unstalled compressor characteristic |

### Key Methods of `Greitzer`:
- `computeCompressorValveIntersection()`: Determines the equilibrium point $(\Phi_{eq}, \Psi_{eq})$ and local characteristic slopes $\Psi_c', \Psi_v'$ using `scipy.optimize.fsolve`.
- `computeLinearizedStabilityMap()`: Sweeps the $(B, G)$ parameter space, solves the characteristic polynomial roots, and extracts the maximum real eigenvalue.
- `plotStabilityMap(save_filename)`: Generates contour maps of the neutral stability boundary ($\sigma = 0$) and growth rates.
- `solveGreitzerSystem()`: Integrates the 1D Greitzer ODE system over non-dimensional time $\xi$.
- `solveMooreGreitzerSystem()`: Integrates the 2D Moore–Greitzer ODE system with stall cell growth.
- `plotTemporalEvolutionGreitzer()`, `plotTrajectoryGreitzer()`: Plots phase portraits and time series.

---

## 3. Configuration File Format (`input.ini`)

```ini
[COMPRESSION SYSTEM]
; Rotational speed [rpm]
RPM = 80e3
; Reference radius [m]
REFERENCE_RADIUS = 13e-3
; Speed of sound [m/s]
SOUND_SPEED = 340
; Plenum volume [m^3]
PLENUM_VOLUME = 0.1
; Inlet duct geometry [m]
INLET_DUCT_DIAMETER = 30.4e-3 
INLET_DUCT_LENGTH = 2.0
; Throttle duct geometry [m]
THROTTLE_DUCT_DIAMETER = 25e-3
THROTTLE_DUCT_LENGTH = 2.0
; Throttle valve resistance coefficient (defines operating point)
K_VALVE = 2.9

[COMPRESSOR]
; Unstalled cubic characteristic: Psi_c(Phi) = PSI_C_0 + H*(1 + 1.5*(Phi/W - 1) - 0.5*(Phi/W - 1)^3)
PSI_C_0_CHAR = 0.3
W_CHAR = 0.25
H_CHAR = 0.18

[SIMULATION]
; Simulation duration [s]
MAX_TIME = 13

[MOORE GREITZER]
; Reciprocal lag parameter
A_LAG = 0.286
; Moore duct parameter
M_PARAM = 1.75
```

---

## 4. Usage Example

```python
from unsflow.greitzer.config import Config
from unsflow.greitzer.greitzer import Greitzer

# 1. Load configuration
config = Config('input.ini')

# 2. Instantiate model
model = Greitzer(config)

# 3. Simulate post-stall transients (Moore-Greitzer)
model.solveMooreGreitzerSystem()

# 4. Generate diagnostic plots
model.plotTemporalEvolutionMooreGreitzer(save_filename='transient_response')
model.plotTrajectoryMooreGreitzer(save_filename='phase_portrait')

# 5. Save results
model.savePickle(save_filename='moore_greitzer_solution')
```

---

## 5. Included Test Cases

Located in [`testcases/greitzer/`](../testcases/greitzer/):

1. **`greitzer_cases/nonlinear_simulation/`**:
   - Compares stable vs. unstable limit cycle dynamics for 1D compression systems.
   - Run: `python simulation.py`
2. **`greitzer_cases/linearised_stability/`**:
   - Generates the $(B, G)$ linearized stability map across $B \in [0.05, 1.25]$ and $G \in [0.05, 3.5]$.
   - Run: `python stability_map.py`
3. **`greitzer_cases/linearised_stability_variable_equilibrium/`**:
   - Parametric sweep of throttle coefficient $K_T \in [4.5, 9.0]$ to trace the movement of eigenvalues across the complex plane.
4. **`moore_greitzer_cases/surge_case/`**:
   - Large plenum volume ($B > B_{cr}$) yielding deep surge limit cycle oscillations.
   - Run: `python simulation.py`
5. **`moore_greitzer_cases/rotating_stall_case/`**:
   - Small plenum volume ($B < B_{cr}$) leading to finite-amplitude rotating stall cell equilibrium ($J > 0$).
   - Run: `python simulation.py`
6. **`moore_greitzer_cases/stalled_characteristic/`**:
   - Computes the full stalled characteristic curve and hysteresis loop for varying throttle settings $K_T \in [2.5, 5.0]$.

---

## References

- **[1] Greitzer, E. M.** (1976). *Surge and Rotating Stall in Axial Flow Compressors—Part I: Theoretical Compression System Model*. ASME Journal of Engineering for Power, 98(2), 190–198.
- **[2] Moore, F. K., & Greitzer, E. M.** (1986). *A Theory of Post-Stall Transients in Axial Compression Systems: Part I—Development of Equations*. ASME Journal of Engineering for Gas Turbines and Power, 108(1), 68–76.
