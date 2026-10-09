<div align="center">

# Unsflow

### Advanced Compression System Instability Analysis & Turbomachinery Modeling Suite

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)
[![Status: Research](https://img.shields.io/badge/status-research--grade-orange.svg)]()
[![Solvers: CTurboBFM & SU2](https://img.shields.io/badge/mesh_export-CTurboBFM%20%7C%20SU2-purple.svg)](https://github.com/Propulsion-Power-TU-Delft/CTurboBFM)

</div>

---

**`unsflow`** is an integrated Python framework developed for the **aerodynamic stability assessment of axial and radial compression systems**, the **post-processing of CFD simulations**, and **automated meridional grid generation for Body Force Modeling (BFM)**.

Developed for turbomachinery researchers and aerodynamicists, `unsflow` bridges the gap between analytical reduced-order stability theories and full 3D CFD by providing:
- Multiple reduced-order instability frameworks (lumped-parameter, actuator-disk/transfer matrix, generalized eigenvalue formulation, and diffuser boundary-layer models).
- Automated meridional grid generation and 3D blade geometry reconstruction (camber, thickness, lean, metal angles, normal vectors, blockage).
- Direct grid and body-force file export for the **[CTurboBFM](https://github.com/Propulsion-Power-TU-Delft/CTurboBFM)** BFM solver and **[SU2](https://su2code.github.io/)** CFD solver.
- Publication-quality visualization tools conforming to academic thesis and journal typography standards.

---

## Architecture Overview

```
                                      +---------------------------------------------+
                                      |                   UNSFLOW                   |
                                      +---------------------------------------------+
                                                             |
         +--------------------+------------------------------+------------------------------+--------------------+
         |                    |                              |                              |                    |
+-----------------+  +-----------------+           +-------------------+          +-------------------+  +-----------------+
| unsflow.greitzer|  |unsflow.spakovszky|          |    unsflow.sun    |          |   unsflow.senoo   |  |  unsflow.grid   |
+-----------------+  +-----------------+           +-------------------+          +-------------------+  +-----------------+
| Lumped-parameter|  |Small-perturbation|          |General instability|          |Vaneless diffuser  |  |Meridional meshes|
| 1D & 2D dynamic |  |transfer-matrix   |          |eigenvalue solver  |          |boundary layers    |  |Blade geometry   |
| stability models|  |system modeling   |          |on 2D CFD/BFM grids|          |and rotating stall |  |Body Force Model |
| (Greitzer 1976; |  |(Spakovszky 2000; |          |(Sun et al. 2013)  |          |(Senoo et al. 1977)|  |(CTurboBFM & SU2)|
| Moore-Greitzer) |  |axial and radial) |          |                   |          |                   |  |                 |
+-----------------+  +-----------------+           +-------------------+          +-------------------+  +-----------------+
         |                    |                              |                              |                    |
         +--------------------+------------------------------+------------------------------+--------------------+
                                                             |
                                      +----------------------+----------------------+
                                      |                                             |
                             +------------------+                          +------------------+
                             |  unsflow.turbo   |                          |  unsflow.utils   |
                             +------------------+                          +------------------+
                             |Aerothermodynamic |                          |Publication plots,|
                             |performance, flow |                          |LaTeX typography, |
                             |& work coeffs     |                          |pretty banners    |
                             +------------------+                          +------------------+
```

---

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Documentation Index](#documentation-index)
- [Package Breakdown](#package-breakdown)
  - [1. unsflow.greitzer](#1-unsflowgreitzer)
  - [2. unsflow.spakovszky](#2-unsflowspakovszky)
  - [3. unsflow.sun](#3-unsflowsun)
  - [4. unsflow.senoo](#4-unsflowsenoo)
  - [5. unsflow.grid](#5-unsflowgrid)
  - [6. unsflow.turbo](#6-unsflowturbo)
  - [7. unsflow.utils](#7-unsflowutils)
- [Use Cases & Validation Suite](#use-cases--validation-suite)
- [Running Tests](#running-tests)
- [Repository Structure](#repository-structure)
- [References & Citation](#references--citation)
- [License](#license)

---

## Installation

### Prerequisites
- Python 3.9 or higher (Python 3.9 recommended).
- [Conda](https://docs.conda.io/en/latest/) or [Mamba](https://mamba.readthedocs.io/).

### 1. Create Conda Environment
From the repository root, create the dedicated environment using [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
```

### 2. Activate the Environment
```bash
conda activate unsflow
```

### 3. Install `unsflow` in Editable Mode
Install the package in development/editable mode so that changes to the source code take effect immediately:

```bash
pip install -e .
```

---

## Quickstart

### 1. Solve a Moore–Greitzer Post-Stall Transient
Simulate the dynamic evolution of mass flow, pressure rise, and rotating stall cell amplitude $J$:

```python
from unsflow.greitzer.config import Config
from unsflow.greitzer.greitzer import Greitzer

# Load compression system parameters from INI file
config = Config('testcases/greitzer/moore_greitzer_cases/surge_case/input.ini')

# Solve dynamic equations and plot trajectory
model = Greitzer(config)
model.solveMooreGreitzerSystem()
model.plotTrajectoryMooreGreitzer(save_filename='quickstart_trajectory')
```

### 2. Detect Critical Stall Angle in a Centrifugal Diffuser
Predict whether a vaneless diffuser operates safely or will develop wall reverse flow:

```python
from unsflow.senoo import SenooConfig, SenooDiffuser

# Configure vaneless diffuser geometry and flow conditions
cfg = SenooConfig.from_dimensional(
    r1=0.251, r2=0.502, b1=0.014,
    alpha1_mean=17.0, V1_mean=45.0,
    rho=1.2, nu=1.5e-5
)

diffuser = SenooDiffuser(cfg)
flowfield = diffuser.solve()

if flowfield.has_reversed_flow:
    print(f"Stall warning: Reverse flow detected at radius {flowfield.reversal_radius:.3f} m")
else:
    print(f"Diffuser flow is stable! Exit Cp = {flowfield.cp[-1]:.3f}")
```

### 3. Generate a Multiblock Meridional Grid for BFM
Generate a structured mesh and reconstruct blade properties for CTurboBFM:

```python
from unsflow.grid.config import Config
from unsflow.grid.multiblock_grid_driver import MultiBlockGridDriver

config = Config('testcases/grid/NR37/multiblock/input.ini')
driver = MultiBlockGridDriver(config)
driver.GenerateGrid()
driver.ComputeBladesData()
driver.AssembleMultiBlockGrid()
driver.SaveOutput()
```

---

## Documentation Index

For in-depth mathematical formulations, configuration file specifications, and advanced workflows, refer to the dedicated package guides in the [`docs/`](docs/) directory:

| Document | Topic | Key Focus |
|---|---|---|
| 📖 **[`docs/greitzer.md`](docs/greitzer.md)** | Lumped-Parameter Models | 1D Greitzer and 2D Moore–Greitzer models, surge cycles, limit cycles, $B$ and $G$ linearized stability maps |
| 📖 **[`docs/spakovszky.md`](docs/spakovszky.md)** | Dynamic Transfer Matrix Framework | Small-perturbation wave decomposition ($A_n, B_n, C_n$), axial/radial components, `Shot_Gun` pole searching |
| 📖 **[`docs/sun.md`](docs/sun.md)** | Sun Inception Theory | Linearized 2D/3D Euler perturbation equations on CFD grids, spectral operators, multi-block generalized EVP |
| 📖 **[`docs/senoo.md`](docs/senoo.md)** | Senoo Diffuser Theory | 3D asymmetric turbulent boundary layers, wall shear stresses, critical flow angle $\alpha_{cr}$ determination |
| 📖 **[`docs/grid.md`](docs/grid.md)** | Grid & BFM Engine | TFI & elliptic grid generation, blade camber/thickness/normals, Body Force Modeling, CTurboBFM and SU2 export |
| 📖 **[`docs/turbo_and_utils.md`](docs/turbo_and_utils.md)** | Aero & Graphics Utilities | Non-dimensional performance metrics, CoolProp real fluids, thesis-quality LaTeX Matplotlib plotting |

---

## Package Breakdown

### 1. `unsflow.greitzer`
*Lumped-parameter compressor dynamic stability models.*

- **Theoretical Basis**: Greitzer (1976) [1] and Moore & Greitzer (1986) [2].
- **Core Features**:
  - Simulates 1D nonlinear compression system dynamics ($\Phi_c, \Phi_t, \Psi_p$).
  - Solves 2D post-stall transients with rotating stall cell growth ($\Psi, \Phi, J$).
  - Computes linearized stability maps across the Greitzer $B$ parameter and duct inertia parameter $G$:
    $$B = \frac{U_{ref}}{2 a} \sqrt{\frac{V_p}{A_c L_c}}, \qquad G = \frac{L_t A_c}{L_c A_t}$$
  - Evaluates the intersection of the throttle line and unstalled cubic compressor characteristic to determine equilibrium operating points and slope criteria.
- **Key Classes**:
  - `Greitzer`: Solver class for 1D and 2D systems, root finding, and trajectory plotting.
  - `Config`: Configuration parser for system dimensions and cubic characteristic parameters.
- **Reference Guide**: [docs/greitzer.md](docs/greitzer.md)

---

### 2. `unsflow.spakovszky`
*Dynamic system stability modeling using transfer matrix analysis.*

- **Theoretical Basis**: Spakovszky (2000) [3], Spakovszky et al. (2001).
- **Core Features**:
  - Decomposes linear flow perturbations $(\delta V_x, \delta V_\theta, \delta p)$ or $(\delta V_r, \delta V_\theta, \delta p)$ into Fourier modes in azimuth ($n$) and Laplace modes in time ($s = \sigma + j\omega$).
  - Expresses physical perturbations via transmission matrices relating potential ($A_n, B_n$) and vorticity ($C_n$) wave amplitudes:
    $$[\delta V_x, \delta V_\theta, \delta p]^T = T(x, s, n) [A_n, B_n, C_n]^T$$
  - Chains component transfer functions:
    - **Axial components**: `AxialDuct`, `AxialGap`, `AxialRotor`, `AxialStator`.
    - **Radial components**: `RadialImpeller`, `VanedDiffuser`, `VanelessDiffuser`, `SwirlingFlow`.
  - Boundary conditions: inlet non-reflecting/standard, outlet infinite duct, finite duct length, or radial plenum discharge.
  - Implements the **`Shot_Gun`** randomized root-search algorithm to find system poles and trace root loci as operating conditions vary.
- **Key Classes**:
  - `Driver`: Orchestrates component connection, transfer function matrix multiplication, and pole searching.
- **Reference Guide**: [docs/spakovszky.md](docs/spakovszky.md)

---

### 3. `unsflow.sun`
*High-order flow-instability inception theory on CFD/BFM mean flow fields.*

- **Theoretical Basis**: Xiaofeng Sun et al. (2013) [4].
- **Core Features**:
  - Solves the linearized compressible Euler perturbation equations on 2D meridional meshes:
    $$\left( -j \omega A + B \frac{\partial}{\partial r} + j \frac{m}{r} C + E \frac{\partial}{\partial z} + R + S \right) \hat{\Phi} = 0$$
  - Directly couples the mean flow fields, spatial velocity/density gradients, and unsteady Body Force Model responses ($S$).
  - High-order spatial discretization via curvilinear metric transformations and spectral collocation (Chebyshev / Lagrange polynomials).
  - Handles multi-block domains (`SunModelMultiBlock`) with interface matching conditions (duct $\to$ rotor $\to$ stator $\to$ duct).
  - Generalized eigenvalue problem $P \hat{\Phi} = \omega Y \hat{\Phi}$ solved with the Arnoldi iteration (`scipy.sparse.linalg.eigs`).
  - Spatial eigenmode extraction (`Eigenmode`) to visualize acoustic vs. hydrodynamic modes and stall precursors.
- **Key Classes**:
  - `SunModel`: Single-block operator assembly and boundary conditions.
  - `SunModelMultiBlock`: Multi-block assembly, interface matching, and EVP solver.
  - `SunGrid`: Metric Jacobians and spectral differentiation matrices.
- **Reference Guide**: [docs/sun.md](docs/sun.md)

---

### 4. `unsflow.senoo`
*Centrifugal compressor vaneless diffuser boundary layer and rotating stall inception model.*

- **Theoretical Basis**: Senoo, Kinoshita & Ishida (1977) [5] and Senoo & Kinoshita (1977).
- **Core Features**:
  - Solves the coupled momentum integral equations governing asymmetric turbulent boundary layer development along the hub and shroud walls under adverse radial pressure gradients.
  - Evaluates cross-flow velocity profiles, wall shear stresses, core flow deceleration, and static pressure rise coefficient $C_p(r)$.
  - Precomputes analytical Beta-function integrals for high computational efficiency.
  - Detects wall reverse flow ($\tan\alpha_w \le 0$ or $v_r < 0$) and calculates the **critical flow angle $\alpha_{cr}$** marking the onset of rotating stall.
- **Key Classes**:
  - `SenooConfig`: Dataclass with factory methods for dimensional (`from_dimensional`) and non-dimensional (`from_nondimensional`) inputs.
  - `SenooDiffuser`: Radial marching solver.
  - `SenooFlowField`: Complete 2D solution container ($v_r, v_\theta, V, \alpha, C_p, \delta_1, \delta_2$).
- **Reference Guide**: [docs/senoo.md](docs/senoo.md)

---

### 5. `unsflow.grid`
*Turbomachinery meridional grid generation, blade reconstruction, and Body Force Modeling.*

- **Core Features**:
  - **Grid Generation**: Algebraic Transfinite Interpolation (TFI) and Elliptic grid smoothing with Thomas–Middlecoff boundary orthogonality.
  - **Blade Reconstruction**: Reads discrete blade curves, extracts camberline $\theta(m, r)$, thickness $t(m, r)$, metal angles $\beta(m, r)$, lean angles, gas path angles, blockage $b(m, r)$, and camber normal vectors $\mathbf{n}$. Supports main blades and splitter blades.
  - **Multi-Block Topologies**: Stacks inlet ducts, blade rows, axial gaps, and diffusers into coherent structured multi-block domains.
  - **Body Force Modeling (BFM)**: Calculates normal (turning) and parallel (loss) body forces (Hall–Thollet, Chima models) and calibrates model coefficients ($\beta_0, K_p$).
  - **3D CFD Processing**: Circumferential averaging of 3D Navier-Stokes CFD flow fields onto 2D meridional distributions (area-weighted and Favre averaging).
  - **Mesh Exporter**: Generates native input meshes for **CTurboBFM** and **SU2** (`.su2`).
- **Key Classes**:
  - `MultiBlockGridDriver`: Master driver for grid generation and blade property calculation.
  - `Blade`: 3D blade parameterization and normal vector calculation.
  - `Block` & `MultiBlock`: Single block and multi-block structured grid handlers.
  - `generate_SU2mesh`: SU2 format mesh exporter.
- **Reference Guide**: [docs/grid.md](docs/grid.md)

---

### 6. `unsflow.turbo`
*Aerothermodynamic performance evaluation for ideal and real gases.*

- **Core Features**:
  - Non-dimensional performance metrics: flow coefficient $\Phi$, work coefficient $\Psi$, tip Mach number $M_{tip}$, isentropic efficiency $\eta_{is}$, and temperature ratio $\tau$.
  - Real-gas calculations integrated with **CoolProp** (e.g. for supercritical CO₂, organic refrigerants, or wet gases).
- **Reference Guide**: [docs/turbo_and_utils.md](docs/turbo_and_utils.md)

---

### 7. `unsflow.utils`
*Publication-quality plotting and terminal formatting utilities.*

- **Core Features**:
  - `thesis_plots.py`: Configures Matplotlib to match PhD thesis and journal typography standards (exact LaTeX font rendering, Computer Modern typefaces, and fraction-based page width scaling).
  - `formatting.py`: Clean terminal logging and banners.
  - `plot_styles.py`: Uniform color cycles, markers, and line styles.
- **Reference Guide**: [docs/turbo_and_utils.md](docs/turbo_and_utils.md)

---

## Use Cases & Validation Suite

The repository includes extensive validation cases and real turbomachinery geometries located in [`testcases/`](testcases/):

### Turbomachinery Grid Generation & BFM Test Cases ([`testcases/grid/`](testcases/grid/))

| Case | Machine Configuration | Key Focus |
|---|---|---|
| **`NR37/`** | Transonic Axial Compressor Rotor | NASA Rotor 37 benchmark; single-blade and multiblock grid generation; metal angle validation against laser measurements |
| **`HECC/`** | High Efficiency Centrifugal Compressor | Impeller with main blades and splitter blades; flowpath and full multiblock mesh assembly |
| **`LSCC/`** | NASA Low Speed Centrifugal Compressor | Large-scale subsonic centrifugal impeller; camberline extraction and blade angle verification |
| **`IRIS/`** | Industrial Centrifugal Compressor | Centrifugal stage with vaneless diffuser; single-blade and multiblock setups |
| **`NASA_GER4/`** | Transonic Axial Stage 35 | NASA Stage 35 rotor and stator; separate single-row and full-stage multiblock grids |
| **`TUD-GLRa/`** | 1.5-Stage Axial Research Compressor | TU Darmstadt rig (Rotor, Stator, Outlet Guide Vane OGV); full multistage multiblock grid generation |
| **`ComputeWallDistance/`** | Curved Wall Distance Utility | Computes normal distance to arbitrary curved hub and shroud lines for turbulence / BFM modeling |

### Compression System Stability Test Cases

| Case Directory | Package | Description |
|---|---|---|
| **`testcases/greitzer/moore_greitzer_cases/`** | `greitzer` | Deep surge limit cycle oscillations, rotating stall inception, and stalled characteristic hysteresis loops |
| **`testcases/greitzer/greitzer_cases/`** | `greitzer` | Nonlinear simulation vs. linearized $(B, G)$ stability maps |
| **`testcases/spakovszky/axial_rotor_stator/`** | `spakovszky` | Replication of Exercise 5.4.3 of Spakovszky's thesis (axial rotor-stator stage with varying gap spacing $\Delta x$) |
| **`testcases/spakovszky/axial_rotor_stator_root_locus/`** | `spakovszky` | Root locus tracing across operating flow coefficients |
| **`testcases/spakovszky/iris/`** | `spakovszky` | Stability assessment of the IRIS centrifugal compressor across 4 speedlines (`speedline_0` to `3`) |
| **`testcases/senoo/replicate_senoo_1977/`** | `senoo` | Exact replication of Senoo et al. (1977) experimental blower rig: Fig. 7 & 9 (unstalled), Fig. 8 & 10 (reverse flow stall), and critical flow angle $\alpha_{cr}$ search |
| **`testcases/senoo/hecc_peak_eff/`** | `senoo` | Vaneless diffuser stability analysis of the HECC centrifugal compressor at peak efficiency |
| **`testcases/sun/testcases/annular_duct/`** | `sun` | Annular duct acoustic/hydrodynamic eigenvalue verification against analytical Bessel–Hankel dispersion solutions |

---

## Running Tests

`unsflow` includes automated test suites using `pytest`.

To run all automated unit tests and test case verification scripts:

```bash
# Ensure the unsflow conda environment is activated
conda activate unsflow

# Run full test suite
pytest
```

To run specific test modules:
```bash
# Run vaneless diffuser tests
pytest testcases/senoo/test_senoo_model.py

# Run all 19 turbomachinery grid generation test cases
pytest testcases/grid/test_run.py
```

---

## Repository Structure

```text
Unsflow/
├── LICENSE.md                      # MIT License
├── README.md                       # Main documentation portal (this file)
├── environment.yml                 # Conda environment definition
├── pyproject.toml                  # Python package configuration
├── docs/                           # In-depth package documentation
│   ├── greitzer.md                 # 1D/2D lumped-parameter stability guide
│   ├── spakovszky.md               # Small-perturbation transfer matrix guide
│   ├── sun.md                      # Sun generalized eigenvalue inception guide
│   ├── senoo.md                    # Senoo vaneless diffuser stability guide
│   ├── grid.md                     # Meridional grid & BFM guide
│   └── turbo_and_utils.md          # Aerothermodynamics & plotting guide
├── src/
│   └── unsflow/                    # Core Python library
│       ├── greitzer/               # Greitzer & Moore–Greitzer models
│       ├── spakovszky/             # Spakovszky dynamic transfer function framework
│       ├── sun/                    # Sun general flow instability solver
│       ├── senoo/                  # Senoo vaneless diffuser stall solver
│       ├── grid/                   # Grid generation, blade reconstruction & BFM
│       ├── turbo/                  # Turbomachinery performance formulas
│       └── utils/                  # Thesis plotting styles and formatting
└── testcases/                      # Benchmarks, test cases, and validation scripts
    ├── greitzer/                   # Surge, stall, and stability map cases
    ├── spakovszky/                 # Rotor-stator stage, root locus, IRIS cases
    ├── sun/                        # Annular duct eigenvalue benchmarks & notebooks
    ├── senoo/                      # Senoo 1977 replication & HECC diffuser cases
    └── grid/                       # NR37, HECC, LSCC, IRIS, GER4, TUD-GLRa grids
```

---

## References & Citation

If you use `unsflow` in your research, thesis, or publications, please cite the underlying models and the framework:

1. **Greitzer, E. M.** (1976). *Surge and Rotating Stall in Axial Flow Compressors—Part I: Theoretical Compression System Model*. ASME Journal of Engineering for Power, 98(2), 190–198.
2. **Moore, F. K., & Greitzer, E. M.** (1986). *A Theory of Post-Stall Transients in Axial Compression Systems: Part I—Development of Equations*. ASME Journal of Engineering for Gas Turbines and Power, 108(1), 68–76.
3. **Spakovszky, Z. S.** (2000). *Applications of Axial and Radial Compressor Dynamic System Modeling*. Ph.D. Thesis, Massachusetts Institute of Technology.
4. **Sun, X., Sun, D., & Shen, X.** (2013). *A General Theory of Flow-Instability Inception in Turbomachinery*. AIAA Journal, 51(7), 1675–1687.
5. **Senoo, Y., Kinoshita, Y., & Ishida, M.** (1977). *Asymmetric Flow in Vaneless Diffusers of Centrifugal Blowers*. ASME Journal of Fluids Engineering, 99(1), 104–111.
6. **Thollet, F., et al.** (2017). *Aerodynamic Performance Prediction of Turbomachinery With a Body Force Method*. ASME Journal of Turbomachinery, 139(2).

```bibtex
@software{unsflow,
  author       = {Francesco Neri},
  title        = {unsflow: Advanced Compression System Instability Analysis & Turbomachinery Modeling Suite},
  year         = {2026},
  url          = {https://github.com/fneri/unsflow}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.