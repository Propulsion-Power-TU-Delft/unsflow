# `unsflow.grid`: Meridional Grid Generation, Blade Reconstruction, and Body Force Modeling

The `unsflow.grid` module is a comprehensive geometry and grid generation suite designed for turbomachinery simulations, CFD post-processing, and **Body Force Modeling (BFM)**. 

It generates structured 2D meridional grids, reconstructs 3D blade surfaces and aerodynamic properties from CAD / coordinate curves, and exports computational meshes directly for **[CTurboBFM](https://github.com/Propulsion-Power-TU-Delft/CTurboBFM)** and the **SU2** open-source CFD solver.

---

## 1. Core Capabilities

```
+-----------------------------------------------------------------------------+
|                               unsflow.grid                                  |
|                                                                             |
|  +---------------------+  +---------------------+  +---------------------+  |
|  |     Meridional      |  |   Blade Geometry    |  |     Body Force      |  |
|  |   Mesh Generation   |  |   Reconstruction    |  |      Modeling       |  |
|  |  * TFI & Elliptic   |  |  * Camber & Thk     |  |  * Hall-Thollet     |  |
|  |  * Sigmoid cluster  |  |  * Metal / Lean ang |  |  * Turning & Loss   |  |
|  |  * Multiblock stack |  |  * Main & Splitter  |  |  * Normal vectors   |  |
|  +----------+----------+  +----------+----------+  +----------+----------+  |
|             |                        |                        |             |
|             +------------------------+------------------------+             |
|                                      |                                      |
|                       +--------------v---------------+                      |
|                       |   Mesh Exporters & Solvers   |                      |
|                       |   * CTurboBFM grid files     |                      |
|                       |   * SU2 mesh (.su2 format)   |                      |
|                       |   * 3D CFD Circumferential   |                      |
|                       |     Averaging (Favre/Area)   |                      |
|                       +------------------------------+                      |
+-----------------------------------------------------------------------------+
```

### 1.1 Meridional Grid Generation
- **Transfinite Interpolation (TFI)**: Rapid algebraic grid generation conforming to complex curved hub and shroud contours.
- **Elliptic Smoothing**: Solves Poisson / Laplace elliptic grid equations with Thomas–Middlecoff boundary orthogonality control.
- **Parametric Clustering**: Uses Eriksson / sigmoid stretching functions to cluster grid lines near hub/shroud boundary layers and blade leading/trailing edges.

### 1.2 Blade Geometry Reconstruction (`Blade`)
- Reconstructs 3D blade geometries from discrete coordinate point clouds (`.curve` format):
  - Camber surface $\theta_{camber}(m, r)$ and thickness distribution $t(m, r)$.
  - Leading and trailing edge detection and curvature smoothing.
  - Blade metal angle $\beta_{metal}(m, r)$, blade lean angle, and gas path angle.
  - Aerodynamic blade blockage factor $b(m, r) = 1 - t / s$ (where $s$ is blade pitch).
  - Normal unit vectors $\mathbf{n} = (n_z, n_r, n_\theta)$ perpendicular to the blade camber surface.
- Supports both **main blades** and **splitter blades** (essential for high-pressure ratio centrifugal compressors).

### 1.3 Multi-Block Topologies (`MultiBlock`, `MultiBlockGridDriver`)
- Assembles multi-component domains:
  - `single_blade`: Isolated blade row with inlet and outlet domains.
  - `multiblock`: Alternating bladed and unbladed blocks (e.g. inlet duct $\to$ rotor $\to$ gap $\to$ stator $\to$ diffuser).
  - `full_machine`: Full stage or multi-stage continuous domains.
- Enforces strict grid continuity, slope matching, and interface boundary conditions.

### 1.4 Body Force Modeling (BFM) & Solvers Export
- Computes body force source terms $(\mathbf{f}_n, \mathbf{f}_p)$ representing blade aerodynamic turning and viscous entropy generation:
  - Implements the Hall–Thollet and Chima BFM formulations.
  - Calibrates turning coefficients $\beta_0$ and loss coefficients $K_p$.
- **Solver Interfaces**:
  - Direct export of grid and blade property files for the **CTurboBFM** solver.
  - Native export of structured grids to **SU2** format (`.su2`) via [`su2_mesh_generator.py`](../src/unsflow/grid/su2_mesh_generator.py).

### 1.5 3D CFD Circumferential Averaging
- Module [`circumferential_average.py`](../src/unsflow/grid/circumferential_average.py) processes 3D full-annulus or single-passage RANS solutions, projecting flow quantities onto the 2D axisymmetric meridional plane using:
  - Area-weighted averaging
  - Mass-weighted / Favre averaging for compressible flow variables ($\rho, \mathbf{u}, h_t, s$)

---

## 2. Module Architecture

Implemented in [`src/unsflow/grid/`](../src/unsflow/grid/):

| File | Primary Classes / Functions | Description |
|---|---|---|
| [`config.py`](../src/unsflow/grid/config.py) | `Config` | Parses grid dimensions, domain topologies, blade rows, and solver settings from INI files |
| [`multiblock_grid_driver.py`](../src/unsflow/grid/multiblock_grid_driver.py) | `MultiBlockGridDriver` | Orchestrates geometry reconstruction, grid generation, blade properties, and multi-block assembly |
| [`blade.py`](../src/unsflow/grid/blade.py) | `Blade` | Camberline, thickness, metal angles, blockage, and normal vectors calculation |
| [`block.py`](../src/unsflow/grid/block.py) | `Block` | Single structured 2D grid block (coordinates, metrics, stretching, boundary markers) |
| [`multiblock.py`](../src/unsflow/grid/multiblock.py) | `MultiBlock` | Connects, stitches, and maps properties across multi-block configurations |
| [`body_force.py`](../src/unsflow/grid/body_force.py) | `BodyForce` | BFM source term formulation, blade forces, and calibration |
| [`su2_mesh_generator.py`](../src/unsflow/grid/su2_mesh_generator.py) | `generate_SU2mesh()` | Exporter writing structured 2D meshes to SU2 format (`.su2`) |
| [`circumferential_average.py`](../src/unsflow/grid/circumferential_average.py) | `CircumferentialAverage` | 3D RANS/URANS CFD circumferential averaging to meridional distributions |
| [`functions.py`](../src/unsflow/grid/functions.py) | Grid Utilities | Spline fitting, Eriksson stretching, minimum wall distance, metric Jacobians |

---

## 3. Configuration File Format (`input.ini`)

```ini
[CFD PROCESSING]
GAMMA_FLUID = 1.4
NORMALIZE_DATA = False

[GRID GENERATION]
; Generation algorithm: TFI or elliptic
MESH_GENERATION_METHOD = TFI
; Streamwise points per block (e.g., 3 blocks)
STREAMWISE_POINTS = 31, 41, 31
; Spanwise points across flow passage
SPANWISE_POINTS = 41
; Stretching coefficients (Sigmoid / Eriksson)
SIGMOID_STREAM_COEFFICIENTS = 2.0, 1.0, 2.0
SIGMOID_SPAN_COEFFICIENT = 1.0
; Coordinates units in input files (cm, m, in)
COORDINATES_FILE_UNITS = cm
HUB_COORDINATES_FILEPATH = coordinates/hub.curve
SHROUD_COORDINATES_FILEPATH = coordinates/shroud.curve
BLOCKS_TRIM_TYPE = axial, axial, axial

[TURBOMACHINERY DATA]
; Driver type: multiblock, single_blade, or full_machine
MULTIBLOCK_DRIVER_TYPE = multiblock
; Domain type for each block
BLOCKS_TYPE = unbladed, rotor, unbladed
BLOCKS_TOPOLOGY = inlet, internal, outlet
; Shaft rotational speed [rpm]
SHAFT_RPM = 0, -17189, 0
; Number of blade rows
BLADE_ROWS_NUMBER = 1
BLADE_INLET_TYPE = axial
BLADE_OUTLET_TYPE = axial
```

---

## 4. Usage Example

```python
from unsflow.grid.config import Config
from unsflow.grid.multiblock_grid_driver import MultiBlockGridDriver

# 1. Initialize configuration
config = Config('input.ini')

# 2. Build MultiBlock driver
driver = MultiBlockGridDriver(config)

# 3. Execute generation pipeline
driver.GenerateGrid()             # Generates meridional mesh coordinates
driver.ComputeBladesData()        # Reconstructs camber, thickness, normals, metal angles
driver.AssembleMultiBlockGrid()   # Stitches blocks and interpolates blade data
driver.SaveOutput()               # Exports pickle files and diagnostic figures
```

---

## 5. Included Turbomachinery Benchmarks & Test Cases

Located in [`testcases/grid/`](../testcases/grid/):

| Case / Directory | Machine Type | Key Features |
|---|---|---|
| **`NR37/`** | Transonic Axial Compressor Rotor | NASA Rotor 37 benchmark; single-blade and multiblock grid generation; metal angle validation against laser measurements |
| **`HECC/`** | High Efficiency Centrifugal Compressor | Impeller with main blades and splitter blades; flowpath and full multiblock mesh assembly |
| **`LSCC/`** | NASA Low Speed Centrifugal Compressor | Large-scale subsonic centrifugal impeller; camberline extraction and blade angle verification |
| **`IRIS/`** | Industrial Centrifugal Compressor | Centrifugal stage with vaneless diffuser; single-blade and multiblock setups |
| **`NASA_GER4/`** | Transonic Axial Stage 35 | NASA Stage 35 rotor and stator; separate single-row and full-stage multiblock grids |
| **`TUD-GLRa/`** | 1.5-Stage Axial Research Compressor | TU Darmstadt rig (Rotor, Stator, Outlet Guide Vane OGV); full multistage multiblock grid generation |
| **`ComputeWallDistance/`** | Curved Wall Distance Utility | Computes normal distance to arbitrary curved hub and shroud lines for turbulence / BFM modeling |

All 19 grid generation benchmarks can be executed automatically via pytest:
```bash
pytest testcases/grid/test_run.py
```

---

## References

- **Chima, R. V.** (1998). *Calculation of Multistage Turbomachinery Using Body Force Models*. AIAA Paper 98-0968.
- **Hall, C. A., Greitzer, E. M., & Tan, C. S.** (2000). *Analysis of Low-Order Body Force Models for Turbomachinery*. ASME Turbo Expo.
- **Thollet, F., et al.** (2017). *Aerodynamic Performance Prediction of Turbomachinery With a Body Force Method*. ASME Journal of Turbomachinery, 139(2).
