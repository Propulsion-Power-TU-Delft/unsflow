# `unsflow.sun`: Flow Instability Inception Theory (Sun Model)

The `unsflow.sun` module provides an implementation of the flow-instability inception theory developed by **Xiaofeng Sun and co-workers (AIAA Journal, 2013)** [1]. 

Unlike lumped-parameter or 1D/actuator disk approaches, the Sun model directly solves the linearized compressible Euler perturbation equations on two-dimensional meridional grids extracted from CFD mean flow fields or Body Force Model (BFM) simulations. It computes the complex eigenvalues (growth rates and frequencies) and spatial mode shapes of acoustic and hydrodynamic disturbances.

---

## 1. Theoretical Background

Small-amplitude unsteady perturbations $\Phi' = [\rho', u_r', u_\theta', u_z', p']^T$ superimposed on a non-uniform steady axisymmetric background flow are expanded in circumferential Fourier harmonics with azimuthal wavenumber $m$:

$$\Phi'(r, \theta, z, t) = \hat{\Phi}(r, z) e^{-j (\omega t - m \theta)}$$

where:
- $\omega = \omega_r + j \omega_i$ is the complex frequency:
  - $\omega_r = \text{Re}(\omega)$: disturbance frequency in rad/s.
  - $\omega_i = \text{Im}(\omega)$: temporal growth rate ($\omega_i > 0$ indicates dynamic instability).
- $m$: circumferential harmonic wavenumber.

### 1.1 Governing Perturbation System

Substituting the harmonic ansatz into the compressible Euler equations yields the linearized system:

$$\left( -j \omega A + B \frac{\partial}{\partial r} + j \frac{m}{r} C + E \frac{\partial}{\partial z} + R + S \right) \hat{\Phi} = 0$$

where:
- **$A$**: Coefficient matrix of temporal derivatives.
- **$B$**: Coefficient matrix of radial spatial derivatives.
- **$C$**: Coefficient matrix of circumferential derivatives (scaled by $j m / r$).
- **$E$**: Coefficient matrix of axial spatial derivatives.
- **$R$**: Mean flow convection, velocity gradients, and density variation matrix.
- **$S$**: Body Force Model (BFM) perturbation coupling matrix (representing the linearized blade row force response to local velocity/angle variations).

### 1.2 Coordinate Transformation & Discretization

The physical domain $(z, r)$ is mapped onto a curvilinear computational domain $(\xi, \eta)$ via the Jacobian matrix computed by `SunGrid`:

$$\frac{\partial}{\partial z} = \xi_z \frac{\partial}{\partial \xi} + \eta_z \frac{\partial}{\partial \eta}, \qquad \frac{\partial}{\partial r} = \xi_r \frac{\partial}{\partial \xi} + \eta_r \frac{\partial}{\partial \eta}$$

Spatial derivatives are discretized using high-order finite difference operators or spectral collocation (Chebyshev / Lagrange polynomials).

### 1.3 Generalized Eigenvalue Problem (EVP)

Applying the boundary conditions (Euler slip-wall at hub/casing, acoustic absorbing or specified impedance at inlet/outlet) and assembling all nodes produces the generalized eigenvalue problem:

$$P \hat{\Phi} = \omega Y \hat{\Phi}$$

For complex multi-row turbomachinery domains, `SunModelMultiBlock` enforces conservative interface matching conditions across block interfaces:

$$\hat{\Phi}_{block, 1}^{(exit)} = \hat{\Phi}_{block, 2}^{(inlet)}$$

The large sparse system is solved for the leading eigenvalues using the Krylov-subspace Arnoldi algorithm (`scipy.sparse.linalg.eigs`).

---

## 2. Module Architecture

Implemented in [`src/unsflow/sun/`](../src/unsflow/sun/):

| File | Class / Functions | Description |
|---|---|---|
| [`config.py`](../src/unsflow/sun/config.py) | `Config` | Parses multi-block settings, grid input files, fluid properties ($\gamma$), and EVP solver parameters |
| [`sun_grid.py`](../src/unsflow/sun/sun_grid.py) | `SunGrid` | Handles grid metrics, unit normals, Jacobian transformations, and spectral nodes |
| [`node.py`](../src/unsflow/sun/node.py) | `Node` | Local node state and matrix equation assembly container |
| [`sun_model.py`](../src/unsflow/sun/sun_model.py) | `SunModel` | Assembles single-block matrices ($A, B, C, E, R, S$), implements boundary conditions |
| [`sun_model_multiblock.py`](../src/unsflow/sun/sun_model_multiblock.py) | `SunModelMultiBlock` | Connects multi-block topologies, enforces interface matching, assembles $(P, Y)$ matrices, and solves the EVP |
| [`eigenmode.py`](../src/unsflow/sun/eigenmode.py) | `Eigenmode` | Encapsulates spatial eigenmode fields ($\rho', u_r', u_\theta', u_z', p'$) and modal energy |
| [`sun_post_process.py`](../src/unsflow/sun/sun_post_process.py) | `SunPostProcess` | Generates 2D contour maps of eigenmode shapes and modal spectra |
| [`general_functions.py`](../src/unsflow/sun/general_functions.py) | Spectral operators | Lagrange polynomials, differentiation matrices, and grid interpolation utilities |

---

## 3. Configuration File Format (`input.ini`)

```ini
[GENERAL]
; Number of grid blocks (e.g., inlet duct, rotor, stator, outlet duct)
NUMBER_OF_BLOCKS = 1

; Type of domain (unbladed, rotor, stator)
BLOCK_TYPES = unbladed

; Fluid ratio of specific heats
GAMMA = 1.4

; Harmonic wavenumber m
HARMONIC_ORDER = 1

; Input grid / flowfield pickle files
GRID_FILES = grid_block_0.pkl

[EVP SOLVER]
; Number of eigenvalues to calculate around target shift
NUMBER_EIGENVALUES = 20
TARGET_SHIFT_REAL = 100.0
TARGET_SHIFT_IMAG = 0.0
TOLERANCE = 1e-6
```

---

## 4. Usage Workflow

```python
from unsflow.sun.config import Config
from unsflow.sun.sun_model import SunModel
from unsflow.sun.sun_model_multiblock import SunModelMultiBlock

# 1. Load configuration
config = Config('input.ini')
nBlocks = config.getNumberOfBlocks()

# 2. Assemble each domain block
sunBlocks = []
for iblock in range(nBlocks):
    block = SunModel(config, iblock)
    block.ComputeBoundaryNormals()
    block.ComputeSpectralGrid()
    block.ComputeJacobianPhysical()
    
    # Assemble governing linearized operators
    block.AddAMatrixToNodes_francesco()
    block.AddBMatrixToNodes_francesco()
    block.AddCMatrixToNodes_francesco()
    block.AddEMatrixToNodes_francesco()
    block.AddRMatrixToNodes_francesco()
    block.AddSMatrixToNodes()
    block.ApplySpectralDifferentiation()
    
    # Global block matrix assembly & boundary conditions
    block.build_A_global_matrix()
    block.build_C_global_matrix()
    block.build_R_global_matrix()
    block.build_S_global_matrix()
    block.build_Z_global_matrix()
    block.compute_L_matrices()
    block.apply_boundary_conditions_generalized()
    sunBlocks.append(block)

# 3. Multi-block assembly & interface matching
multiBlock = SunModelMultiBlock(sunBlocks, config)
multiBlock.construct_L_global_matrices()
multiBlock.apply_matching_conditions()
multiBlock.compute_P_Y_matrices()

# 4. Solve Generalized Eigenvalue Problem
multiBlock.solve_evp()
multiBlock.extract_eigenfields()
multiBlock.SaveOutput()
```

---

## 5. Included Test Cases

Located in [`testcases/sun/`](../testcases/sun/):

1. **`testcases/annular_duct/single_block/normal_geometry/`**:
   - Canonical annular duct eigenvalue benchmark.
   - Compares numerical acoustic eigenfrequencies against exact analytical Bessel–Hankel dispersion solutions (`analytical/main.py`).
   - Workflow:
     ```bash
     python 01_GenerateData.py
     python 02_Run.py
     python 03_PostProcess.py
     ```
2. **`testcases/annular_duct/single_block/alternative_geometry/`**:
   - Curvilinear distorted annular duct geometry validating grid metric transformations.
3. **`testcases/annular_duct/comparison/`**:
   - Jupyter notebook (`Post_Process.ipynb`) validating convergence and spectral resolution vs. analytical acoustic solutions.
4. **`notebooks/`**:
   - `Arnoldi Algorithm Hermitian.ipynb` & `Arnoldi Algorithm Not-Hermitian.ipynb`: Algorithmic validation of the Arnoldi iteration for non-Hermitian turbomachinery flow operators.
   - `Lagrange Polynomials.ipynb`: Verification of Gauss–Lobatto collocation nodes and Lagrange differentiation matrices.

---

## References

- **[1] Sun, X., Sun, D., & Shen, X.** (2013). *A General Theory of Flow-Instability Inception in Turbomachinery*. AIAA Journal, 51(7), 1675–1687.
