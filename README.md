# unsflow

`unsflow` is a Python package developed to support the analysis of compressor instabilities and the post-processing of CFD simulations.

The framework integrates multiple reduced-order stability models with dedicated tools for geometry handling and data analysis, enabling reproducible and research-oriented workflows for turbomachinery applications.

---

## Overview

The package is organized into modular components, each focused on a specific modeling or post-processing task.

### Modules

#### `greitzer`

Implementation of the classical lumped-parameter compressor instability models:

- Greitzer model
- Moore–Greitzer model

These models are used for low-order dynamic analysis of surge and rotating stall phenomena.

---

#### `spakovszky`

Tools for compressor stability assessment based on the Spakovszky stability framework.

This module enables linear stability analysis and modal interpretation of compressor dynamics.

---

#### `sun`

Implementation of the Sun stability model for the analysis of flow instabilities.

The model is formulated using circumferentially averaged equations with body-force representations and is suitable for modal and reduced-order investigations of compressor behavior.

---

#### `grid`

Utilities for:

- Meridional grid generation  
- Blade geometry reconstruction  
- CFD data post-processing  
- Flow-field manipulation and analysis  

This module supports both research-grade preprocessing and advanced post-processing workflows.

---

## Installation

* Go to the root folder and create a new python environment (unsflow) through the .yml file:
```bash
conda env create -f unsflow_env.yml
```

* Activate the environment:
```bash
conda activate pyshockflow
```

* Install the packages:
```bash
python -m pip install -e .
```

* Check the testcases folder, and look for the run scripts that you are interested in