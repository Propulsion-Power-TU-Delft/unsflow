# unsflow

Python package developed to support the analysis of compressor instabilities and the post-processing of CFD simulations.

The framework integrates multiple reduced-order stability models with dedicated tools for geometry handling and data analysis, enabling reproducible and research-oriented workflows for turbomachinery applications.
The grid generation module can build grid files for [CTurboBFM](https://github.com/Propulsion-Power-TU-Delft/CTurboBFM)

## Overview

The package is organized into modular components, each focused on a specific modeling or post-processing task:

* src/unsflow       → source files

* testcases/          → running scripts

### Content of src/unsflow

* `greitzer`: implementation of lumped-parameter compressor stability models: Greitzer model [1] and Moore–Greitzer model [2]

* `spakovszky`: tools for compressor stability assessment based on the Spakovszky stability framework [3]


* `sun`: Implementation of the Sun stability model for the analysis of flow instabilities [4]

* `grid`: utilities for meridional grid generation, blade geometry reconstruction, CFD post-processing and generation of grid files for [CTurboBFM](https://github.com/Propulsion-Power-TU-Delft/CTurboBFM) BFM solver


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

## Use

* Go to the testcases folder, and look for the run scripts that you are interested in:
```bash
python main.py
```



### References ###

[1] Greitzer, Edward M. "Surge and rotating stall in axial flow compressors—Part I: Theoretical compression system model." (1976): 190-198.

[2] Moore, Franklin K., and Edward Marc Greitzer. "A theory of post-stall transients in axial compression systems: Part I—Development of equations." (1986): 68-76.

[3] Spakovszky, Zoltán Sándor. Applications of axial and radial compressor dynamic system modeling. Diss. Massachusetts Institute of Technology, 2000.

[4] Sun, Xiaofeng, et al. "A general theory of flow-instability inception in turbomachinery." AIAA journal 51.7 (2013): 1675-1687.