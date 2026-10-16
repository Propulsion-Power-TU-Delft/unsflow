"""
unsflow.senoo
=============
Vaneless diffuser stability, rotating stall, and surge prediction model
based on Senoo & Kinoshita (ASME 1977, 1978).
"""

from .config import SenooConfig
from .results import SenooResults, SenooFlowField
from .map import senoo_solution_map
from .senoo_diffuser import SenooDiffuser

__all__ = [
    "SenooConfig",
    "SenooDiffuser",
    "SenooResults",
    "SenooFlowField",
    "senoo_solution_map",
]
