"""FSI LBM-MPM package.

This package will provide a 3D two-way fluid-solid coupling simulator using:
- LBM for fluid
- MPM for deformable solid
- Taichi kernels for high-performance computation
"""

from .config import (
    BoundaryConfig,
    CouplingConfig,
    LBMConfig,
    MPMConfig,
    OutputConfig,
    SimulationConfig,
)

__all__ = [
    "BoundaryConfig",
    "CouplingConfig",
    "LBMConfig",
    "MPMConfig",
    "OutputConfig",
    "SimulationConfig",
]
