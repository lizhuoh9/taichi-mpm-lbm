"""FSI LBM-MPM package.

This package will provide a 3D two-way fluid-solid coupling simulator using:
- LBM for fluid
- MPM for deformable solid
- Taichi kernels for high-performance computation
"""

from typing import TYPE_CHECKING

from .config import (
    BoundaryConfig,
    CouplingConfig,
    LBMConfig,
    MPMConfig,
    OutputConfig,
    SimulationConfig,
)

if TYPE_CHECKING:
    from .coupling import LBMMpmCoupler
    from .lbm3d import LBMSolver3D
    from .mpm3d import MPMSolver3D
    from .simulation import FSISimulation

__all__ = [
    "BoundaryConfig",
    "CouplingConfig",
    "FSISimulation",
    "LBMConfig",
    "LBMMpmCoupler",
    "LBMSolver3D",
    "MPMConfig",
    "MPMSolver3D",
    "OutputConfig",
    "SimulationConfig",
]


def __getattr__(name: str):
    if name == "LBMSolver3D":
        from .lbm3d import LBMSolver3D

        globals()[name] = LBMSolver3D
        return LBMSolver3D
    if name == "MPMSolver3D":
        from .mpm3d import MPMSolver3D

        globals()[name] = MPMSolver3D
        return MPMSolver3D
    if name == "LBMMpmCoupler":
        from .coupling import LBMMpmCoupler

        globals()[name] = LBMMpmCoupler
        return LBMMpmCoupler
    if name == "FSISimulation":
        from .simulation import FSISimulation

        globals()[name] = FSISimulation
        return FSISimulation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
