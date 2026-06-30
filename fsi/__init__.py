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
    from .output import SimulationOutputWriter
    from .postprocess import (
        SnapshotInfo,
        extract_snapshot_timeseries,
        list_npz_snapshots,
        load_validation_summary,
        validation_summary_table,
    )
    from .simulation import FSISimulation
    from .validation import ValidationMetric, ValidationReport
    from .validation_cases import run_validation_suite

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
    "SimulationOutputWriter",
    "SnapshotInfo",
    "ValidationMetric",
    "ValidationReport",
    "extract_snapshot_timeseries",
    "list_npz_snapshots",
    "load_validation_summary",
    "run_validation_suite",
    "validation_summary_table",
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
    if name == "SimulationOutputWriter":
        from .output import SimulationOutputWriter

        globals()[name] = SimulationOutputWriter
        return SimulationOutputWriter
    if name == "SnapshotInfo":
        from .postprocess import SnapshotInfo

        globals()[name] = SnapshotInfo
        return SnapshotInfo
    if name == "list_npz_snapshots":
        from .postprocess import list_npz_snapshots

        globals()[name] = list_npz_snapshots
        return list_npz_snapshots
    if name == "extract_snapshot_timeseries":
        from .postprocess import extract_snapshot_timeseries

        globals()[name] = extract_snapshot_timeseries
        return extract_snapshot_timeseries
    if name == "load_validation_summary":
        from .postprocess import load_validation_summary

        globals()[name] = load_validation_summary
        return load_validation_summary
    if name == "validation_summary_table":
        from .postprocess import validation_summary_table

        globals()[name] = validation_summary_table
        return validation_summary_table
    if name == "ValidationMetric":
        from .validation import ValidationMetric

        globals()[name] = ValidationMetric
        return ValidationMetric
    if name == "ValidationReport":
        from .validation import ValidationReport

        globals()[name] = ValidationReport
        return ValidationReport
    if name == "run_validation_suite":
        from .validation_cases import run_validation_suite

        globals()[name] = run_validation_suite
        return run_validation_suite
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
