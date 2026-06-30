from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Backend = Literal["cpu", "gpu", "cuda", "vulkan"]
BoundaryType = Literal["periodic", "pressure", "velocity", "wall"]
CouplingKernel = Literal["quadratic"]
OutputFormat = Literal["npz", "vtk", "both"]

_BACKENDS = {"cpu", "gpu", "cuda", "vulkan"}
_BOUNDARY_TYPES = {"periodic", "pressure", "velocity", "wall"}
_COUPLING_KERNELS = {"quadratic"}
_OUTPUT_FORMATS = {"npz", "vtk", "both"}
_MATERIALS = {"elastic"}


@dataclass(frozen=True)
class BoundaryConfig:
    """Boundary condition configuration for one side of the LBM domain."""

    boundary_type: BoundaryType = "periodic"
    rho: float = 1.0
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        if self.boundary_type not in _BOUNDARY_TYPES:
            raise ValueError(f"Unsupported boundary type: {self.boundary_type}.")
        if self.rho <= 0.0:
            raise ValueError(f"Boundary rho must be positive, got {self.rho}.")
        if len(self.velocity) != 3:
            raise ValueError("Boundary velocity must have 3 components.")


@dataclass(frozen=True)
class LBMConfig:
    """Configuration for the 3D LBM fluid solver."""

    nx: int = 64
    ny: int = 32
    nz: int = 16

    viscosity: float = 0.1
    rho0: float = 1.0

    force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    x_left: BoundaryConfig = field(default_factory=lambda: BoundaryConfig("velocity"))
    x_right: BoundaryConfig = field(default_factory=lambda: BoundaryConfig("pressure"))
    y_left: BoundaryConfig = field(default_factory=BoundaryConfig)
    y_right: BoundaryConfig = field(default_factory=BoundaryConfig)
    z_left: BoundaryConfig = field(default_factory=BoundaryConfig)
    z_right: BoundaryConfig = field(default_factory=BoundaryConfig)

    use_sparse: bool = False

    def validate(self) -> None:
        if self.nx <= 4 or self.ny <= 4 or self.nz <= 4:
            raise ValueError(
                f"LBM grid is too small: {(self.nx, self.ny, self.nz)}. "
                "Each dimension should be greater than 4."
            )
        if self.viscosity <= 0.0:
            raise ValueError(f"LBM viscosity must be positive, got {self.viscosity}.")
        if self.rho0 <= 0.0:
            raise ValueError(f"LBM rho0 must be positive, got {self.rho0}.")
        if len(self.force) != 3:
            raise ValueError("LBM force must have 3 components.")
        if len(self.initial_velocity) != 3:
            raise ValueError("LBM initial_velocity must have 3 components.")

        for boundary in (
            self.x_left,
            self.x_right,
            self.y_left,
            self.y_right,
            self.z_left,
            self.z_right,
        ):
            boundary.validate()


@dataclass(frozen=True)
class MPMConfig:
    """Configuration for the 3D MPM solid solver."""

    nx: int = 64
    ny: int = 32
    nz: int = 16

    max_particles: int = 200_000
    particles_per_cell: int = 2

    dx: float = 1.0
    dt: float = 0.25

    density: float = 1.0
    youngs_modulus: float = 1_000.0
    poisson_ratio: float = 0.25

    gravity: tuple[float, float, float] = (0.0, -9.8e-4, 0.0)

    apic: bool = True
    material: Literal["elastic"] = "elastic"
    boundary_width: int = 3
    boundary_damping: float = 0.0

    def validate(self) -> None:
        if self.nx <= 4 or self.ny <= 4 or self.nz <= 4:
            raise ValueError(
                f"MPM grid is too small: {(self.nx, self.ny, self.nz)}. "
                "Each dimension should be greater than 4."
            )
        if self.max_particles <= 0:
            raise ValueError("max_particles must be positive.")
        if self.particles_per_cell <= 0:
            raise ValueError("particles_per_cell must be positive.")
        if self.dx <= 0.0:
            raise ValueError("MPM dx must be positive.")
        if self.dt <= 0.0:
            raise ValueError("MPM dt must be positive.")
        if self.density <= 0.0:
            raise ValueError("MPM density must be positive.")
        if self.youngs_modulus <= 0.0:
            raise ValueError("Young's modulus must be positive.")
        if not (-1.0 < self.poisson_ratio < 0.5):
            raise ValueError(f"Poisson ratio must be in (-1, 0.5), got {self.poisson_ratio}.")
        if len(self.gravity) != 3:
            raise ValueError("MPM gravity must have 3 components.")
        if self.material not in _MATERIALS:
            raise ValueError(f"Unsupported MPM material: {self.material}.")
        if self.boundary_width < 1:
            raise ValueError("boundary_width must be at least 1.")
        if not (0.0 <= self.boundary_damping <= 1.0):
            raise ValueError("boundary_damping must be in [0, 1].")


@dataclass(frozen=True)
class CouplingConfig:
    """Configuration for LBM-MPM two-way coupling."""

    enabled: bool = True

    # Penalty coupling:
    # F_p = gamma * m_p * (u_fluid_at_particle - v_particle)
    gamma: float = 0.05

    mpm_substeps_per_lbm_step: int = 4
    kernel: CouplingKernel = "quadratic"

    build_solid_volume_fraction: bool = True
    force_limit: float | None = None
    relative_velocity_limit: float | None = None
    gamma_ramp_steps: int = 0
    min_valid_weight: float = 1.0e-6
    immersed_boundary_enabled: bool = False
    immersed_boundary_drag: float = 0.0
    immersed_boundary_fraction_threshold: float = 0.1
    immersed_boundary_max_force: float | None = None
    contact_enabled: bool = False
    contact_velocity_damping: float = 0.0
    contact_fraction_threshold: float = 0.5

    def validate(self) -> None:
        if self.gamma < 0.0:
            raise ValueError(f"Coupling gamma must be non-negative, got {self.gamma}.")
        if self.mpm_substeps_per_lbm_step <= 0:
            raise ValueError("mpm_substeps_per_lbm_step must be positive.")
        if self.kernel not in _COUPLING_KERNELS:
            raise ValueError(f"Unsupported coupling kernel: {self.kernel}.")
        if self.force_limit is not None and self.force_limit <= 0.0:
            raise ValueError("force_limit must be positive when provided.")
        if self.relative_velocity_limit is not None and self.relative_velocity_limit <= 0.0:
            raise ValueError("relative_velocity_limit must be positive when provided.")
        if self.gamma_ramp_steps < 0:
            raise ValueError("gamma_ramp_steps must be non-negative.")
        if self.min_valid_weight <= 0.0:
            raise ValueError("min_valid_weight must be positive.")
        if self.min_valid_weight > 1.0:
            raise ValueError("min_valid_weight must not exceed 1.")
        if self.immersed_boundary_drag < 0.0:
            raise ValueError("immersed_boundary_drag must be non-negative.")
        if not (0.0 <= self.immersed_boundary_fraction_threshold <= 1.0):
            raise ValueError("immersed_boundary_fraction_threshold must be in [0, 1].")
        if (
            self.immersed_boundary_max_force is not None
            and self.immersed_boundary_max_force <= 0.0
        ):
            raise ValueError("immersed_boundary_max_force must be positive when provided.")
        if not (0.0 <= self.contact_velocity_damping <= 1.0):
            raise ValueError("contact_velocity_damping must be in [0, 1].")
        if not (0.0 <= self.contact_fraction_threshold <= 1.0):
            raise ValueError("contact_fraction_threshold must be in [0, 1].")


@dataclass(frozen=True)
class OutputConfig:
    """Configuration for simulation output."""

    output_dir: Path = Path("outputs")
    output_interval: int = 100
    output_format: OutputFormat = "npz"

    write_lbm_fields: bool = True
    write_mpm_particles: bool = True
    write_coupling_fields: bool = True

    def validate(self) -> None:
        if self.output_interval <= 0:
            raise ValueError("output_interval must be positive.")
        if self.output_format not in _OUTPUT_FORMATS:
            raise ValueError(f"Unsupported output format: {self.output_format}.")


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level configuration for the coupled simulator."""

    backend: Backend = "cpu"
    random_seed: int = 0

    num_steps: int = 1000
    lbm_dt: float = 1.0

    lbm: LBMConfig = field(default_factory=LBMConfig)
    mpm: MPMConfig = field(default_factory=MPMConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError(f"Unsupported backend: {self.backend}.")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.lbm_dt <= 0.0:
            raise ValueError("lbm_dt must be positive.")

        self.lbm.validate()
        self.mpm.validate()
        self.coupling.validate()
        self.output.validate()

        if (self.lbm.nx, self.lbm.ny, self.lbm.nz) != (
            self.mpm.nx,
            self.mpm.ny,
            self.mpm.nz,
        ):
            raise ValueError(
                "For MVP, LBM and MPM grids must have the same resolution. "
                f"Got LBM={(self.lbm.nx, self.lbm.ny, self.lbm.nz)}, "
                f"MPM={(self.mpm.nx, self.mpm.ny, self.mpm.nz)}."
            )

    @property
    def mpm_dt(self) -> float:
        return self.lbm_dt / float(self.coupling.mpm_substeps_per_lbm_step)
