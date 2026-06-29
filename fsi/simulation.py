from __future__ import annotations

import numpy as np

from .config import SimulationConfig
from .coupling import LBMMpmCoupler
from .lbm3d import LBMSolver3D
from .mpm3d import MPMSolver3D
from .output import SimulationOutputWriter


class FSISimulation:
    """Top-level explicit LBM-MPM coupled simulation wrapper."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.config.validate()

        self.lbm = LBMSolver3D(config.lbm)
        self.mpm = MPMSolver3D(config.mpm)
        self.coupler = LBMMpmCoupler(config.coupling, self.lbm, self.mpm)
        self.output_writer = SimulationOutputWriter(config.output)

        self.step_index = 0
        self.time = 0.0
        self._lbm_initialized = False
        self._mpm_initialized = False

    def initialize_lbm(self, solid_np: np.ndarray | None = None) -> None:
        self.lbm.initialize(solid_np=solid_np)
        self._lbm_initialized = True

    def initialize_mpm_box(
        self,
        lower: tuple[float, float, float],
        upper: tuple[float, float, float],
        spacing: float | tuple[float, float, float] | None = None,
        initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> int:
        count = self.mpm.initialize_particles_box(
            lower=lower,
            upper=upper,
            spacing=spacing,
            initial_velocity=initial_velocity,
        )
        self._mpm_initialized = True
        return count

    def initialize_mpm_from_numpy(
        self,
        positions: np.ndarray,
        velocities: np.ndarray | None = None,
        particle_mass: float | None = None,
        particle_volume: float | None = None,
    ) -> int:
        count = self.mpm.initialize_particles_from_numpy(
            positions=positions,
            velocities=velocities,
            particle_mass=particle_mass,
            particle_volume=particle_volume,
        )
        self._mpm_initialized = True
        return count

    def initialize(
        self,
        solid_np: np.ndarray | None = None,
        mpm_box_lower: tuple[float, float, float] | None = None,
        mpm_box_upper: tuple[float, float, float] | None = None,
        mpm_box_spacing: float | tuple[float, float, float] | None = None,
        mpm_initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> int:
        if mpm_box_lower is None or mpm_box_upper is None:
            raise ValueError("mpm_box_lower and mpm_box_upper are required.")

        self.initialize_lbm(solid_np=solid_np)
        return self.initialize_mpm_box(
            lower=mpm_box_lower,
            upper=mpm_box_upper,
            spacing=mpm_box_spacing,
            initial_velocity=mpm_initial_velocity,
        )

    def step(self) -> dict[str, object]:
        self._validate_initialized()
        self.coupler.step(self.config.lbm_dt)
        self.step_index += 1
        self.time += self.config.lbm_dt
        return self.diagnostics()

    def run(
        self,
        steps: int | None = None,
        write_output: bool = False,
        write_initial: bool = False,
    ) -> list[dict[str, object]]:
        self._validate_initialized()
        count = self.config.num_steps if steps is None else int(steps)
        if count <= 0:
            raise ValueError("steps must be positive.")

        history = []
        if write_output and write_initial:
            self.write_snapshot()

        for _ in range(count):
            diag = self.step()
            history.append(diag)
            if write_output and self.output_writer.should_write(self.step_index):
                self.write_snapshot()
        return history

    def write_snapshot(self):
        self._validate_initialized()
        return self.output_writer.write_snapshot(self)

    def diagnostics(self) -> dict[str, object]:
        self._validate_initialized()
        return {
            "step": self.step_index,
            "time": self.time,
            "lbm_total_mass": self.lbm.total_mass(),
            "lbm_max_velocity_norm": self.lbm.max_velocity_norm(),
            "mpm_particle_count": self.mpm.particle_count(),
            "mpm_center_of_mass": self.mpm.center_of_mass(),
            "mpm_max_velocity_norm": self.mpm.max_velocity_norm(),
            "total_particle_coupling_force": self.coupler.total_particle_coupling_force(),
            "total_fluid_coupling_force": self.coupler.total_fluid_coupling_force(),
        }

    def _validate_initialized(self) -> None:
        if not self._lbm_initialized or not getattr(self.lbm, "_initialized", False):
            raise RuntimeError("FSISimulation.initialize_lbm must be called before stepping.")
        if not self._mpm_initialized or not getattr(self.mpm, "_initialized", False):
            raise RuntimeError("FSISimulation MPM initialization must be called before stepping.")
