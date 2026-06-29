from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pyevtk.hl import imageToVTK, pointsToVTK

from .config import OutputConfig

if TYPE_CHECKING:
    from .simulation import FSISimulation


class SimulationOutputWriter:
    """Write lightweight simulation snapshots for analysis and visualization."""

    def __init__(self, config: OutputConfig):
        self.config = config
        self.config.validate()
        self.output_dir = Path(config.output_dir)

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def should_write(self, step: int) -> bool:
        return int(step) % self.config.output_interval == 0

    def write_snapshot(self, sim: FSISimulation) -> list[Path]:
        output_format = self.config.output_format
        if output_format == "npz":
            return [self.write_npz_snapshot(sim)]
        if output_format == "vtk":
            return self.write_vtk_snapshot(sim)
        if output_format == "both":
            return [self.write_npz_snapshot(sim), *self.write_vtk_snapshot(sim)]
        raise ValueError(f"Unsupported output format: {output_format}.")

    def write_npz_snapshot(self, sim: FSISimulation) -> Path:
        self.ensure_output_dir()
        path = self.output_dir / f"snapshot_{sim.step_index:06d}.npz"

        arrays: dict[str, np.ndarray] = {
            "step": np.array(sim.step_index, dtype=np.int64),
            "time": np.array(sim.time, dtype=np.float64),
            "lbm_shape": np.array([sim.lbm.nx, sim.lbm.ny, sim.lbm.nz], dtype=np.int32),
            "mpm_particle_count": np.array(sim.mpm.particle_count(), dtype=np.int32),
        }

        if self.config.write_lbm_fields:
            arrays.update(
                {
                    "lbm_density": sim.lbm.density_numpy(),
                    "lbm_velocity": sim.lbm.velocity_numpy(),
                    "lbm_force": sim.lbm.force_numpy(),
                    "lbm_solid": sim.lbm.solid_numpy(),
                }
            )

        if self.config.write_mpm_particles:
            arrays.update(
                {
                    "mpm_positions": sim.mpm.positions_numpy(),
                    "mpm_velocities": sim.mpm.velocities_numpy(),
                    "mpm_particle_forces": sim.mpm.particle_forces_numpy(),
                    "mpm_deformation_gradients": sim.mpm.deformation_gradients_numpy(),
                    "mpm_active": sim.mpm.active_numpy(),
                }
            )

        if self.config.write_coupling_fields:
            arrays.update(
                {
                    "coupling_force": sim.coupler.coupling_force_numpy(),
                    "solid_volume_fraction": sim.coupler.solid_volume_fraction_numpy(),
                    "total_particle_coupling_force": sim.coupler.total_particle_coupling_force(),
                    "total_fluid_coupling_force": sim.coupler.total_fluid_coupling_force(),
                }
            )

        np.savez_compressed(path, **arrays)
        return path

    def write_vtk_snapshot(self, sim: FSISimulation) -> list[Path]:
        self.ensure_output_dir()
        step = sim.step_index
        fluid_path = self._write_fluid_vtk(sim, step)
        particle_path = self._write_particles_vtk(sim, step)
        return [fluid_path, particle_path]

    def _write_fluid_vtk(self, sim: FSISimulation, step: int) -> Path:
        density = np.ascontiguousarray(sim.lbm.density_numpy())
        velocity = np.ascontiguousarray(sim.lbm.velocity_numpy())
        force = np.ascontiguousarray(sim.lbm.force_numpy())
        solid = np.ascontiguousarray(sim.lbm.solid_numpy())
        coupling_force = np.ascontiguousarray(sim.coupler.coupling_force_numpy())
        solid_fraction = np.ascontiguousarray(sim.coupler.solid_volume_fraction_numpy())

        cell_data = {
            "rho": density,
            "velocity": self._vector_components(velocity),
            "force": self._vector_components(force),
            "solid": solid,
            "coupling_force": self._vector_components(coupling_force),
            "solid_volume_fraction": solid_fraction,
        }
        filename = imageToVTK(
            str(self.output_dir / f"fluid_{step:06d}"),
            origin=(0.0, 0.0, 0.0),
            spacing=(sim.mpm.dx, sim.mpm.dx, sim.mpm.dx),
            cellData=cell_data,
        )
        return Path(filename)

    def _write_particles_vtk(self, sim: FSISimulation, step: int) -> Path:
        positions = np.ascontiguousarray(sim.mpm.positions_numpy())
        velocities = np.ascontiguousarray(sim.mpm.velocities_numpy())
        forces = np.ascontiguousarray(sim.mpm.particle_forces_numpy())

        if len(positions) == 0:
            x = np.zeros(0, dtype=np.float32)
            y = np.zeros(0, dtype=np.float32)
            z = np.zeros(0, dtype=np.float32)
        else:
            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]

        data = {
            "velocity": self._vector_components(velocities),
            "particle_force": self._vector_components(forces),
        }
        filename = pointsToVTK(str(self.output_dir / f"particles_{step:06d}"), x, y, z, data=data)
        return Path(filename)

    @staticmethod
    def _vector_components(vector_array: np.ndarray):
        return (
            np.ascontiguousarray(vector_array[..., 0]),
            np.ascontiguousarray(vector_array[..., 1]),
            np.ascontiguousarray(vector_array[..., 2]),
        )
