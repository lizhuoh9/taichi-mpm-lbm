from __future__ import annotations

import numpy as np
import taichi as ti

from .config import CouplingConfig
from .lbm3d import LBMSolver3D
from .mpm3d import MPMSolver3D


@ti.data_oriented
class LBMMpmCoupler:
    """Explicit partitioned penalty coupling between LBM and MPM solvers.

    The coupler computes particle forces from interpolated LBM velocity and
    scatters equal-and-opposite reaction force density back to the LBM force
    field. It does not own solver initialization.
    """

    def __init__(self, config: CouplingConfig, lbm: LBMSolver3D, mpm: MPMSolver3D):
        self.config = config
        self.config.validate()
        self.lbm = lbm
        self.mpm = mpm

        self._validate_solver_contract()
        self.enabled = bool(config.enabled)
        self.build_solid_volume_fraction = bool(config.build_solid_volume_fraction)
        self.substeps = int(config.mpm_substeps_per_lbm_step)

        self.nx = int(lbm.nx)
        self.ny = int(lbm.ny)
        self.nz = int(lbm.nz)
        self.inv_dx = 1.0 / float(mpm.dx)
        self.cell_volume = float(mpm.dx) ** 3

        self._allocate_fields()
        self._set_scalar_fields()

    def _validate_solver_contract(self) -> None:
        if self.config.kernel != "quadratic":
            raise ValueError(f"Unsupported coupling kernel: {self.config.kernel}.")
        if (self.lbm.nx, self.lbm.ny, self.lbm.nz) != (
            self.mpm.nx,
            self.mpm.ny,
            self.mpm.nz,
        ):
            raise ValueError(
                "LBM and MPM grids must match for Step 5 coupling. "
                f"Got LBM={(self.lbm.nx, self.lbm.ny, self.lbm.nz)}, "
                f"MPM={(self.mpm.nx, self.mpm.ny, self.mpm.nz)}."
            )
        if self.mpm.dx <= 0.0:
            raise ValueError("MPM dx must be positive.")

    def _allocate_fields(self) -> None:
        shape = (self.nx, self.ny, self.nz)
        self.coupling_force = ti.Vector.field(3, ti.f32, shape=shape)
        self.solid_volume_fraction = ti.field(ti.f32, shape=shape)
        self.gamma_field = ti.field(ti.f32, shape=())
        self.dt_ratio_field = ti.field(ti.f32, shape=())
        self.cell_volume_field = ti.field(ti.f32, shape=())
        self.build_solid_fraction_field = ti.field(ti.i32, shape=())
        self.force_limit_field = ti.field(ti.f32, shape=())
        self.relative_velocity_limit_field = ti.field(ti.f32, shape=())
        self.use_force_limit_field = ti.field(ti.i32, shape=())
        self.use_relative_velocity_limit_field = ti.field(ti.i32, shape=())
        self.gamma_ramp_steps_field = ti.field(ti.i32, shape=())
        self.coupled_step_field = ti.field(ti.i32, shape=())
        self.min_valid_weight_field = ti.field(ti.f32, shape=())
        self.particle_valid_weight = ti.field(ti.f32, shape=self.mpm.max_particles)
        self.particle_coupling_mask = ti.field(ti.i32, shape=self.mpm.max_particles)
        self.unsupported_particle_count = ti.field(ti.i32, shape=())
        self.partial_support_particle_count = ti.field(ti.i32, shape=())
        self.clipped_particle_count = ti.field(ti.i32, shape=())

    def _set_scalar_fields(self) -> None:
        self.gamma_field[None] = float(self.config.gamma)
        self.dt_ratio_field[None] = 1.0
        self.cell_volume_field[None] = self.cell_volume
        self.build_solid_fraction_field[None] = 0
        self.force_limit_field[None] = (
            0.0 if self.config.force_limit is None else float(self.config.force_limit)
        )
        self.relative_velocity_limit_field[None] = (
            0.0
            if self.config.relative_velocity_limit is None
            else float(self.config.relative_velocity_limit)
        )
        self.use_force_limit_field[None] = 0 if self.config.force_limit is None else 1
        self.use_relative_velocity_limit_field[None] = (
            0 if self.config.relative_velocity_limit is None else 1
        )
        self.gamma_ramp_steps_field[None] = int(self.config.gamma_ramp_steps)
        self.coupled_step_field[None] = 0
        self.min_valid_weight_field[None] = float(self.config.min_valid_weight)

    def step(self, lbm_dt: float) -> None:
        """Run one explicit coupled LBM step with configured MPM substeps."""

        lbm_dt_value = float(lbm_dt)
        if lbm_dt_value <= 0.0:
            raise ValueError("lbm_dt must be positive.")
        self._validate_initialized()

        self.clear_coupling_fields()
        mpm_dt = lbm_dt_value / float(self.substeps)
        dt_ratio = mpm_dt / lbm_dt_value

        for substep in range(self.substeps):
            self.mpm.clear_particle_force()
            if self.enabled:
                self.clear_particle_diagnostics()
                self.dt_ratio_field[None] = dt_ratio
                self.build_solid_fraction_field[None] = (
                    1
                    if substep == 0 and self.build_solid_volume_fraction
                    else 0
                )
                self._compute_coupling_forces_kernel()
            self.mpm.substep(dt=mpm_dt)

        if self.enabled and self.build_solid_volume_fraction:
            self._clamp_solid_volume_fraction_kernel()

        self.lbm.reset_force()
        if self.enabled:
            self._apply_coupling_force_to_lbm_kernel()
        self.lbm.step()
        if self.enabled:
            self.coupled_step_field[None] = int(self.coupled_step_field[None]) + 1

    def compute_coupling_forces(self, dt_ratio: float = 1.0) -> None:
        """Compute particle force and fluid reaction force without stepping."""

        dt_ratio_value = float(dt_ratio)
        if dt_ratio_value < 0.0:
            raise ValueError("dt_ratio must be non-negative.")
        self._validate_initialized()

        self.clear_coupling_fields()
        self.mpm.clear_particle_force()
        if not self.enabled:
            return

        self.dt_ratio_field[None] = dt_ratio_value
        self.build_solid_fraction_field[None] = 1 if self.build_solid_volume_fraction else 0
        self._compute_coupling_forces_kernel()
        if self.build_solid_volume_fraction:
            self._clamp_solid_volume_fraction_kernel()

    def clear_coupling_fields(self) -> None:
        self._clear_coupling_fields_kernel()
        self._clear_particle_diagnostics_kernel()

    def clear_particle_diagnostics(self) -> None:
        self._clear_particle_diagnostics_kernel()

    def coupling_force_numpy(self) -> np.ndarray:
        return self.coupling_force.to_numpy()

    def solid_volume_fraction_numpy(self) -> np.ndarray:
        return self.solid_volume_fraction.to_numpy()

    def particle_valid_weights_numpy(self) -> np.ndarray:
        count = self.mpm.particle_count()
        return self.particle_valid_weight.to_numpy()[:count].copy()

    def particle_coupling_mask_numpy(self) -> np.ndarray:
        count = self.mpm.particle_count()
        return self.particle_coupling_mask.to_numpy()[:count].copy()

    def total_particle_coupling_force(self) -> np.ndarray:
        return self.mpm.particle_forces_numpy().sum(axis=0)

    def total_fluid_coupling_force(self) -> np.ndarray:
        force_np = self.coupling_force_numpy()
        solid_np = self.lbm.solid_numpy()
        return force_np[solid_np == 0].sum(axis=0)

    def effective_gamma(self) -> float:
        gamma = float(self.gamma_field[None])
        ramp_steps = int(self.gamma_ramp_steps_field[None])
        if ramp_steps <= 0:
            return gamma
        step = int(self.coupled_step_field[None])
        return gamma * min(1.0, float(step + 1) / float(ramp_steps))

    def coupling_diagnostics(self) -> dict[str, int | float]:
        valid_weights = self.particle_valid_weights_numpy()
        if len(valid_weights) == 0:
            min_weight = 0.0
            mean_weight = 0.0
        else:
            min_weight = float(valid_weights.min())
            mean_weight = float(valid_weights.mean())
        return {
            "unsupported_particle_count": int(self.unsupported_particle_count[None]),
            "partial_support_particle_count": int(self.partial_support_particle_count[None]),
            "clipped_particle_count": int(self.clipped_particle_count[None]),
            "min_particle_valid_weight": min_weight,
            "mean_particle_valid_weight": mean_weight,
            "effective_gamma": self.effective_gamma(),
        }

    def _validate_initialized(self) -> None:
        if not getattr(self.lbm, "_initialized", False):
            raise RuntimeError("LBMSolver3D.initialize must be called before coupling.")
        if not getattr(self.mpm, "_initialized", False):
            raise RuntimeError(
                "MPMSolver3D particle initialization must be called before coupling."
            )

    @ti.kernel
    def _clear_coupling_fields_kernel(self):
        for cell in ti.grouped(self.coupling_force):
            self.coupling_force[cell] = ti.Vector([0.0, 0.0, 0.0])
            self.solid_volume_fraction[cell] = 0.0

    @ti.kernel
    def _clear_particle_diagnostics_kernel(self):
        self.unsupported_particle_count[None] = 0
        self.partial_support_particle_count[None] = 0
        self.clipped_particle_count[None] = 0
        for p in self.particle_valid_weight:
            self.particle_valid_weight[p] = 0.0
            self.particle_coupling_mask[p] = 0

    @ti.func
    def _inside_grid(self, node):
        return (
            0 <= node[0]
            and node[0] < self.nx
            and 0 <= node[1]
            and node[1] < self.ny
            and 0 <= node[2]
            and node[2] < self.nz
        )

    @ti.func
    def _weights(self, fx):
        w0 = 0.5 * (1.5 - fx) ** 2
        w1 = 0.75 - (fx - 1.0) ** 2
        w2 = 0.5 * (fx - 0.5) ** 2
        return ti.Matrix.rows([w0, w1, w2])

    @ti.func
    def _effective_gamma(self):
        gamma = self.gamma_field[None]
        ramp_steps = self.gamma_ramp_steps_field[None]
        if ramp_steps > 0:
            ramp = (ti.cast(self.coupled_step_field[None], ti.f32) + 1.0) / ti.cast(
                ramp_steps, ti.f32
            )
            gamma *= ti.min(1.0, ramp)
        return gamma

    @ti.kernel
    def _compute_coupling_forces_kernel(self):
        gamma = self.gamma_field[None]
        dt_ratio = self.dt_ratio_field[None]
        cell_volume = self.cell_volume_field[None]
        build_fraction = self.build_solid_fraction_field[None]
        min_valid_weight = self.min_valid_weight_field[None]
        support_tolerance = 1.0e-5

        for p in range(self.mpm.num_particles[None]):
            if self.mpm.active[p] != 0:
                grid_pos = self.mpm.x[p] * self.inv_dx
                base = (grid_pos - 0.5).cast(ti.i32)
                fx = grid_pos - base.cast(ti.f32)
                weights = self._weights(fx)

                valid_weight_sum = 0.0
                u_fluid = ti.Vector([0.0, 0.0, 0.0])

                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])
                    node = base + offset
                    if self._inside_grid(node) and self.lbm.solid[node] == 0:
                        weight = weights[i, 0] * weights[j, 1] * weights[k, 2]
                        valid_weight_sum += weight
                        u_fluid += weight * self.lbm.v[node]

                self.particle_valid_weight[p] = valid_weight_sum
                if valid_weight_sum > min_valid_weight:
                    if valid_weight_sum < 1.0 - support_tolerance:
                        ti.atomic_add(self.partial_support_particle_count[None], 1)

                    u_fluid /= valid_weight_sum
                    dv = u_fluid - self.mpm.v[p]
                    clipped = 0

                    if self.use_relative_velocity_limit_field[None] != 0:
                        dv_norm = dv.norm()
                        limit = self.relative_velocity_limit_field[None]
                        if dv_norm > limit and dv_norm > 0.0:
                            dv = dv / dv_norm * limit
                            clipped = 1

                    gamma = self._effective_gamma()
                    particle_force = gamma * self.mpm.mass[p] * dv

                    if self.use_force_limit_field[None] != 0:
                        force_norm = particle_force.norm()
                        limit = self.force_limit_field[None]
                        if force_norm > limit and force_norm > 0.0:
                            particle_force = particle_force / force_norm * limit
                            clipped = 1

                    self.mpm.particle_force[p] = particle_force
                    self.particle_coupling_mask[p] = 1
                    if clipped != 0:
                        ti.atomic_add(self.clipped_particle_count[None], 1)

                    for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                        offset = ti.Vector([i, j, k])
                        node = base + offset
                        if self._inside_grid(node) and self.lbm.solid[node] == 0:
                            weight = weights[i, 0] * weights[j, 1] * weights[k, 2]
                            normalized_weight = weight / valid_weight_sum
                            self.coupling_force[node] += (
                                -normalized_weight * particle_force * dt_ratio / cell_volume
                            )
                            if build_fraction != 0:
                                self.solid_volume_fraction[node] += (
                                    normalized_weight * self.mpm.volume[p] / cell_volume
                                )
                else:
                    self.mpm.particle_force[p] = ti.Vector([0.0, 0.0, 0.0])
                    self.particle_coupling_mask[p] = 0
                    ti.atomic_add(self.unsupported_particle_count[None], 1)

    @ti.kernel
    def _clamp_solid_volume_fraction_kernel(self):
        for cell in ti.grouped(self.solid_volume_fraction):
            self.solid_volume_fraction[cell] = ti.min(self.solid_volume_fraction[cell], 1.0)

    @ti.kernel
    def _apply_coupling_force_to_lbm_kernel(self):
        for cell in ti.grouped(self.lbm.rho):
            if self.lbm.solid[cell] == 0:
                self.lbm.force[cell] += self.coupling_force[cell]
            else:
                self.lbm.force[cell] = ti.Vector([0.0, 0.0, 0.0])
