from __future__ import annotations

import numpy as np
import taichi as ti

from .config import MPMConfig


@ti.data_oriented
class MPMSolver3D:
    """Standalone dense 3D elastic MLS-MPM solver.

    Coordinates are grid/lattice coordinates. Callers must initialize Taichi
    before creating this class.
    """

    def __init__(self, config: MPMConfig):
        self.config = config
        self.config.validate()

        self.nx = int(config.nx)
        self.ny = int(config.ny)
        self.nz = int(config.nz)
        self.max_particles = int(config.max_particles)

        self.dx = float(config.dx)
        self.inv_dx = 1.0 / self.dx
        self.dt_default = float(config.dt)
        self.boundary_width = int(config.boundary_width)
        self.boundary_damping = float(config.boundary_damping)
        self.apic = bool(config.apic)

        self.mu = config.youngs_modulus / (2.0 * (1.0 + config.poisson_ratio))
        self.lam = (
            config.youngs_modulus
            * config.poisson_ratio
            / ((1.0 + config.poisson_ratio) * (1.0 - 2.0 * config.poisson_ratio))
        )

        self._allocate_fields()
        self._set_scalar_fields()
        self._initialized = False

    def _allocate_fields(self) -> None:
        grid_shape = (self.nx, self.ny, self.nz)

        self.num_particles = ti.field(ti.i32, shape=())
        self.x = ti.Vector.field(3, ti.f32, shape=self.max_particles)
        self.v = ti.Vector.field(3, ti.f32, shape=self.max_particles)
        self.C = ti.Matrix.field(3, 3, ti.f32, shape=self.max_particles)
        self.F = ti.Matrix.field(3, 3, ti.f32, shape=self.max_particles)
        self.mass = ti.field(ti.f32, shape=self.max_particles)
        self.volume = ti.field(ti.f32, shape=self.max_particles)
        self.active = ti.field(ti.i32, shape=self.max_particles)
        self.particle_force = ti.Vector.field(3, ti.f32, shape=self.max_particles)

        self.grid_v = ti.Vector.field(3, ti.f32, shape=grid_shape)
        self.grid_m = ti.field(ti.f32, shape=grid_shape)

        self.dt_field = ti.field(ti.f32, shape=())
        self.gravity = ti.Vector.field(3, ti.f32, shape=())
        self.identity = ti.Matrix.field(3, 3, ti.f32, shape=())

    def _set_scalar_fields(self) -> None:
        self.dt_field[None] = self.dt_default
        self.gravity[None] = ti.Vector([float(x) for x in self.config.gravity])
        self.identity[None] = np.eye(3, dtype=np.float32)

    def initialize_particles_box(
        self,
        lower: tuple[float, float, float],
        upper: tuple[float, float, float],
        spacing: float | tuple[float, float, float] | None = None,
        initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> int:
        lower_np = self._validate_vec3(lower, "lower")
        upper_np = self._validate_vec3(upper, "upper")
        if np.any(upper_np <= lower_np):
            raise ValueError("upper must be greater than lower in every dimension.")

        spacing_np = self._default_spacing() if spacing is None else self._validate_spacing(spacing)
        axes = [np.arange(lower_np[axis], upper_np[axis], spacing_np[axis]) for axis in range(3)]
        if any(len(axis_values) == 0 for axis_values in axes):
            raise ValueError("particle box produced no particles.")

        mesh = np.meshgrid(*axes, indexing="ij")
        positions = np.stack([component.ravel() for component in mesh], axis=1).astype(np.float32)
        particle_volume = float(np.prod(spacing_np))
        velocities = np.broadcast_to(
            self._validate_vec3(initial_velocity, "initial_velocity"), positions.shape
        ).astype(np.float32)
        return self.initialize_particles_from_numpy(
            positions,
            velocities=velocities,
            particle_volume=particle_volume,
        )

    def initialize_particles_from_numpy(
        self,
        positions: np.ndarray,
        velocities: np.ndarray | None = None,
        particle_mass: float | None = None,
        particle_volume: float | None = None,
    ) -> int:
        pos = np.asarray(positions, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must have shape (n, 3).")
        count = int(pos.shape[0])
        if count <= 0:
            raise ValueError("positions must contain at least one particle.")
        if count > self.max_particles:
            raise ValueError(
                f"particle count {count} exceeds max_particles={self.max_particles}."
            )

        if velocities is None:
            vel = np.zeros((count, 3), dtype=np.float32)
        else:
            vel = np.asarray(velocities, dtype=np.float32)
            if vel.shape != (count, 3):
                raise ValueError(f"velocities must have shape {(count, 3)}, got {vel.shape}.")

        volume_value = (
            float(particle_volume)
            if particle_volume is not None
            else self.dx**3 / float(self.config.particles_per_cell**3)
        )
        if volume_value <= 0.0:
            raise ValueError("particle_volume must be positive.")

        mass_value = (
            float(particle_mass)
            if particle_mass is not None
            else float(self.config.density) * volume_value
        )
        if mass_value <= 0.0:
            raise ValueError("particle_mass must be positive.")

        x_np = np.zeros((self.max_particles, 3), dtype=np.float32)
        v_np = np.zeros((self.max_particles, 3), dtype=np.float32)
        c_np = np.zeros((self.max_particles, 3, 3), dtype=np.float32)
        f_np = np.zeros((self.max_particles, 3, 3), dtype=np.float32)
        mass_np = np.zeros(self.max_particles, dtype=np.float32)
        volume_np = np.zeros(self.max_particles, dtype=np.float32)
        active_np = np.zeros(self.max_particles, dtype=np.int32)
        particle_force_np = np.zeros((self.max_particles, 3), dtype=np.float32)

        x_np[:count] = pos
        v_np[:count] = vel
        f_np[:count] = np.eye(3, dtype=np.float32)
        mass_np[:count] = mass_value
        volume_np[:count] = volume_value
        active_np[:count] = 1

        self.num_particles[None] = count
        self.x.from_numpy(x_np)
        self.v.from_numpy(v_np)
        self.C.from_numpy(c_np)
        self.F.from_numpy(f_np)
        self.mass.from_numpy(mass_np)
        self.volume.from_numpy(volume_np)
        self.active.from_numpy(active_np)
        self.particle_force.from_numpy(particle_force_np)
        self._initialized = True
        return count

    def _default_spacing(self) -> np.ndarray:
        return np.full(3, self.dx / float(self.config.particles_per_cell), dtype=np.float32)

    @staticmethod
    def _validate_vec3(values: tuple[float, float, float], name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape != (3,):
            raise ValueError(f"{name} must have 3 components.")
        return arr

    @staticmethod
    def _validate_spacing(spacing: float | tuple[float, float, float]) -> np.ndarray:
        if np.isscalar(spacing):
            arr = np.full(3, float(spacing), dtype=np.float32)
        else:
            arr = np.asarray(spacing, dtype=np.float32)
        if arr.shape != (3,):
            raise ValueError("spacing must be a scalar or have 3 components.")
        if np.any(arr <= 0.0):
            raise ValueError("spacing must be positive.")
        return arr

    def substep(self, dt: float | None = None) -> None:
        if not self._initialized:
            raise RuntimeError(
                "MPMSolver3D.initialize_particles_box or initialize_particles_from_numpy "
                "must be called before substep()."
            )
        dt_value = self.dt_default if dt is None else float(dt)
        if dt_value <= 0.0:
            raise ValueError("dt must be positive.")
        self.dt_field[None] = dt_value

        self.clear_grid()
        self.p2g()
        self.grid_op()
        self.g2p()

    @ti.kernel
    def clear_grid(self):
        for node in ti.grouped(self.grid_m):
            self.grid_v[node] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_m[node] = 0.0

    @ti.kernel
    def clear_particle_force(self):
        for p in range(self.num_particles[None]):
            self.particle_force[p] = ti.Vector([0.0, 0.0, 0.0])

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

    @ti.kernel
    def p2g(self):
        dt = self.dt_field[None]
        for p in range(self.num_particles[None]):
            if self.active[p] != 0:
                base = (self.x[p] * self.inv_dx - 0.5).cast(ti.i32)
                fx = self.x[p] * self.inv_dx - base.cast(ti.f32)
                weights = self._weights(fx)

                self.F[p] = (self.identity[None] + dt * self.C[p]) @ self.F[p]
                j = self.F[p].determinant()
                u, _, v = ti.svd(self.F[p])
                r = u @ v.transpose()
                stress = (
                    2.0 * self.mu * (self.F[p] - r) @ self.F[p].transpose()
                    + self.lam * j * (j - 1.0) * self.identity[None]
                )
                stress = -dt * self.volume[p] * 4.0 * self.inv_dx * self.inv_dx * stress
                affine = stress + self.mass[p] * self.C[p]

                for i, j_offset, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j_offset, k])
                    node = base + offset
                    if self._inside_grid(node):
                        dpos = (offset.cast(ti.f32) - fx) * self.dx
                        weight = weights[i, 0] * weights[j_offset, 1] * weights[k, 2]
                        momentum = (
                            self.mass[p] * self.v[p]
                            + affine @ dpos
                            + dt * self.particle_force[p]
                        )
                        self.grid_v[node] += weight * momentum
                        self.grid_m[node] += weight * self.mass[p]

    @ti.kernel
    def grid_op(self):
        dt = self.dt_field[None]
        for node in ti.grouped(self.grid_m):
            mass = self.grid_m[node]
            if mass > 0.0:
                self.grid_v[node] /= mass
                self.grid_v[node] += dt * self.gravity[None]

                if node[0] < self.boundary_width and self.grid_v[node][0] < 0.0:
                    self.grid_v[node][0] = -self.boundary_damping * self.grid_v[node][0]
                if node[0] >= self.nx - self.boundary_width and self.grid_v[node][0] > 0.0:
                    self.grid_v[node][0] = -self.boundary_damping * self.grid_v[node][0]
                if node[1] < self.boundary_width and self.grid_v[node][1] < 0.0:
                    self.grid_v[node][1] = -self.boundary_damping * self.grid_v[node][1]
                if node[1] >= self.ny - self.boundary_width and self.grid_v[node][1] > 0.0:
                    self.grid_v[node][1] = -self.boundary_damping * self.grid_v[node][1]
                if node[2] < self.boundary_width and self.grid_v[node][2] < 0.0:
                    self.grid_v[node][2] = -self.boundary_damping * self.grid_v[node][2]
                if node[2] >= self.nz - self.boundary_width and self.grid_v[node][2] > 0.0:
                    self.grid_v[node][2] = -self.boundary_damping * self.grid_v[node][2]

    @ti.kernel
    def g2p(self):
        dt = self.dt_field[None]
        for p in range(self.num_particles[None]):
            if self.active[p] != 0:
                base = (self.x[p] * self.inv_dx - 0.5).cast(ti.i32)
                fx = self.x[p] * self.inv_dx - base.cast(ti.f32)
                weights = self._weights(fx)

                new_v = ti.Vector([0.0, 0.0, 0.0])
                new_c = ti.Matrix.zero(ti.f32, 3, 3)

                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])
                    node = base + offset
                    if self._inside_grid(node):
                        dpos = offset.cast(ti.f32) - fx
                        weight = weights[i, 0] * weights[j, 1] * weights[k, 2]
                        grid_velocity = self.grid_v[node]
                        new_v += weight * grid_velocity
                        new_c += 4.0 * self.inv_dx * weight * grid_velocity.outer_product(dpos)

                self.v[p] = new_v
                if ti.static(self.apic):
                    self.C[p] = new_c
                else:
                    self.C[p] = ti.Matrix.zero(ti.f32, 3, 3)
                self.x[p] += dt * self.v[p]
                self._clamp_particle_position(p)

    @ti.func
    def _clamp_particle_position(self, p):
        lower = 0.5 * self.dx
        upper_x = (self.nx - 1.5) * self.dx
        upper_y = (self.ny - 1.5) * self.dx
        upper_z = (self.nz - 1.5) * self.dx
        self.x[p][0] = ti.min(ti.max(self.x[p][0], lower), upper_x)
        self.x[p][1] = ti.min(ti.max(self.x[p][1], lower), upper_y)
        self.x[p][2] = ti.min(ti.max(self.x[p][2], lower), upper_z)

    def particle_count(self) -> int:
        return int(self.num_particles[None])

    def positions_numpy(self) -> np.ndarray:
        count = self.particle_count()
        return self.x.to_numpy()[:count].copy()

    def velocities_numpy(self) -> np.ndarray:
        count = self.particle_count()
        return self.v.to_numpy()[:count].copy()

    def deformation_gradients_numpy(self) -> np.ndarray:
        count = self.particle_count()
        return self.F.to_numpy()[:count].copy()

    def active_numpy(self) -> np.ndarray:
        count = self.particle_count()
        return self.active.to_numpy()[:count].copy()

    def center_of_mass(self) -> np.ndarray:
        count = self.particle_count()
        if count == 0:
            return np.zeros(3, dtype=np.float32)
        positions = self.positions_numpy()
        masses = self.mass.to_numpy()[:count]
        total_mass = masses.sum()
        if total_mass <= 0.0:
            return np.zeros(3, dtype=np.float32)
        return (positions * masses[:, None]).sum(axis=0) / total_mass

    def max_velocity_norm(self) -> float:
        velocities = self.velocities_numpy()
        if len(velocities) == 0:
            return 0.0
        return float(np.linalg.norm(velocities, axis=1).max())
