from __future__ import annotations

from pathlib import Path

import numpy as np
import taichi as ti

from .config import BoundaryConfig, LBMConfig


_Q = 19
_SUPPORTED_X_BOUNDARIES = {"periodic", "pressure", "velocity"}
_PERIODIC = "periodic"


def _boundary_mode(boundary: BoundaryConfig) -> int:
    if boundary.boundary_type == "periodic":
        return 0
    if boundary.boundary_type == "pressure":
        return 1
    if boundary.boundary_type == "velocity":
        return 2
    if boundary.boundary_type == "wall":
        raise NotImplementedError(
            "wall face boundaries are reserved for a later step; use solid masks for now."
        )
    raise ValueError(f"Unsupported boundary type: {boundary.boundary_type}.")


def _mrt_matrix() -> np.ndarray:
    return np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, -2, -2, -2, -2, -2, -2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0],
            [0, -2, 2, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1],
            [0, 0, 0, -2, 2, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1],
            [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1],
            [0, 0, 0, 0, 0, -2, 2, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1],
            [0, 2, 2, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2],
            [0, -2, -2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2],
            [0, 0, 0, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0],
            [0, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, -1, 1, -1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, -1, 1, 1, -1],
        ],
        dtype=np.float32,
    )


@ti.data_oriented
class LBMSolver3D:
    """Dense single-phase 3D D3Q19/MRT LBM solver.

    The solver adapts the upstream ``taichi_LBM3D`` dense single-phase solver
    into an instance-based API. Callers must initialize Taichi before creating
    this class.
    """

    def __init__(self, config: LBMConfig):
        self.config = config
        self.config.validate()
        self._validate_supported_config()

        self.nx = int(config.nx)
        self.ny = int(config.ny)
        self.nz = int(config.nz)
        self.q = _Q

        self.bc_x_left_mode = _boundary_mode(config.x_left)
        self.bc_x_right_mode = _boundary_mode(config.x_right)

        self._allocate_fields()
        self._initialize_static_numpy_data()
        self._copy_static_data_to_fields()
        self._set_scalar_fields()
        self._initialized = False

    def _validate_supported_config(self) -> None:
        if self.config.use_sparse:
            raise NotImplementedError("Sparse LBM storage is not implemented in Step 2.")

        for name, boundary in (
            ("x_left", self.config.x_left),
            ("x_right", self.config.x_right),
        ):
            if boundary.boundary_type not in _SUPPORTED_X_BOUNDARIES:
                raise NotImplementedError(
                    f"{name}={boundary.boundary_type!r} is not implemented in Step 2."
                )

        for name, boundary in (
            ("y_left", self.config.y_left),
            ("y_right", self.config.y_right),
            ("z_left", self.config.z_left),
            ("z_right", self.config.z_right),
        ):
            if boundary.boundary_type != _PERIODIC:
                raise NotImplementedError(
                    f"{name}={boundary.boundary_type!r} is not implemented in Step 2; "
                    "use periodic Y/Z boundaries and solid masks for now."
                )

    def _allocate_fields(self) -> None:
        shape = (self.nx, self.ny, self.nz)
        self.f = ti.field(ti.f32, shape=(*shape, self.q))
        self.F = ti.field(ti.f32, shape=(*shape, self.q))
        self.moment = ti.field(ti.f32, shape=(*shape, self.q))
        self.rho = ti.field(ti.f32, shape=shape)
        self.v = ti.Vector.field(3, ti.f32, shape=shape)

        self.e = ti.Vector.field(3, ti.i32, shape=self.q)
        self.e_f = ti.Vector.field(3, ti.f32, shape=self.q)
        self.w = ti.field(ti.f32, shape=self.q)
        self.solid = ti.field(ti.i32, shape=shape)

        self.LR = ti.field(ti.i32, shape=self.q)
        self.S_dig = ti.field(ti.f32, shape=self.q)

        self.ext_f = ti.Vector.field(3, ti.f32, shape=())
        self.rho0_field = ti.field(ti.f32, shape=())
        self.initial_velocity_field = ti.Vector.field(3, ti.f32, shape=())

        self.bc_rho_x_left = ti.field(ti.f32, shape=())
        self.bc_rho_x_right = ti.field(ti.f32, shape=())
        self.bc_vel_x_left = ti.Vector.field(3, ti.f32, shape=())
        self.bc_vel_x_right = ti.Vector.field(3, ti.f32, shape=())

        self.M = ti.field(ti.f32, shape=(self.q, self.q))
        self.inv_M = ti.field(ti.f32, shape=(self.q, self.q))

    def _initialize_static_numpy_data(self) -> None:
        tau_f = 3.0 * self.config.viscosity + 0.5
        s_v = 1.0 / tau_f
        s_other = 8.0 * (2.0 - s_v) / (8.0 - s_v)

        self._s_dig_np = np.array(
            [
                0.0,
                s_v,
                s_v,
                0.0,
                s_other,
                0.0,
                s_other,
                0.0,
                s_other,
                s_v,
                s_v,
                s_v,
                s_v,
                s_v,
                s_v,
                s_v,
                s_other,
                s_other,
                s_other,
            ],
            dtype=np.float32,
        )
        self._m_np = _mrt_matrix()
        self._inv_m_np = np.linalg.inv(self._m_np).astype(np.float32)
        self._lr_np = np.array(
            [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17],
            dtype=np.int32,
        )

    def _copy_static_data_to_fields(self) -> None:
        self.M.from_numpy(self._m_np)
        self.inv_M.from_numpy(self._inv_m_np)
        self.LR.from_numpy(self._lr_np)
        self.S_dig.from_numpy(self._s_dig_np)

    def _set_scalar_fields(self) -> None:
        self.ext_f[None] = ti.Vector([float(x) for x in self.config.force])
        self.rho0_field[None] = float(self.config.rho0)
        self.initial_velocity_field[None] = ti.Vector(
            [float(x) for x in self.config.initial_velocity]
        )
        self.bc_rho_x_left[None] = float(self.config.x_left.rho)
        self.bc_rho_x_right[None] = float(self.config.x_right.rho)
        self.bc_vel_x_left[None] = ti.Vector([float(x) for x in self.config.x_left.velocity])
        self.bc_vel_x_right[None] = ti.Vector([float(x) for x in self.config.x_right.velocity])

    def initialize(self, solid_np: np.ndarray | None = None) -> None:
        if solid_np is None:
            solid = np.zeros((self.nx, self.ny, self.nz), dtype=np.int32)
        else:
            solid = self._normalize_solid_mask(solid_np)

        self.solid.from_numpy(solid)
        self.static_init()
        self.init_fields()
        self._initialized = True

    def _normalize_solid_mask(self, solid_np: np.ndarray) -> np.ndarray:
        arr = np.asarray(solid_np)
        expected_shape = (self.nx, self.ny, self.nz)
        if arr.shape != expected_shape:
            raise ValueError(f"solid_np shape must be {expected_shape}, got {arr.shape}.")
        normalized = np.where(arr > 0, 1, 0).astype(np.int32)
        return np.ascontiguousarray(normalized)

    @classmethod
    def load_solid_from_txt(cls, filename: str | Path, nx: int, ny: int, nz: int) -> np.ndarray:
        data = np.loadtxt(filename)
        data = np.reshape(data, (nx, ny, nz), order="F")
        return np.ascontiguousarray(np.where(data > 0, 1, 0).astype(np.int32))

    @ti.kernel
    def static_init(self):
        self.e[0] = ti.Vector([0, 0, 0])
        self.e[1] = ti.Vector([1, 0, 0])
        self.e[2] = ti.Vector([-1, 0, 0])
        self.e[3] = ti.Vector([0, 1, 0])
        self.e[4] = ti.Vector([0, -1, 0])
        self.e[5] = ti.Vector([0, 0, 1])
        self.e[6] = ti.Vector([0, 0, -1])
        self.e[7] = ti.Vector([1, 1, 0])
        self.e[8] = ti.Vector([-1, -1, 0])
        self.e[9] = ti.Vector([1, -1, 0])
        self.e[10] = ti.Vector([-1, 1, 0])
        self.e[11] = ti.Vector([1, 0, 1])
        self.e[12] = ti.Vector([-1, 0, -1])
        self.e[13] = ti.Vector([1, 0, -1])
        self.e[14] = ti.Vector([-1, 0, 1])
        self.e[15] = ti.Vector([0, 1, 1])
        self.e[16] = ti.Vector([0, -1, -1])
        self.e[17] = ti.Vector([0, 1, -1])
        self.e[18] = ti.Vector([0, -1, 1])

        for q in ti.static(range(_Q)):
            self.e_f[q] = ti.cast(self.e[q], ti.f32)

        self.w[0] = 1.0 / 3.0
        for q in ti.static(range(1, 7)):
            self.w[q] = 1.0 / 18.0
        for q in ti.static(range(7, _Q)):
            self.w[q] = 1.0 / 36.0

    @ti.func
    def feq(self, q, rho_local, velocity):
        eu = self.e_f[q].dot(velocity)
        uv = velocity.dot(velocity)
        return self.w[q] * rho_local * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.kernel
    def init_fields(self):
        for i, j, k in self.rho:
            self.rho[i, j, k] = self.rho0_field[None]
            self.v[i, j, k] = self.initial_velocity_field[None]
            if self.solid[i, j, k] != 0:
                self.v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

            for q in ti.static(range(_Q)):
                value = self.feq(q, self.rho[i, j, k], self.v[i, j, k])
                self.f[i, j, k, q] = value
                self.F[i, j, k, q] = value

    @ti.func
    def multiply_m(self, i, j, k, row):
        out = 0.0
        for q in range(19):
            out += self.M[row, q] * self.F[i, j, k, q]
        return out

    @ti.func
    def guo_force(self, moment, velocity):
        out = 0.0
        for q in range(19):
            e_minus_u_dot_f = (self.e_f[q] - velocity).dot(self.ext_f[None])
            eu_ef = self.e_f[q].dot(velocity) * self.e_f[q].dot(self.ext_f[None])
            out += self.w[q] * (e_minus_u_dot_f + eu_ef) * self.M[moment, q]
        return out

    @ti.func
    def meq_vec(self, moment, rho_local, velocity):
        out = 0.0
        if moment == 0:
            out = rho_local
        if moment == 1:
            out = velocity.dot(velocity)
        if moment == 3:
            out = velocity[0]
        if moment == 5:
            out = velocity[1]
        if moment == 7:
            out = velocity[2]
        if moment == 9:
            out = (
                2.0 * velocity.x * velocity.x
                - velocity.y * velocity.y
                - velocity.z * velocity.z
            )
        if moment == 11:
            out = velocity.y * velocity.y - velocity.z * velocity.z
        if moment == 13:
            out = velocity.x * velocity.y
        if moment == 14:
            out = velocity.y * velocity.z
        if moment == 15:
            out = velocity.x * velocity.z
        return out

    @ti.kernel
    def collide(self):
        for i, j, k in self.rho:
            if self.solid[i, j, k] == 0:
                for moment in range(19):
                    moment_value = self.multiply_m(i, j, k, moment)
                    meq = self.meq_vec(moment, self.rho[i, j, k], self.v[i, j, k])
                    moment_value -= self.S_dig[moment] * (moment_value - meq)
                    moment_value += (
                        1.0 - 0.5 * self.S_dig[moment]
                    ) * self.guo_force(moment, self.v[i, j, k])
                    self.moment[i, j, k, moment] = moment_value

                for q in range(19):
                    self.f[i, j, k, q] = 0.0
                    for moment in range(19):
                        self.f[i, j, k, q] += (
                            self.inv_M[q, moment] * self.moment[i, j, k, moment]
                        )

    @ti.func
    def periodic_index(self, idx):
        out = ti.Vector([idx[0], idx[1], idx[2]])
        if idx[0] < 0:
            out[0] = self.nx - 1
        if idx[0] > self.nx - 1:
            out[0] = 0
        if idx[1] < 0:
            out[1] = self.ny - 1
        if idx[1] > self.ny - 1:
            out[1] = 0
        if idx[2] < 0:
            out[2] = self.nz - 1
        if idx[2] > self.nz - 1:
            out[2] = 0
        return out

    @ti.kernel
    def stream(self):
        for cell in ti.grouped(self.rho):
            if self.solid[cell] == 0:
                for q in ti.static(range(_Q)):
                    target = self.periodic_index(cell + self.e[q])
                    if self.solid[target] == 0:
                        self.F[target, q] = self.f[cell, q]
                    else:
                        self.F[cell, self.LR[q]] = self.f[cell, q]

    @ti.kernel
    def apply_boundary(self):
        if ti.static(self.bc_x_left_mode == 1):
            for j, k in ti.ndrange(self.ny, self.nz):
                if self.solid[0, j, k] == 0:
                    for q in ti.static(range(_Q)):
                        if self.solid[1, j, k] > 0:
                            self.F[0, j, k, q] = self.feq(q, self.bc_rho_x_left[None], self.v[1, j, k])
                        else:
                            self.F[0, j, k, q] = self.feq(q, self.bc_rho_x_left[None], self.v[0, j, k])

        if ti.static(self.bc_x_left_mode == 2):
            for j, k in ti.ndrange(self.ny, self.nz):
                if self.solid[0, j, k] == 0:
                    for q in ti.static(range(_Q)):
                        opposite = self.LR[q]
                        self.F[0, j, k, q] = (
                            self.feq(opposite, self.rho0_field[None], self.bc_vel_x_left[None])
                            - self.F[0, j, k, opposite]
                            + self.feq(q, self.rho0_field[None], self.bc_vel_x_left[None])
                        )

        if ti.static(self.bc_x_right_mode == 1):
            for j, k in ti.ndrange(self.ny, self.nz):
                if self.solid[self.nx - 1, j, k] == 0:
                    for q in ti.static(range(_Q)):
                        if self.solid[self.nx - 2, j, k] > 0:
                            self.F[self.nx - 1, j, k, q] = self.feq(
                                q, self.bc_rho_x_right[None], self.v[self.nx - 2, j, k]
                            )
                        else:
                            self.F[self.nx - 1, j, k, q] = self.feq(
                                q, self.bc_rho_x_right[None], self.v[self.nx - 1, j, k]
                            )

        if ti.static(self.bc_x_right_mode == 2):
            for j, k in ti.ndrange(self.ny, self.nz):
                if self.solid[self.nx - 1, j, k] == 0:
                    for q in ti.static(range(_Q)):
                        opposite = self.LR[q]
                        self.F[self.nx - 1, j, k, q] = (
                            self.feq(opposite, self.rho0_field[None], self.bc_vel_x_right[None])
                            - self.F[self.nx - 1, j, k, opposite]
                            + self.feq(q, self.rho0_field[None], self.bc_vel_x_right[None])
                        )

    @ti.kernel
    def update_macro(self):
        for cell in ti.grouped(self.rho):
            if self.solid[cell] == 0:
                self.rho[cell] = 0.0
                self.v[cell] = ti.Vector([0.0, 0.0, 0.0])

                for q in ti.static(range(_Q)):
                    self.f[cell, q] = self.F[cell, q]
                    self.rho[cell] += self.f[cell, q]
                    self.v[cell] += self.e_f[q] * self.f[cell, q]

                self.v[cell] /= self.rho[cell]
                self.v[cell] += (self.ext_f[None] * 0.5) / self.rho[cell]
            else:
                self.rho[cell] = self.rho0_field[None]
                self.v[cell] = ti.Vector([0.0, 0.0, 0.0])

    def step(self) -> None:
        if not self._initialized:
            raise RuntimeError("LBMSolver3D.initialize must be called before step().")
        self.collide()
        self.stream()
        self.apply_boundary()
        self.update_macro()

    def density_numpy(self) -> np.ndarray:
        return self.rho.to_numpy()

    def velocity_numpy(self) -> np.ndarray:
        return self.v.to_numpy()

    def solid_numpy(self) -> np.ndarray:
        return self.solid.to_numpy()

    def distribution_numpy(self) -> np.ndarray:
        return self.f.to_numpy()

    def total_mass(self) -> float:
        rho_np = self.density_numpy()
        solid_np = self.solid_numpy()
        return float(rho_np[solid_np == 0].sum())

    def max_velocity_norm(self) -> float:
        velocity_np = self.velocity_numpy()
        return float(np.linalg.norm(velocity_np, axis=-1).max())
