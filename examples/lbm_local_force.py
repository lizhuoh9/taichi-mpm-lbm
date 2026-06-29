import numpy as np
import taichi as ti

from fsi.config import BoundaryConfig, LBMConfig
from fsi.lbm3d import LBMSolver3D


def main() -> None:
    ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False, offline_cache=False)

    cfg = LBMConfig(
        nx=32,
        ny=16,
        nz=8,
        viscosity=0.1,
        rho0=1.0,
        force=(0.0, 0.0, 0.0),
        x_left=BoundaryConfig("periodic"),
        x_right=BoundaryConfig("periodic"),
        y_left=BoundaryConfig("periodic"),
        y_right=BoundaryConfig("periodic"),
        z_left=BoundaryConfig("periodic"),
        z_right=BoundaryConfig("periodic"),
    )

    solver = LBMSolver3D(cfg)
    solver.initialize()

    force = np.zeros((cfg.nx, cfg.ny, cfg.nz, 3), dtype=np.float32)
    forced_slice = slice(cfg.nx // 4, cfg.nx // 2)
    force[forced_slice, :, :, 0] = 1.0e-6
    solver.set_force_from_numpy(force)

    for step in range(100):
        solver.step()
        if step % 20 == 0:
            velocity = solver.velocity_numpy()
            forced_mean = float(velocity[forced_slice, :, :, 0].mean())
            global_mean = float(velocity[..., 0].mean())
            print(
                f"step={step:04d}, "
                f"forced_mean_ux={forced_mean:.6e}, "
                f"global_mean_ux={global_mean:.6e}, "
                f"max|u|={solver.max_velocity_norm():.6e}"
            )


if __name__ == "__main__":
    main()
