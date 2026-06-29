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
        force=(1.0e-6, 0.0, 0.0),
        x_left=BoundaryConfig("periodic"),
        x_right=BoundaryConfig("periodic"),
        y_left=BoundaryConfig("periodic"),
        y_right=BoundaryConfig("periodic"),
        z_left=BoundaryConfig("periodic"),
        z_right=BoundaryConfig("periodic"),
    )
    solid = np.zeros((cfg.nx, cfg.ny, cfg.nz), dtype=np.int32)

    solver = LBMSolver3D(cfg)
    solver.initialize(solid_np=solid)

    mass0 = solver.total_mass()
    for step in range(100):
        solver.step()
        if step % 20 == 0:
            print(
                f"step={step:04d}, "
                f"mass={solver.total_mass():.6f}, "
                f"max|u|={solver.max_velocity_norm():.6e}"
            )

    mass1 = solver.total_mass()
    print(f"mass error: {mass1 - mass0:.6e}")


if __name__ == "__main__":
    main()
