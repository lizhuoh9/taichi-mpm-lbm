import taichi as ti

from fsi.config import MPMConfig
from fsi.mpm3d import MPMSolver3D


def _format_vec(values) -> str:
    return "[" + ", ".join(f"{float(value):.8f}" for value in values) + "]"


def main() -> None:
    ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False, offline_cache=False)

    cfg = MPMConfig(
        nx=32,
        ny=32,
        nz=32,
        max_particles=100_000,
        dx=1.0,
        dt=1.0e-3,
        density=1.0,
        youngs_modulus=1_000.0,
        poisson_ratio=0.25,
        gravity=(0.0, -1.0e-3, 0.0),
    )

    solver = MPMSolver3D(cfg)
    count = solver.initialize_particles_box(
        lower=(12.0, 18.0, 12.0),
        upper=(20.0, 26.0, 20.0),
        spacing=1.0,
    )

    print(f"particles={count}")
    print(f"initial center_of_mass={_format_vec(solver.center_of_mass())}")

    for step in range(100):
        solver.substep()
        if step % 20 == 0:
            print(
                f"step={step:04d}, "
                f"center_of_mass={_format_vec(solver.center_of_mass())}, "
                f"max|v|={solver.max_velocity_norm():.6e}"
            )


if __name__ == "__main__":
    main()
