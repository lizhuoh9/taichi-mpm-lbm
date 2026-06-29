import taichi as ti

from fsi import BoundaryConfig, CouplingConfig, FSISimulation, LBMConfig, MPMConfig, SimulationConfig


def _format_vec(values) -> str:
    return "[" + ", ".join(f"{float(value):.8e}" for value in values) + "]"


def main() -> None:
    ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False, offline_cache=False)

    cfg = SimulationConfig(
        num_steps=20,
        lbm_dt=1.0e-3,
        lbm=LBMConfig(
            nx=16,
            ny=16,
            nz=16,
            viscosity=0.1,
            rho0=1.0,
            force=(0.0, 0.0, 0.0),
            initial_velocity=(1.0e-5, 0.0, 0.0),
            x_left=BoundaryConfig("periodic"),
            x_right=BoundaryConfig("periodic"),
            y_left=BoundaryConfig("periodic"),
            y_right=BoundaryConfig("periodic"),
            z_left=BoundaryConfig("periodic"),
            z_right=BoundaryConfig("periodic"),
        ),
        mpm=MPMConfig(
            nx=16,
            ny=16,
            nz=16,
            max_particles=10_000,
            dx=1.0,
            dt=5.0e-4,
            density=1.0,
            youngs_modulus=100.0,
            poisson_ratio=0.25,
            gravity=(0.0, 0.0, 0.0),
            boundary_width=1,
        ),
        coupling=CouplingConfig(
            enabled=True,
            gamma=0.01,
            mpm_substeps_per_lbm_step=2,
        ),
    )

    sim = FSISimulation(cfg)
    count = sim.initialize(
        mpm_box_lower=(6.0, 6.0, 6.0),
        mpm_box_upper=(10.0, 10.0, 10.0),
        mpm_box_spacing=1.0,
    )

    print(f"particles={count}")
    print(f"initial center_of_mass={_format_vec(sim.mpm.center_of_mass())}")

    for step in range(cfg.num_steps):
        diag = sim.step()
        if step % 5 == 0:
            print(
                f"step={diag['step']:04d}, "
                f"time={diag['time']:.6e}, "
                f"lbm_mass={diag['lbm_total_mass']:.6f}, "
                f"lbm_max|u|={diag['lbm_max_velocity_norm']:.6e}, "
                f"mpm_com={_format_vec(diag['mpm_center_of_mass'])}, "
                f"mpm_max|v|={diag['mpm_max_velocity_norm']:.6e}"
            )


if __name__ == "__main__":
    main()
