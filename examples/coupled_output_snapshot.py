from pathlib import Path

import taichi as ti

from fsi import (
    BoundaryConfig,
    CouplingConfig,
    FSISimulation,
    LBMConfig,
    MPMConfig,
    OutputConfig,
    SimulationConfig,
)


def main() -> None:
    ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False, offline_cache=False)

    cfg = SimulationConfig(
        num_steps=10,
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
        output=OutputConfig(
            output_dir=Path("outputs/coupled_output_snapshot"),
            output_interval=5,
            output_format="npz",
            write_lbm_fields=True,
            write_mpm_particles=True,
            write_coupling_fields=True,
        ),
    )

    sim = FSISimulation(cfg)
    sim.initialize(
        mpm_box_lower=(6.0, 6.0, 6.0),
        mpm_box_upper=(10.0, 10.0, 10.0),
        mpm_box_spacing=1.0,
    )

    sim.run(steps=10, write_output=True, write_initial=True)
    snapshots = sorted(cfg.output.output_dir.glob("snapshot_*.npz"))

    print(f"wrote {len(snapshots)} snapshots to {cfg.output.output_dir}")
    for path in snapshots:
        print(path.name)


if __name__ == "__main__":
    main()
