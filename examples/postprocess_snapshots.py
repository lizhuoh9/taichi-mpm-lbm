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
from fsi.postprocess import (
    extract_snapshot_timeseries,
    plot_timeseries,
    summarize_snapshots,
    write_timeseries_csv,
    write_timeseries_json,
)


def main() -> None:
    ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False, offline_cache=False)

    output_dir = Path("outputs/postprocess_snapshots")
    cfg = SimulationConfig(
        num_steps=4,
        lbm_dt=1.0e-3,
        lbm=LBMConfig(
            nx=8,
            ny=8,
            nz=8,
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
            nx=8,
            ny=8,
            nz=8,
            max_particles=128,
            dx=1.0,
            dt=1.0e-3,
            density=1.0,
            youngs_modulus=100.0,
            poisson_ratio=0.25,
            gravity=(0.0, 0.0, 0.0),
            boundary_width=1,
        ),
        coupling=CouplingConfig(
            gamma=0.01,
            force_limit=1.0e-3,
            relative_velocity_limit=1.0e-2,
            mpm_substeps_per_lbm_step=1,
        ),
        output=OutputConfig(
            output_dir=output_dir,
            output_interval=2,
            output_format="npz",
            write_lbm_fields=True,
            write_mpm_particles=True,
            write_coupling_fields=True,
        ),
    )

    sim = FSISimulation(cfg)
    sim.initialize(
        mpm_box_lower=(4.0, 4.0, 4.0),
        mpm_box_upper=(5.0, 5.0, 5.0),
        mpm_box_spacing=1.0,
    )
    sim.run(steps=4, write_output=True, write_initial=True)

    snapshots = summarize_snapshots(output_dir)
    timeseries = extract_snapshot_timeseries(output_dir)
    csv_path = write_timeseries_csv(timeseries, output_dir / "timeseries.csv")
    json_path = write_timeseries_json(timeseries, output_dir / "timeseries.json")
    plot_path = plot_timeseries(timeseries, "mpm_center_of_mass_x", output_dir / "mpm_com_x.png")

    expected_steps = [0, 2, 4]
    if [info.step for info in snapshots] != expected_steps:
        raise SystemExit(f"unexpected snapshot steps: {[info.step for info in snapshots]}")

    print(f"snapshots: {len(snapshots)} in {output_dir}")
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")
    print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
