import numpy as np
import taichi as ti

from fsi import (
    BoundaryConfig,
    CouplingConfig,
    FSISimulation,
    LBMConfig,
    MPMConfig,
    SimulationConfig,
)


def main() -> None:
    ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False, offline_cache=False)

    cfg = SimulationConfig(
        num_steps=5,
        lbm_dt=1.0e-3,
        lbm=LBMConfig(
            nx=8,
            ny=8,
            nz=8,
            viscosity=0.1,
            rho0=1.0,
            force=(0.0, 0.0, 0.0),
            initial_velocity=(1.0e-3, 0.0, 0.0),
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
            gamma=10.0,
            force_limit=1.0e-3,
            relative_velocity_limit=1.0e-2,
            gamma_ramp_steps=5,
            min_valid_weight=1.0e-6,
            mpm_substeps_per_lbm_step=1,
        ),
    )

    sim = FSISimulation(cfg)
    sim.initialize_mpm_from_numpy(
        positions=np.array([[-0.75, 4.0, 4.0]], dtype=np.float32),
        velocities=np.zeros((1, 3), dtype=np.float32),
        particle_mass=1.0,
        particle_volume=1.0,
    )
    sim.initialize_lbm()

    sim.coupler.compute_coupling_forces(dt_ratio=1.0)
    print("initial coupling diagnostics:", sim.coupler.coupling_diagnostics())

    for diag in sim.run(steps=5):
        print(
            "step={step:04d}, gamma={gamma:.3e}, lbm_max|u|={umax:.3e}, "
            "mpm_max|v|={vmax:.3e}, unsupported={unsupported}, "
            "partial={partial}, clipped={clipped}".format(
                step=diag["step"],
                gamma=diag["coupling_effective_gamma"],
                umax=diag["lbm_max_velocity_norm"],
                vmax=diag["mpm_max_velocity_norm"],
                unsupported=diag["coupling_unsupported_particle_count"],
                partial=diag["coupling_partial_support_particle_count"],
                clipped=diag["coupling_clipped_particle_count"],
            )
        )


if __name__ == "__main__":
    main()
