from __future__ import annotations

import numpy as np
import taichi as ti

from fsi.config import BoundaryConfig, CouplingConfig, LBMConfig, MPMConfig, SimulationConfig
from fsi.simulation import FSISimulation


def _periodic_lbm_config() -> LBMConfig:
    return LBMConfig(
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
    )


def main() -> None:
    ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False, offline_cache=False)

    config = SimulationConfig(
        num_steps=5,
        lbm_dt=1.0e-3,
        lbm=_periodic_lbm_config(),
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
            mpm_substeps_per_lbm_step=1,
            immersed_boundary_enabled=True,
            immersed_boundary_drag=0.1,
            immersed_boundary_fraction_threshold=0.01,
            immersed_boundary_max_force=1.0e-3,
            contact_enabled=True,
            contact_velocity_damping=0.05,
            contact_fraction_threshold=0.01,
        ),
    )

    solid = np.zeros((8, 8, 8), dtype=np.int32)
    solid[2, 3:6, 3:6] = 1

    sim = FSISimulation(config)
    sim.initialize_lbm(solid_np=solid)
    sim.initialize_mpm_from_numpy(
        positions=np.array(
            [
                [5.0, 4.0, 4.0],
                [2.5, 4.0, 4.0],
            ],
            dtype=np.float32,
        ),
        velocities=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        particle_mass=1.0,
        particle_volume=4.0,
    )

    for _ in range(config.num_steps):
        diag = sim.step()
        print(
            "step={step} fluid_speed={fluid:.6e} particle_speed={particle:.6e} "
            "ib_cells={ib_cells} ib_force={ib_force:.6e} contact={contact}".format(
                step=diag["step"],
                fluid=diag["lbm_max_velocity_norm"],
                particle=diag["mpm_max_velocity_norm"],
                ib_cells=diag["coupling_ib_active_cell_count"],
                ib_force=diag["coupling_ib_total_force_norm"],
                contact=diag["coupling_contact_candidate_count"],
            )
        )


if __name__ == "__main__":
    main()
