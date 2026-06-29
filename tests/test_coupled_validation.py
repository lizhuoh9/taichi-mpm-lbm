import numpy as np
import pytest

from fsi.config import BoundaryConfig, CouplingConfig, LBMConfig, MPMConfig, SimulationConfig
from fsi.simulation import FSISimulation


def _simulation_config(
    *,
    enabled: bool = True,
    gamma: float = 0.05,
    initial_velocity: tuple[float, float, float] = (1.0e-4, 0.0, 0.0),
    lbm_dt: float = 1.0e-3,
    num_steps: int = 3,
) -> SimulationConfig:
    return SimulationConfig(
        num_steps=num_steps,
        lbm_dt=lbm_dt,
        lbm=LBMConfig(
            nx=8,
            ny=8,
            nz=8,
            viscosity=0.1,
            rho0=1.0,
            force=(0.0, 0.0, 0.0),
            initial_velocity=initial_velocity,
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
            enabled=enabled,
            gamma=gamma,
            mpm_substeps_per_lbm_step=1,
        ),
    )


def _initialized_simulation(config: SimulationConfig) -> FSISimulation:
    sim = FSISimulation(config)
    sim.initialize_mpm_from_numpy(
        positions=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
        velocities=np.zeros((1, 3), dtype=np.float32),
        particle_mass=1.0,
        particle_volume=1.0,
    )
    sim.initialize_lbm()
    return sim


def test_enabled_coupled_simulation_moves_particles_with_flow():
    sim = _initialized_simulation(
        _simulation_config(gamma=5.0, initial_velocity=(1.0e-3, 0.0, 0.0), lbm_dt=0.1)
    )
    initial_com = sim.mpm.center_of_mass()

    sim.run(steps=2)

    final_com = sim.mpm.center_of_mass()
    assert final_com[0] > initial_com[0]


def test_disabled_coupled_simulation_does_not_push_particles_without_gravity():
    sim = _initialized_simulation(
        _simulation_config(
            enabled=False,
            gamma=5.0,
            initial_velocity=(1.0e-3, 0.0, 0.0),
            lbm_dt=0.1,
        )
    )
    initial_com = sim.mpm.center_of_mass()

    sim.run(steps=2)

    final_com = sim.mpm.center_of_mass()
    assert final_com[0] == pytest.approx(initial_com[0], abs=1.0e-6)


def test_coupled_simulation_fields_remain_finite():
    sim = _initialized_simulation(_simulation_config(gamma=0.01, num_steps=5))

    sim.run()

    assert np.isfinite(sim.lbm.density_numpy()).all()
    assert np.isfinite(sim.lbm.velocity_numpy()).all()
    assert np.isfinite(sim.mpm.positions_numpy()).all()
    assert np.isfinite(sim.mpm.velocities_numpy()).all()
    assert np.isfinite(sim.mpm.deformation_gradients_numpy()).all()
    assert np.isfinite(sim.coupler.coupling_force_numpy()).all()


def test_solid_volume_fraction_is_bounded():
    sim = _initialized_simulation(_simulation_config(gamma=0.01))

    sim.run(steps=1)

    fraction = sim.coupler.solid_volume_fraction_numpy()
    assert fraction.min() >= 0.0
    assert fraction.max() <= 1.0


def test_lbm_mass_remains_finite_and_reasonably_stable():
    sim = _initialized_simulation(_simulation_config(gamma=0.01, num_steps=5))
    mass0 = sim.lbm.total_mass()

    sim.run()
    mass1 = sim.lbm.total_mass()

    assert np.isfinite(mass1)
    assert mass1 == pytest.approx(mass0, rel=1.0e-4, abs=1.0e-4)
