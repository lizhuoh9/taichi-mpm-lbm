import numpy as np
import pytest

from fsi.config import BoundaryConfig, CouplingConfig, LBMConfig, MPMConfig, SimulationConfig
from fsi.coupling import LBMMpmCoupler
from fsi.lbm3d import LBMSolver3D
from fsi.mpm3d import MPMSolver3D
from fsi.simulation import FSISimulation


def _simulation_config(**overrides) -> SimulationConfig:
    params = {
        "num_steps": 3,
        "lbm_dt": 1.0e-3,
        "lbm": LBMConfig(
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
        "mpm": MPMConfig(
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
        "coupling": CouplingConfig(gamma=0.01, mpm_substeps_per_lbm_step=1),
    }
    params.update(overrides)
    return SimulationConfig(**params)


def _initialized_simulation(**overrides) -> FSISimulation:
    sim = FSISimulation(_simulation_config(**overrides))
    sim.initialize(
        mpm_box_lower=(3.0, 3.0, 3.0),
        mpm_box_upper=(5.0, 5.0, 5.0),
        mpm_box_spacing=1.0,
    )
    return sim


def test_simulation_constructs_solvers_and_coupler():
    sim = FSISimulation(_simulation_config())

    assert isinstance(sim.lbm, LBMSolver3D)
    assert isinstance(sim.mpm, MPMSolver3D)
    assert isinstance(sim.coupler, LBMMpmCoupler)
    assert sim.step_index == 0
    assert sim.time == 0.0


def test_simulation_initializes_lbm_mpm_and_coupler():
    sim = _initialized_simulation()

    assert sim.lbm._initialized
    assert sim.mpm._initialized
    assert sim.mpm.particle_count() > 0
    assert sim.step_index == 0
    assert sim.time == 0.0


def test_simulation_step_advances_time_and_returns_diagnostics():
    sim = _initialized_simulation()

    diag = sim.step()

    assert diag["step"] == 1
    assert diag["time"] == pytest.approx(sim.config.lbm_dt)
    assert diag["mpm_particle_count"] == sim.mpm.particle_count()
    assert np.isfinite(diag["lbm_total_mass"])
    assert np.isfinite(diag["lbm_max_velocity_norm"])
    assert np.isfinite(diag["mpm_center_of_mass"]).all()
    assert np.isfinite(diag["total_particle_coupling_force"]).all()
    assert np.isfinite(diag["total_fluid_coupling_force"]).all()


def test_simulation_run_uses_config_num_steps():
    sim = _initialized_simulation(num_steps=3)

    history = sim.run()

    assert len(history) == 3
    assert sim.step_index == 3
    assert sim.time == pytest.approx(3.0 * sim.config.lbm_dt)
    assert history[-1]["step"] == 3


def test_simulation_run_accepts_override_steps():
    sim = _initialized_simulation(num_steps=5)

    history = sim.run(steps=2)

    assert len(history) == 2
    assert sim.step_index == 2
    assert sim.time == pytest.approx(2.0 * sim.config.lbm_dt)


def test_simulation_run_before_initialize_fails():
    sim = FSISimulation(_simulation_config())

    with pytest.raises(RuntimeError):
        sim.run(steps=1)


def test_simulation_initialize_requires_mpm_box_bounds():
    sim = FSISimulation(_simulation_config())

    with pytest.raises(ValueError):
        sim.initialize()


@pytest.mark.parametrize("steps", [0, -1])
def test_simulation_run_rejects_non_positive_steps(steps):
    sim = _initialized_simulation()

    with pytest.raises(ValueError):
        sim.run(steps=steps)
