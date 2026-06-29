import numpy as np
import pytest

from fsi.config import BoundaryConfig, CouplingConfig, LBMConfig, MPMConfig
from fsi.coupling import LBMMpmCoupler
from fsi.lbm3d import LBMSolver3D
from fsi.mpm3d import MPMSolver3D


def _periodic_lbm_config(**overrides) -> LBMConfig:
    params = {
        "nx": 8,
        "ny": 8,
        "nz": 8,
        "viscosity": 0.1,
        "rho0": 1.0,
        "force": (0.0, 0.0, 0.0),
        "initial_velocity": (0.0, 0.0, 0.0),
        "x_left": BoundaryConfig("periodic"),
        "x_right": BoundaryConfig("periodic"),
        "y_left": BoundaryConfig("periodic"),
        "y_right": BoundaryConfig("periodic"),
        "z_left": BoundaryConfig("periodic"),
        "z_right": BoundaryConfig("periodic"),
    }
    params.update(overrides)
    return LBMConfig(**params)


def _mpm_config(**overrides) -> MPMConfig:
    params = {
        "nx": 8,
        "ny": 8,
        "nz": 8,
        "max_particles": 128,
        "dx": 1.0,
        "dt": 1.0e-3,
        "density": 1.0,
        "youngs_modulus": 100.0,
        "poisson_ratio": 0.25,
        "gravity": (0.0, 0.0, 0.0),
        "boundary_width": 1,
    }
    params.update(overrides)
    return MPMConfig(**params)


def _initialized_lbm(
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    solid: np.ndarray | None = None,
    **overrides,
) -> LBMSolver3D:
    cfg = _periodic_lbm_config(initial_velocity=velocity, **overrides)
    lbm = LBMSolver3D(cfg)
    lbm.initialize(solid_np=solid)
    return lbm


def _initialized_mpm(
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    **overrides,
) -> MPMSolver3D:
    mpm = MPMSolver3D(_mpm_config(**overrides))
    mpm.initialize_particles_from_numpy(
        positions=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
        velocities=np.array([velocity], dtype=np.float32),
        particle_mass=2.0,
        particle_volume=1.0,
    )
    return mpm


def test_coupler_validates_matching_grid_shape():
    lbm = _initialized_lbm()
    mpm = _initialized_mpm(nx=9)

    with pytest.raises(ValueError):
        LBMMpmCoupler(CouplingConfig(), lbm, mpm)


def test_gamma_zero_produces_zero_coupling_force():
    lbm = _initialized_lbm(velocity=(1.0e-4, 0.0, 0.0))
    mpm = _initialized_mpm()
    coupler = LBMMpmCoupler(CouplingConfig(gamma=0.0), lbm, mpm)

    coupler.compute_coupling_forces()

    assert np.allclose(mpm.particle_forces_numpy(), 0.0)
    assert np.allclose(coupler.coupling_force_numpy(), 0.0)
    assert coupler.total_particle_coupling_force() == pytest.approx(np.zeros(3))
    assert coupler.total_fluid_coupling_force() == pytest.approx(np.zeros(3))


def test_fluid_velocity_drives_particle_in_correct_direction():
    lbm = _initialized_lbm(velocity=(1.0e-4, 0.0, 0.0))
    mpm = _initialized_mpm()
    coupler = LBMMpmCoupler(CouplingConfig(gamma=0.05), lbm, mpm)

    coupler.compute_coupling_forces()

    total_particle_force = coupler.total_particle_coupling_force()
    total_fluid_force = coupler.total_fluid_coupling_force()

    assert total_particle_force[0] > 0.0
    assert total_fluid_force[0] < 0.0
    assert total_particle_force[1:] == pytest.approx(np.zeros(2), abs=1.0e-10)
    assert total_fluid_force[1:] == pytest.approx(np.zeros(2), abs=1.0e-10)


def test_coupling_force_balance_for_single_compute():
    lbm = _initialized_lbm(velocity=(1.0e-4, 0.0, 0.0))
    mpm = _initialized_mpm()
    coupler = LBMMpmCoupler(CouplingConfig(gamma=0.05), lbm, mpm)

    coupler.compute_coupling_forces(dt_ratio=1.0)

    balance = (
        coupler.total_particle_coupling_force()
        + coupler.total_fluid_coupling_force() * coupler.cell_volume
    )
    assert balance == pytest.approx(np.zeros(3), rel=1.0e-5, abs=1.0e-7)


def test_coupling_force_is_zero_on_lbm_solid_cells():
    solid = np.zeros((8, 8, 8), dtype=np.int32)
    solid[4, 4, 4] = 1
    lbm = _initialized_lbm(velocity=(1.0e-4, 0.0, 0.0), solid=solid)
    mpm = _initialized_mpm()
    coupler = LBMMpmCoupler(CouplingConfig(gamma=0.05), lbm, mpm)

    coupler.compute_coupling_forces()

    force = coupler.coupling_force_numpy()
    assert np.allclose(force[solid == 1], 0.0)
    assert coupler.total_particle_coupling_force()[0] > 0.0
    assert coupler.total_fluid_coupling_force()[0] < 0.0


def test_coupled_step_runs_finite():
    lbm = _initialized_lbm(velocity=(1.0e-5, 0.0, 0.0))
    mpm = _initialized_mpm()
    coupler = LBMMpmCoupler(
        CouplingConfig(gamma=0.01, mpm_substeps_per_lbm_step=2),
        lbm,
        mpm,
    )

    for _ in range(3):
        coupler.step(lbm_dt=1.0e-3)

    positions = mpm.positions_numpy()
    assert np.isfinite(lbm.density_numpy()).all()
    assert np.isfinite(lbm.velocity_numpy()).all()
    assert np.isfinite(mpm.positions_numpy()).all()
    assert np.isfinite(mpm.velocities_numpy()).all()
    assert (positions >= 0.0).all()
    assert (positions <= 8.0).all()


def test_disabled_coupling_steps_without_force_exchange():
    lbm = _initialized_lbm(velocity=(1.0e-4, 0.0, 0.0))
    mpm = _initialized_mpm()
    coupler = LBMMpmCoupler(
        CouplingConfig(enabled=False, gamma=0.05, mpm_substeps_per_lbm_step=2),
        lbm,
        mpm,
    )

    coupler.step(lbm_dt=1.0e-3)

    assert np.allclose(mpm.particle_forces_numpy(), 0.0)
    assert np.allclose(coupler.coupling_force_numpy(), 0.0)
    assert np.isfinite(lbm.density_numpy()).all()
    assert np.isfinite(mpm.positions_numpy()).all()


@pytest.mark.parametrize("lbm_dt", [0.0, -1.0])
def test_invalid_coupled_step_dt_fails(lbm_dt):
    lbm = _initialized_lbm()
    mpm = _initialized_mpm()
    coupler = LBMMpmCoupler(CouplingConfig(), lbm, mpm)

    with pytest.raises(ValueError):
        coupler.step(lbm_dt=lbm_dt)


def test_coupler_reports_uninitialized_solvers():
    lbm = LBMSolver3D(_periodic_lbm_config())
    mpm = _initialized_mpm()
    coupler = LBMMpmCoupler(CouplingConfig(), lbm, mpm)

    with pytest.raises(RuntimeError):
        coupler.compute_coupling_forces()

    lbm.initialize()
    mpm = MPMSolver3D(_mpm_config())
    coupler = LBMMpmCoupler(CouplingConfig(), lbm, mpm)

    with pytest.raises(RuntimeError):
        coupler.step(lbm_dt=1.0e-3)
