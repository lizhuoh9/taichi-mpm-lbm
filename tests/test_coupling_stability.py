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
) -> LBMSolver3D:
    lbm = LBMSolver3D(_periodic_lbm_config(initial_velocity=velocity))
    lbm.initialize(solid_np=solid)
    return lbm


def _initialized_mpm(
    position: tuple[float, float, float] = (4.0, 4.0, 4.0),
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mass: float = 1.0,
) -> MPMSolver3D:
    mpm = MPMSolver3D(_mpm_config())
    mpm.initialize_particles_from_numpy(
        positions=np.array([position], dtype=np.float32),
        velocities=np.array([velocity], dtype=np.float32),
        particle_mass=mass,
        particle_volume=1.0,
    )
    return mpm


def _coupler(
    *,
    lbm_velocity: tuple[float, float, float] = (1.0, 0.0, 0.0),
    particle_position: tuple[float, float, float] = (4.0, 4.0, 4.0),
    particle_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    particle_mass: float = 1.0,
    solid: np.ndarray | None = None,
    config: CouplingConfig | None = None,
) -> LBMMpmCoupler:
    lbm = _initialized_lbm(velocity=lbm_velocity, solid=solid)
    mpm = _initialized_mpm(
        position=particle_position,
        velocity=particle_velocity,
        mass=particle_mass,
    )
    return LBMMpmCoupler(config or CouplingConfig(gamma=1.0), lbm, mpm)


def _force_balance(coupler: LBMMpmCoupler) -> np.ndarray:
    return (
        coupler.total_particle_coupling_force()
        + coupler.total_fluid_coupling_force() * coupler.cell_volume
    )


def test_force_limit_clips_particle_force():
    coupler = _coupler(config=CouplingConfig(gamma=100.0, force_limit=0.05))

    coupler.compute_coupling_forces(dt_ratio=1.0)

    force_norm = np.linalg.norm(coupler.mpm.particle_forces_numpy()[0])
    diagnostics = coupler.coupling_diagnostics()
    assert force_norm <= 0.05 + 1.0e-7
    assert diagnostics["clipped_particle_count"] == 1


def test_relative_velocity_limit_clips_before_force_computation():
    coupler = _coupler(
        particle_mass=2.0,
        config=CouplingConfig(gamma=10.0, relative_velocity_limit=0.01),
    )

    coupler.compute_coupling_forces(dt_ratio=1.0)

    force_norm = np.linalg.norm(coupler.mpm.particle_forces_numpy()[0])
    assert force_norm <= 10.0 * 2.0 * 0.01 + 1.0e-7
    assert coupler.coupling_diagnostics()["clipped_particle_count"] == 1


def test_clipped_force_still_balances_fluid_reaction():
    coupler = _coupler(config=CouplingConfig(gamma=100.0, force_limit=0.05))

    coupler.compute_coupling_forces(dt_ratio=1.0)

    assert _force_balance(coupler) == pytest.approx(np.zeros(3), rel=1.0e-5, abs=1.0e-6)


def test_gamma_ramp_reduces_initial_force():
    unramped = _coupler(config=CouplingConfig(gamma=10.0, gamma_ramp_steps=0))
    ramped = _coupler(config=CouplingConfig(gamma=10.0, gamma_ramp_steps=4))

    unramped.compute_coupling_forces(dt_ratio=1.0)
    ramped.compute_coupling_forces(dt_ratio=1.0)

    force_unramped = np.linalg.norm(unramped.mpm.particle_forces_numpy()[0])
    force_ramped = np.linalg.norm(ramped.mpm.particle_forces_numpy()[0])
    assert ramped.effective_gamma() == pytest.approx(2.5)
    assert force_ramped < force_unramped


def test_gamma_ramp_reaches_full_gamma_after_steps():
    coupler = _coupler(
        lbm_velocity=(1.0e-3, 0.0, 0.0),
        config=CouplingConfig(gamma=4.0, gamma_ramp_steps=4, force_limit=0.1),
    )

    assert coupler.effective_gamma() == pytest.approx(1.0)
    for _ in range(4):
        coupler.step(lbm_dt=1.0e-3)

    assert coupler.effective_gamma() == pytest.approx(4.0)


def test_near_domain_boundary_records_partial_support():
    coupler = _coupler(
        particle_position=(-0.75, 4.0, 4.0),
        config=CouplingConfig(gamma=1.0, min_valid_weight=1.0e-6),
    )

    coupler.compute_coupling_forces(dt_ratio=1.0)

    diagnostics = coupler.coupling_diagnostics()
    valid_weight = coupler.particle_valid_weights_numpy()[0]
    assert diagnostics["partial_support_particle_count"] == 1
    assert diagnostics["unsupported_particle_count"] == 0
    assert 1.0e-6 < valid_weight < 1.0
    assert coupler.particle_coupling_mask_numpy()[0] == 1
    assert np.isfinite(coupler.mpm.particle_forces_numpy()).all()
    assert _force_balance(coupler) == pytest.approx(np.zeros(3), rel=1.0e-5, abs=1.0e-6)


def test_solid_mask_partial_support_does_not_force_solid_cells():
    solid = np.zeros((8, 8, 8), dtype=np.int32)
    solid[4, 4, 4] = 1
    coupler = _coupler(solid=solid, config=CouplingConfig(gamma=1.0))

    coupler.compute_coupling_forces(dt_ratio=1.0)

    diagnostics = coupler.coupling_diagnostics()
    force = coupler.coupling_force_numpy()
    assert diagnostics["partial_support_particle_count"] == 1
    assert diagnostics["unsupported_particle_count"] == 0
    assert np.allclose(force[solid == 1], 0.0)
    assert _force_balance(coupler) == pytest.approx(np.zeros(3), rel=1.0e-5, abs=1.0e-6)


def test_full_solid_support_produces_unsupported_particle():
    solid = np.zeros((8, 8, 8), dtype=np.int32)
    solid[3:6, 3:6, 3:6] = 1
    coupler = _coupler(solid=solid, config=CouplingConfig(gamma=1.0))

    coupler.compute_coupling_forces(dt_ratio=1.0)

    diagnostics = coupler.coupling_diagnostics()
    assert diagnostics["unsupported_particle_count"] == 1
    assert diagnostics["partial_support_particle_count"] == 0
    assert coupler.particle_coupling_mask_numpy()[0] == 0
    assert np.allclose(coupler.mpm.particle_forces_numpy(), 0.0)
    assert np.allclose(coupler.coupling_force_numpy(), 0.0)


def test_high_gamma_limited_coupled_run_stays_finite():
    coupler = _coupler(
        lbm_velocity=(1.0e-3, 0.0, 0.0),
        config=CouplingConfig(
            gamma=50.0,
            force_limit=0.1,
            relative_velocity_limit=0.01,
            gamma_ramp_steps=3,
            mpm_substeps_per_lbm_step=1,
        ),
    )

    for _ in range(3):
        coupler.step(lbm_dt=1.0e-3)

    assert np.isfinite(coupler.lbm.density_numpy()).all()
    assert np.isfinite(coupler.lbm.velocity_numpy()).all()
    assert np.isfinite(coupler.mpm.positions_numpy()).all()
    assert np.isfinite(coupler.mpm.velocities_numpy()).all()
    assert coupler.coupling_diagnostics()["clipped_particle_count"] >= 0
