import numpy as np
import pytest

from fsi.config import MPMConfig
from fsi.mpm3d import MPMSolver3D


def _mpm_config(**overrides) -> MPMConfig:
    params = {
        "nx": 16,
        "ny": 16,
        "nz": 16,
        "max_particles": 10_000,
        "dt": 1.0e-3,
        "gravity": (0.0, 0.0, 0.0),
        "boundary_width": 2,
    }
    params.update(overrides)
    return MPMConfig(**params)


def test_mpm_initialize_box_particles():
    solver = MPMSolver3D(_mpm_config())

    count = solver.initialize_particles_box(
        lower=(6.0, 6.0, 6.0),
        upper=(10.0, 10.0, 10.0),
        spacing=1.0,
    )

    assert count > 0
    assert solver.particle_count() == count
    assert np.all(solver.active_numpy() == 1)

    positions = solver.positions_numpy()
    assert positions.shape == (count, 3)
    assert np.isfinite(positions).all()
    assert positions.min() >= 6.0
    assert positions.max() < 10.0


def test_mpm_initialize_too_many_particles_fails():
    solver = MPMSolver3D(_mpm_config(max_particles=10))

    with pytest.raises(ValueError):
        solver.initialize_particles_box(
            lower=(2.0, 2.0, 2.0),
            upper=(10.0, 10.0, 10.0),
            spacing=0.5,
        )


def test_mpm_substep_before_initialize_fails():
    solver = MPMSolver3D(_mpm_config())

    with pytest.raises(RuntimeError):
        solver.substep()


def test_mpm_invalid_substep_dt_fails():
    solver = MPMSolver3D(_mpm_config())
    solver.initialize_particles_box((6.0, 6.0, 6.0), (10.0, 10.0, 10.0), spacing=1.0)

    with pytest.raises(ValueError):
        solver.substep(0.0)


def test_mpm_no_gravity_center_of_mass_stable():
    solver = MPMSolver3D(_mpm_config(gravity=(0.0, 0.0, 0.0), dt=1.0e-3))
    solver.initialize_particles_box((6.0, 6.0, 6.0), (10.0, 10.0, 10.0), spacing=1.0)

    com0 = solver.center_of_mass()
    for _ in range(5):
        solver.substep()
    com1 = solver.center_of_mass()

    assert np.linalg.norm(com1 - com0) < 1.0e-3


def test_mpm_gravity_moves_center_of_mass_down():
    solver = MPMSolver3D(_mpm_config(gravity=(0.0, -1.0e-3, 0.0), dt=1.0e-2))
    solver.initialize_particles_box((6.0, 8.0, 6.0), (10.0, 12.0, 10.0), spacing=1.0)

    com0 = solver.center_of_mass()
    for _ in range(10):
        solver.substep()
    com1 = solver.center_of_mass()

    assert com1[1] < com0[1]


def test_mpm_multiple_steps_no_nan():
    solver = MPMSolver3D(_mpm_config(gravity=(0.0, -1.0e-3, 0.0), dt=1.0e-3))
    solver.initialize_particles_box((6.0, 8.0, 6.0), (10.0, 12.0, 10.0), spacing=1.0)

    for _ in range(20):
        solver.substep()

    assert np.isfinite(solver.positions_numpy()).all()
    assert np.isfinite(solver.velocities_numpy()).all()
    assert np.isfinite(solver.deformation_gradients_numpy()).all()


def test_mpm_particles_remain_inside_domain():
    cfg = _mpm_config(gravity=(0.0, -1.0e-3, 0.0), dt=1.0e-3)
    solver = MPMSolver3D(cfg)
    solver.initialize_particles_box((2.0, 2.0, 2.0), (5.0, 5.0, 5.0), spacing=1.0)

    for _ in range(20):
        solver.substep()

    positions = solver.positions_numpy()
    assert (positions[:, 0] >= 0.0).all()
    assert (positions[:, 0] <= cfg.nx - 1.0e-4).all()
    assert (positions[:, 1] >= 0.0).all()
    assert (positions[:, 1] <= cfg.ny - 1.0e-4).all()
    assert (positions[:, 2] >= 0.0).all()
    assert (positions[:, 2] <= cfg.nz - 1.0e-4).all()
