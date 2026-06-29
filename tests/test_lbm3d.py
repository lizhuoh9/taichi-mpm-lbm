import numpy as np
import pytest

from fsi.config import BoundaryConfig, LBMConfig
from fsi.lbm3d import LBMSolver3D


def _periodic_config(**overrides) -> LBMConfig:
    params = {
        "nx": 8,
        "ny": 8,
        "nz": 8,
        "x_left": BoundaryConfig("periodic"),
        "x_right": BoundaryConfig("periodic"),
        "y_left": BoundaryConfig("periodic"),
        "y_right": BoundaryConfig("periodic"),
        "z_left": BoundaryConfig("periodic"),
        "z_right": BoundaryConfig("periodic"),
    }
    params.update(overrides)
    return LBMConfig(**params)


def test_lbm_initialize_small_grid_no_nan():
    solver = LBMSolver3D(_periodic_config())
    solver.initialize()

    rho = solver.density_numpy()
    velocity = solver.velocity_numpy()

    assert rho.shape == (8, 8, 8)
    assert velocity.shape == (8, 8, 8, 3)
    assert np.isfinite(rho).all()
    assert np.isfinite(velocity).all()


def test_lbm_mass_conservation_no_force():
    solver = LBMSolver3D(_periodic_config(force=(0.0, 0.0, 0.0)))
    solver.initialize()

    mass0 = solver.total_mass()
    for _ in range(20):
        solver.step()
    mass1 = solver.total_mass()

    assert mass1 == pytest.approx(mass0, rel=1e-5, abs=1e-5)


def test_lbm_uniform_force_increases_mean_ux():
    solver = LBMSolver3D(_periodic_config(force=(1.0e-6, 0.0, 0.0)))
    solver.initialize()

    ux0 = float(solver.velocity_numpy()[..., 0].mean())
    for _ in range(20):
        solver.step()
    ux1 = float(solver.velocity_numpy()[..., 0].mean())

    assert ux1 > ux0


def test_lbm_solid_cells_remain_zero_velocity():
    cfg = _periodic_config()
    solid = np.zeros((cfg.nx, cfg.ny, cfg.nz), dtype=np.int32)
    solid[3:5, 3:5, 3:5] = 1

    solver = LBMSolver3D(cfg)
    solver.initialize(solid_np=solid)

    for _ in range(5):
        solver.step()

    velocity = solver.velocity_numpy()
    assert np.allclose(velocity[solid == 1], 0.0)


def test_lbm_invalid_solid_shape_fails():
    solver = LBMSolver3D(_periodic_config())
    bad_solid = np.zeros((8, 8, 7), dtype=np.int32)

    with pytest.raises(ValueError):
        solver.initialize(solid_np=bad_solid)


def test_lbm_sparse_storage_fails():
    with pytest.raises(NotImplementedError):
        LBMSolver3D(_periodic_config(use_sparse=True))


def test_lbm_nonperiodic_yz_boundary_fails():
    with pytest.raises(NotImplementedError):
        LBMSolver3D(_periodic_config(y_left=BoundaryConfig("pressure")))


def test_lbm_step_before_initialize_fails():
    solver = LBMSolver3D(_periodic_config())

    with pytest.raises(RuntimeError):
        solver.step()


def test_lbm_force_field_initialized_from_config_force():
    cfg = _periodic_config(force=(1.0e-6, 2.0e-6, -3.0e-6))
    solver = LBMSolver3D(cfg)
    solver.initialize()

    force = solver.force_numpy()

    assert force.shape == (8, 8, 8, 3)
    assert np.allclose(force[..., 0], 1.0e-6)
    assert np.allclose(force[..., 1], 2.0e-6)
    assert np.allclose(force[..., 2], -3.0e-6)
    assert solver.total_force() == pytest.approx(
        np.array([512.0e-6, 1024.0e-6, -1536.0e-6]), rel=1e-5, abs=1e-9
    )


def test_lbm_clear_force_sets_force_to_zero():
    solver = LBMSolver3D(_periodic_config(force=(1.0e-6, 0.0, 0.0)))
    solver.initialize()

    solver.clear_force()

    assert np.allclose(solver.force_numpy(), 0.0)
    assert solver.total_force() == pytest.approx(np.zeros(3))


def test_lbm_reset_force_restores_base_force():
    solver = LBMSolver3D(_periodic_config(force=(1.0e-6, 0.0, 0.0)))
    solver.initialize()

    solver.clear_force()
    solver.reset_force()
    force = solver.force_numpy()

    assert np.allclose(force[..., 0], 1.0e-6)
    assert np.allclose(force[..., 1:], 0.0)


def test_lbm_set_and_add_uniform_force():
    solver = LBMSolver3D(_periodic_config(force=(0.0, 0.0, 0.0)))
    solver.initialize()

    solver.set_uniform_force((1.0e-6, 2.0e-6, 0.0))
    solver.add_uniform_force((3.0e-6, -1.0e-6, 4.0e-6))
    force = solver.force_numpy()

    assert np.allclose(force[..., 0], 4.0e-6)
    assert np.allclose(force[..., 1], 1.0e-6)
    assert np.allclose(force[..., 2], 4.0e-6)


def test_lbm_set_force_from_numpy_accepts_local_field():
    solver = LBMSolver3D(_periodic_config(force=(0.0, 0.0, 0.0)))
    solver.initialize()

    local_force = np.zeros((8, 8, 8, 3), dtype=np.float32)
    local_force[2:6, :, :, 0] = 1.0e-6

    solver.set_force_from_numpy(local_force)

    assert np.allclose(solver.force_numpy(), local_force)
    assert solver.total_force() == pytest.approx(
        np.array([256.0e-6, 0.0, 0.0]), rel=1e-5, abs=1e-9
    )


def test_lbm_invalid_force_shape_fails():
    solver = LBMSolver3D(_periodic_config())
    solver.initialize()

    bad_force = np.zeros((8, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError):
        solver.set_force_from_numpy(bad_force)


def test_lbm_invalid_force_vector_fails():
    solver = LBMSolver3D(_periodic_config())
    solver.initialize()

    with pytest.raises(ValueError):
        solver.set_uniform_force((1.0e-6, 0.0))
    with pytest.raises(ValueError):
        solver.add_uniform_force((1.0e-6, 0.0))


def test_lbm_local_force_drives_forced_region_more_than_unforced_region():
    solver = LBMSolver3D(_periodic_config(force=(0.0, 0.0, 0.0)))
    solver.initialize()

    local_force = np.zeros((8, 8, 8, 3), dtype=np.float32)
    local_force[2:6, :, :, 0] = 1.0e-6
    solver.set_force_from_numpy(local_force)

    solver.step()

    velocity = solver.velocity_numpy()
    forced_mean = float(velocity[2:6, :, :, 0].mean())
    unforced_mean = float(
        np.concatenate([velocity[0:2, :, :, 0].ravel(), velocity[6:8, :, :, 0].ravel()]).mean()
    )

    assert forced_mean > unforced_mean


def test_lbm_force_is_zero_on_solid_cells_after_force_operations():
    cfg = _periodic_config(force=(1.0e-6, 0.0, 0.0))
    solid = np.zeros((8, 8, 8), dtype=np.int32)
    solid[3:5, 3:5, 3:5] = 1

    solver = LBMSolver3D(cfg)
    solver.initialize(solid_np=solid)
    assert np.allclose(solver.force_numpy()[solid == 1], 0.0)

    solver.set_uniform_force((2.0e-6, 0.0, 0.0))
    assert np.allclose(solver.force_numpy()[solid == 1], 0.0)

    solver.add_uniform_force((1.0e-6, 0.0, 0.0))
    assert np.allclose(solver.force_numpy()[solid == 1], 0.0)

    local_force = np.full((8, 8, 8, 3), 3.0e-6, dtype=np.float32)
    solver.set_force_from_numpy(local_force)
    force = solver.force_numpy()

    assert np.allclose(force[solid == 1], 0.0)
    assert np.allclose(force[solid == 0], 3.0e-6)
