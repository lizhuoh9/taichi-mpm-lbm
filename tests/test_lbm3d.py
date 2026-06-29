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
