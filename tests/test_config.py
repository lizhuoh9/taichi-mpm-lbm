from pathlib import Path

import pytest

from fsi.config import BoundaryConfig, CouplingConfig, LBMConfig, MPMConfig, SimulationConfig
from fsi.coupling import LBMMpmCoupler
from fsi.mpm3d import MPMSolver3D
from fsi.simulation import FSISimulation


def test_default_simulation_config_validates():
    cfg = SimulationConfig()
    cfg.validate()


def test_mpm_dt_property():
    cfg = SimulationConfig(lbm_dt=1.0, coupling=CouplingConfig(mpm_substeps_per_lbm_step=4))
    assert cfg.mpm_dt == pytest.approx(0.25)


def test_invalid_lbm_viscosity_fails():
    cfg = LBMConfig(viscosity=0.0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_lbm_initial_velocity_fails():
    cfg = LBMConfig(initial_velocity=(0.0, 1.0))
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_mpm_poisson_ratio_fails():
    cfg = MPMConfig(poisson_ratio=0.5)
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_coupling_gamma_fails():
    cfg = CouplingConfig(gamma=-1.0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_grid_mismatch_fails():
    cfg = SimulationConfig(
        lbm=LBMConfig(nx=64, ny=32, nz=16),
        mpm=MPMConfig(nx=32, ny=32, nz=16),
    )

    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_boundary_velocity_fails():
    cfg = BoundaryConfig(velocity=(0.0, 1.0))
    with pytest.raises(ValueError):
        cfg.validate()


def test_remaining_placeholder_methods_raise_not_implemented():
    mpm = MPMSolver3D(MPMConfig())
    coupler = LBMMpmCoupler(CouplingConfig())
    simulation = FSISimulation(SimulationConfig())

    with pytest.raises(NotImplementedError):
        mpm.initialize_particles_box()
    with pytest.raises(NotImplementedError):
        mpm.substep(0.25)
    with pytest.raises(NotImplementedError):
        coupler.step(1.0)
    with pytest.raises(NotImplementedError):
        simulation.run()


def test_library_modules_do_not_call_taichi_init():
    for module_path in Path("fsi").glob("*.py"):
        source = module_path.read_text(encoding="utf-8")
        assert "ti.init(" not in source
        assert "taichi.init(" not in source
