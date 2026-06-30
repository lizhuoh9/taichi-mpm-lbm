from pathlib import Path

import pytest

from fsi.config import (
    BoundaryConfig,
    CouplingConfig,
    LBMConfig,
    MPMConfig,
    OutputConfig,
    SimulationConfig,
)


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


def test_invalid_mpm_boundary_width_fails():
    cfg = MPMConfig(boundary_width=0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_mpm_boundary_damping_fails():
    cfg = MPMConfig(boundary_damping=1.5)
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_coupling_gamma_fails():
    cfg = CouplingConfig(gamma=-1.0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_coupling_force_limit_fails():
    cfg = CouplingConfig(force_limit=0.0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_coupling_relative_velocity_limit_fails():
    cfg = CouplingConfig(relative_velocity_limit=0.0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_coupling_gamma_ramp_steps_fails():
    cfg = CouplingConfig(gamma_ramp_steps=-1)
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_coupling_min_valid_weight_fails():
    with pytest.raises(ValueError):
        CouplingConfig(min_valid_weight=0.0).validate()
    with pytest.raises(ValueError):
        CouplingConfig(min_valid_weight=1.1).validate()


def test_invalid_output_interval_fails():
    cfg = OutputConfig(output_interval=0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_output_format_fails():
    cfg = OutputConfig(output_format="bad")
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


def test_simulation_config_defaults_are_consistent():
    cfg = SimulationConfig()

    assert cfg.num_steps == 1000
    assert cfg.lbm.nx == cfg.mpm.nx


def test_library_modules_do_not_call_taichi_init():
    for module_path in Path("fsi").glob("*.py"):
        source = module_path.read_text(encoding="utf-8")
        assert "ti.init(" not in source
        assert "taichi.init(" not in source
