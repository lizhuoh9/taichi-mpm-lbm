from pathlib import Path

import numpy as np
import pytest

from fsi.config import BoundaryConfig, CouplingConfig, LBMConfig, MPMConfig, OutputConfig, SimulationConfig
from fsi.coupling import LBMMpmCoupler
from fsi.lbm3d import LBMSolver3D
from fsi.mpm3d import MPMSolver3D
from fsi.output import SimulationOutputWriter
from fsi.postprocess import extract_snapshot_timeseries
from fsi.simulation import FSISimulation
from fsi.validation_cases import (
    run_contact_diagnostics_case,
    run_immersed_boundary_drag_case,
)


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
    velocity: tuple[float, float, float] = (1.0e-3, 0.0, 0.0),
    solid: np.ndarray | None = None,
) -> LBMSolver3D:
    lbm = LBMSolver3D(_periodic_lbm_config(initial_velocity=velocity))
    lbm.initialize(solid_np=solid)
    return lbm


def _initialized_mpm(
    position: tuple[float, float, float] = (4.0, 4.0, 4.0),
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    volume: float = 1.0,
) -> MPMSolver3D:
    mpm = MPMSolver3D(_mpm_config())
    mpm.initialize_particles_from_numpy(
        positions=np.array([position], dtype=np.float32),
        velocities=np.array([velocity], dtype=np.float32),
        particle_mass=1.0,
        particle_volume=volume,
    )
    return mpm


def _coupler(
    config: CouplingConfig,
    *,
    lbm_velocity: tuple[float, float, float] = (1.0e-3, 0.0, 0.0),
    particle_position: tuple[float, float, float] = (4.0, 4.0, 4.0),
    particle_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    particle_volume: float = 1.0,
    solid: np.ndarray | None = None,
) -> LBMMpmCoupler:
    return LBMMpmCoupler(
        config,
        _initialized_lbm(lbm_velocity, solid),
        _initialized_mpm(particle_position, particle_velocity, particle_volume),
    )


def test_invalid_immersed_boundary_config_fails():
    with pytest.raises(ValueError):
        CouplingConfig(immersed_boundary_drag=-1.0).validate()
    with pytest.raises(ValueError):
        CouplingConfig(immersed_boundary_fraction_threshold=-0.1).validate()
    with pytest.raises(ValueError):
        CouplingConfig(immersed_boundary_fraction_threshold=1.1).validate()
    with pytest.raises(ValueError):
        CouplingConfig(immersed_boundary_max_force=0.0).validate()


def test_invalid_contact_config_fails():
    with pytest.raises(ValueError):
        CouplingConfig(contact_velocity_damping=-0.1).validate()
    with pytest.raises(ValueError):
        CouplingConfig(contact_velocity_damping=1.1).validate()
    with pytest.raises(ValueError):
        CouplingConfig(contact_fraction_threshold=-0.1).validate()
    with pytest.raises(ValueError):
        CouplingConfig(contact_fraction_threshold=1.1).validate()


def test_ib_disabled_keeps_immersed_boundary_force_zero():
    coupler = _coupler(
        CouplingConfig(gamma=0.0, immersed_boundary_enabled=False),
        particle_volume=4.0,
    )

    coupler.compute_coupling_forces(dt_ratio=1.0)

    diagnostics = coupler.coupling_diagnostics()
    assert np.max(coupler.solid_volume_fraction_numpy()) > 0.0
    assert np.allclose(coupler.immersed_boundary_force_numpy(), 0.0)
    assert diagnostics["ib_active_cell_count"] == 0
    assert diagnostics["ib_total_force_norm"] == pytest.approx(0.0)


def test_ib_drag_force_opposes_fluid_velocity():
    coupler = _coupler(
        CouplingConfig(
            gamma=0.0,
            immersed_boundary_enabled=True,
            immersed_boundary_drag=0.5,
            immersed_boundary_fraction_threshold=0.01,
        ),
        particle_volume=4.0,
    )

    coupler.compute_coupling_forces(dt_ratio=1.0)

    diagnostics = coupler.coupling_diagnostics()
    assert diagnostics["ib_active_cell_count"] > 0
    assert diagnostics["ib_total_force_x"] < 0.0
    assert diagnostics["ib_total_force_norm"] > 0.0
    assert coupler.immersed_boundary_force_numpy()[..., 0].min() < 0.0


def test_ib_max_force_clips_per_cell_force():
    limit = 1.0e-5
    coupler = _coupler(
        CouplingConfig(
            gamma=0.0,
            immersed_boundary_enabled=True,
            immersed_boundary_drag=100.0,
            immersed_boundary_fraction_threshold=0.01,
            immersed_boundary_max_force=limit,
        ),
        particle_volume=4.0,
    )

    coupler.compute_coupling_forces(dt_ratio=1.0)

    force_norm = np.linalg.norm(coupler.immersed_boundary_force_numpy(), axis=-1)
    assert force_norm.max() <= limit + 1.0e-8
    assert coupler.coupling_diagnostics()["ib_clipped_cell_count"] > 0


def test_ib_force_ignores_static_solid_cells():
    solid = np.zeros((8, 8, 8), dtype=np.int32)
    solid[4, 4, 4] = 1
    coupler = _coupler(
        CouplingConfig(
            gamma=0.0,
            immersed_boundary_enabled=True,
            immersed_boundary_drag=0.5,
            immersed_boundary_fraction_threshold=0.0,
        ),
        particle_volume=4.0,
        solid=solid,
    )

    coupler.compute_coupling_forces(dt_ratio=1.0)

    assert np.allclose(coupler.immersed_boundary_force_numpy()[solid == 1], 0.0)


def test_coupled_step_with_ib_enabled_stays_finite():
    coupler = _coupler(
        CouplingConfig(
            gamma=0.01,
            mpm_substeps_per_lbm_step=1,
            immersed_boundary_enabled=True,
            immersed_boundary_drag=0.1,
            immersed_boundary_fraction_threshold=0.01,
            immersed_boundary_max_force=1.0e-3,
        ),
        particle_volume=2.0,
    )

    for _ in range(3):
        coupler.step(lbm_dt=1.0e-3)

    assert np.isfinite(coupler.lbm.density_numpy()).all()
    assert np.isfinite(coupler.lbm.velocity_numpy()).all()
    assert np.isfinite(coupler.mpm.positions_numpy()).all()
    assert np.isfinite(coupler.mpm.velocities_numpy()).all()
    assert np.isfinite(coupler.immersed_boundary_force_numpy()).all()


def test_contact_candidate_detection_with_static_solid_support():
    solid = np.zeros((8, 8, 8), dtype=np.int32)
    solid[3:6, 3:6, 3:6] = 1
    coupler = _coupler(
        CouplingConfig(
            gamma=0.0,
            contact_enabled=True,
            contact_velocity_damping=0.0,
            contact_fraction_threshold=0.01,
        ),
        particle_position=(4.0, 4.0, 4.0),
        solid=solid,
    )

    coupler.compute_coupling_forces(dt_ratio=1.0)

    assert coupler.coupling_diagnostics()["contact_candidate_count"] == 1
    assert coupler.particle_contact_mask_numpy().tolist() == [1]


def test_contact_damping_reduces_particle_speed():
    initial_speed = 0.1
    solid = np.zeros((8, 8, 8), dtype=np.int32)
    solid[3:6, 3:6, 3:6] = 1
    coupler = _coupler(
        CouplingConfig(
            gamma=0.0,
            contact_enabled=True,
            contact_velocity_damping=0.25,
            contact_fraction_threshold=0.01,
        ),
        particle_position=(4.0, 4.0, 4.0),
        particle_velocity=(initial_speed, 0.0, 0.0),
        solid=solid,
    )

    coupler.compute_coupling_forces(dt_ratio=1.0)

    final_speed = np.linalg.norm(coupler.mpm.velocities_numpy()[0])
    assert final_speed == pytest.approx(initial_speed * 0.75)
    assert coupler.coupling_diagnostics()["contact_damped_particle_count"] == 1


def test_output_and_postprocess_include_ib_contact_diagnostics(tmp_path):
    output = OutputConfig(output_dir=tmp_path, output_interval=1, output_format="npz")
    config = SimulationConfig(
        num_steps=1,
        lbm_dt=1.0e-3,
        lbm=_periodic_lbm_config(initial_velocity=(1.0e-3, 0.0, 0.0)),
        mpm=_mpm_config(),
        coupling=CouplingConfig(
            gamma=0.0,
            mpm_substeps_per_lbm_step=1,
            immersed_boundary_enabled=True,
            immersed_boundary_drag=0.1,
            immersed_boundary_fraction_threshold=0.01,
            contact_enabled=True,
            contact_velocity_damping=0.1,
            contact_fraction_threshold=0.01,
        ),
        output=output,
    )
    solid = np.zeros((8, 8, 8), dtype=np.int32)
    solid[3:6, 3:6, 3:6] = 1
    sim = FSISimulation(config)
    sim.initialize_lbm(solid_np=solid)
    sim.initialize_mpm_from_numpy(
        positions=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
        velocities=np.array([[0.1, 0.0, 0.0]], dtype=np.float32),
        particle_mass=1.0,
        particle_volume=4.0,
    )

    sim.step()
    path = sim.write_snapshot()[0]
    data = np.load(path)
    timeseries = extract_snapshot_timeseries(tmp_path)

    assert data["immersed_boundary_force"].shape == (8, 8, 8, 3)
    assert data["particle_contact_mask"].shape == (1,)
    assert int(data["ib_active_cell_count"]) >= 0
    assert int(data["contact_candidate_count"]) >= 1
    assert "ib_force_norm" in timeseries
    assert "ib_total_force_norm" in timeseries
    assert "contact_candidate_count" in timeseries
    assert "contact_damped_particle_count" in timeseries


def test_validation_cases_report_ib_contact_metrics():
    ib_report = run_immersed_boundary_drag_case()
    contact_report = run_contact_diagnostics_case()

    ib_metrics = {metric.name: metric.value for metric in ib_report.metrics}
    contact_metrics = {metric.name: metric.value for metric in contact_report.metrics}

    assert ib_report.passed
    assert ib_metrics["ib_active_cell_count"] >= 1.0
    assert ib_metrics["ib_total_force_x"] < 0.0
    assert contact_report.passed
    assert contact_metrics["contact_candidate_count"] >= 1.0
    assert contact_metrics["contact_damped_particle_count"] >= 1.0


def test_vtk_writer_accepts_ib_contact_fields(tmp_path):
    output = OutputConfig(output_dir=tmp_path, output_interval=1, output_format="vtk")
    sim = FSISimulation(
        SimulationConfig(
            num_steps=1,
            lbm_dt=1.0e-3,
            lbm=_periodic_lbm_config(initial_velocity=(1.0e-3, 0.0, 0.0)),
            mpm=_mpm_config(),
            coupling=CouplingConfig(
                gamma=0.0,
                immersed_boundary_enabled=True,
                immersed_boundary_drag=0.1,
                immersed_boundary_fraction_threshold=0.01,
                contact_enabled=True,
                contact_fraction_threshold=0.01,
            ),
            output=output,
        )
    )
    sim.initialize_lbm()
    sim.initialize_mpm_from_numpy(
        positions=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
        velocities=np.zeros((1, 3), dtype=np.float32),
        particle_mass=1.0,
        particle_volume=4.0,
    )

    paths = SimulationOutputWriter(output).write_snapshot(sim)

    assert any(Path(path).suffix == ".vti" for path in paths)
    assert any(Path(path).suffix == ".vtu" for path in paths)
