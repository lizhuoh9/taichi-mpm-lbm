from pathlib import Path

import numpy as np
import pytest

from fsi.config import BoundaryConfig, CouplingConfig, LBMConfig, MPMConfig, OutputConfig, SimulationConfig
from fsi.output import SimulationOutputWriter
from fsi.simulation import FSISimulation


def _simulation_config(output: OutputConfig, **overrides) -> SimulationConfig:
    params = {
        "num_steps": 5,
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
        "output": output,
    }
    params.update(overrides)
    return SimulationConfig(**params)


def _initialized_simulation(output: OutputConfig, **overrides) -> FSISimulation:
    sim = FSISimulation(_simulation_config(output, **overrides))
    sim.initialize_mpm_from_numpy(
        positions=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
        velocities=np.zeros((1, 3), dtype=np.float32),
        particle_mass=1.0,
        particle_volume=1.0,
    )
    sim.initialize_lbm()
    return sim


def _npz_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("*.npz"))


def test_output_writer_should_write_respects_interval(tmp_path):
    writer = SimulationOutputWriter(OutputConfig(output_dir=tmp_path, output_interval=2))

    assert not writer.should_write(1)
    assert writer.should_write(2)
    assert not writer.should_write(3)
    assert writer.should_write(4)


def test_output_writer_writes_npz_snapshot(tmp_path):
    output = OutputConfig(output_dir=tmp_path, output_interval=1, output_format="npz")
    sim = _initialized_simulation(output)
    sim.step()

    paths = sim.write_snapshot()

    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].name == "snapshot_000001.npz"

    data = np.load(paths[0])
    assert int(data["step"]) == 1
    assert float(data["time"]) == pytest.approx(sim.time)
    assert data["lbm_shape"].tolist() == [8, 8, 8]
    assert int(data["mpm_particle_count"]) == sim.mpm.particle_count()
    assert data["lbm_density"].shape == (8, 8, 8)
    assert data["lbm_velocity"].shape == (8, 8, 8, 3)
    assert data["mpm_positions"].shape == (1, 3)
    assert data["mpm_velocities"].shape == (1, 3)
    assert data["coupling_force"].shape == (8, 8, 8, 3)
    assert data["solid_volume_fraction"].shape == (8, 8, 8)


def test_output_respects_field_flags(tmp_path):
    output = OutputConfig(
        output_dir=tmp_path,
        output_interval=1,
        output_format="npz",
        write_lbm_fields=False,
        write_mpm_particles=True,
        write_coupling_fields=False,
    )
    sim = _initialized_simulation(output)

    path = sim.write_snapshot()[0]
    data = np.load(path)

    assert "lbm_density" not in data
    assert "mpm_positions" in data
    assert "coupling_force" not in data


def test_run_writes_at_interval_and_initial_when_requested(tmp_path):
    output = OutputConfig(output_dir=tmp_path, output_interval=2, output_format="npz")
    sim = _initialized_simulation(output)

    history = sim.run(steps=5, write_output=True, write_initial=True)

    assert len(history) == 5
    assert _npz_names(tmp_path) == [
        "snapshot_000000.npz",
        "snapshot_000002.npz",
        "snapshot_000004.npz",
    ]


def test_run_does_not_write_by_default(tmp_path):
    output = OutputConfig(output_dir=tmp_path, output_interval=1, output_format="npz")
    sim = _initialized_simulation(output)

    sim.run(steps=2)

    assert list(tmp_path.iterdir()) == []


def test_write_snapshot_requires_initialization(tmp_path):
    output = OutputConfig(output_dir=tmp_path, output_interval=1, output_format="npz")
    sim = FSISimulation(_simulation_config(output))

    with pytest.raises(RuntimeError):
        sim.write_snapshot()


def test_output_format_vtk_smoke(tmp_path):
    output = OutputConfig(output_dir=tmp_path, output_interval=1, output_format="vtk")
    sim = _initialized_simulation(output)

    paths = sim.write_snapshot()

    assert len(paths) == 2
    assert all(path.exists() for path in paths)
    assert {path.suffix for path in paths} == {".vti", ".vtu"}


def test_vtk_output_respects_field_flags(tmp_path):
    output = OutputConfig(
        output_dir=tmp_path,
        output_interval=1,
        output_format="vtk",
        write_lbm_fields=False,
        write_mpm_particles=False,
        write_coupling_fields=True,
    )
    sim = _initialized_simulation(output)

    paths = sim.write_snapshot()

    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].name.startswith("fluid_")
    assert paths[0].suffix == ".vti"


def test_vtk_output_returns_no_files_when_all_groups_disabled(tmp_path):
    output = OutputConfig(
        output_dir=tmp_path,
        output_interval=1,
        output_format="vtk",
        write_lbm_fields=False,
        write_mpm_particles=False,
        write_coupling_fields=False,
    )
    sim = _initialized_simulation(output)

    paths = sim.write_snapshot()

    assert paths == []
    assert list(tmp_path.iterdir()) == []


def test_output_format_both_writes_npz_and_vtk(tmp_path):
    output = OutputConfig(output_dir=tmp_path, output_interval=1, output_format="both")
    sim = _initialized_simulation(output)

    paths = sim.write_snapshot()

    assert any(path.name == "snapshot_000000.npz" for path in paths)
    assert any(path.suffix == ".vti" for path in paths)
    assert any(path.suffix == ".vtu" for path in paths)
