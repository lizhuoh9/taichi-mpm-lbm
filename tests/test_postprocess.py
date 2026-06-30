import pytest


def _write_minimal_snapshot(path, step: int, time: float = 0.0) -> None:
    import numpy as np

    np.savez_compressed(
        path,
        step=np.array(step, dtype=np.int64),
        time=np.array(time, dtype=np.float64),
        lbm_shape=np.array([8, 8, 8], dtype=np.int32),
        mpm_particle_count=np.array(1, dtype=np.int32),
    )


def _write_full_schema_snapshot(path, step: int) -> None:
    import numpy as np

    density = np.full((8, 8, 8), 1.0 + step * 1.0e-4, dtype=np.float32)
    lbm_velocity = np.zeros((8, 8, 8, 3), dtype=np.float32)
    lbm_velocity[..., 0] = step * 1.0e-5
    mpm_position = np.array([[4.0 + step * 1.0e-3, 4.0, 4.0]], dtype=np.float32)
    mpm_velocity = np.array([[step * 1.0e-4, 0.0, 0.0]], dtype=np.float32)
    coupling_force = np.zeros((8, 8, 8, 3), dtype=np.float32)
    coupling_force[4, 4, 4, 0] = step * 1.0e-6

    np.savez_compressed(
        path,
        step=np.array(step, dtype=np.int64),
        time=np.array(step * 1.0e-3, dtype=np.float64),
        lbm_shape=np.array([8, 8, 8], dtype=np.int32),
        mpm_particle_count=np.array(1, dtype=np.int32),
        lbm_density=density,
        lbm_velocity=lbm_velocity,
        lbm_force=np.zeros((8, 8, 8, 3), dtype=np.float32),
        lbm_solid=np.zeros((8, 8, 8), dtype=np.int32),
        mpm_positions=mpm_position,
        mpm_velocities=mpm_velocity,
        mpm_particle_forces=np.zeros((1, 3), dtype=np.float32),
        mpm_deformation_gradients=np.eye(3, dtype=np.float32)[None, :, :],
        mpm_active=np.ones(1, dtype=np.int32),
        coupling_force=coupling_force,
        solid_volume_fraction=np.zeros((8, 8, 8), dtype=np.float32),
        total_particle_coupling_force=np.array([step * 1.0e-6, 0.0, 0.0], dtype=np.float32),
        total_fluid_coupling_force=np.array([-step * 1.0e-6, 0.0, 0.0], dtype=np.float32),
        coupling_particle_valid_weight=np.ones(1, dtype=np.float32),
        coupling_particle_mask=np.ones(1, dtype=np.int32),
        coupling_unsupported_particle_count=np.array(0, dtype=np.int32),
        coupling_partial_support_particle_count=np.array(step > 0, dtype=np.int32),
        coupling_clipped_particle_count=np.array(step // 2, dtype=np.int32),
    )


def test_list_npz_snapshots_sorts_by_numeric_step(tmp_path):
    from fsi.postprocess import list_npz_snapshots

    _write_minimal_snapshot(tmp_path / "snapshot_000010.npz", 10)
    _write_minimal_snapshot(tmp_path / "snapshot_000000.npz", 0)
    _write_minimal_snapshot(tmp_path / "snapshot_000002.npz", 2)

    snapshots = list_npz_snapshots(tmp_path)

    assert [path.name for path in snapshots] == [
        "snapshot_000000.npz",
        "snapshot_000002.npz",
        "snapshot_000010.npz",
    ]


def test_load_snapshot_and_info(tmp_path):
    import pytest

    from fsi.postprocess import load_npz_snapshot, snapshot_info

    path = tmp_path / "snapshot_000004.npz"
    _write_minimal_snapshot(path, 4, time=0.25)

    data = load_npz_snapshot(path)
    info = snapshot_info(path)

    assert set(data) == {"step", "time", "lbm_shape", "mpm_particle_count"}
    assert int(data["step"]) == 4
    assert info.path == path
    assert info.step == 4
    assert info.time == pytest.approx(0.25)
    assert info.particle_count == 1
    assert info.lbm_shape == (8, 8, 8)
    assert info.to_dict()["lbm_shape"] == (8, 8, 8)


def test_summarize_snapshots_returns_metadata(tmp_path):
    from fsi.postprocess import summarize_snapshots

    _write_minimal_snapshot(tmp_path / "snapshot_000002.npz", 2)
    _write_minimal_snapshot(tmp_path / "snapshot_000000.npz", 0)

    infos = summarize_snapshots(tmp_path)

    assert [info.step for info in infos] == [0, 2]
    assert all(info.lbm_shape == (8, 8, 8) for info in infos)


def test_extract_snapshot_timeseries_from_full_schema_snapshots(tmp_path):
    from fsi.postprocess import extract_snapshot_timeseries

    for step in (0, 2, 4):
        _write_full_schema_snapshot(tmp_path / f"snapshot_{step:06d}.npz", step)

    timeseries = extract_snapshot_timeseries(tmp_path)

    assert timeseries["step"].tolist() == [0, 2, 4]
    assert timeseries["mpm_particle_count"].tolist() == [1, 1, 1]
    assert "lbm_mean_density" in timeseries
    assert "lbm_total_mass_estimate" in timeseries
    assert "lbm_max_velocity_norm" in timeseries
    assert "mpm_center_of_mass_x" in timeseries
    assert "mpm_center_of_mass_y" in timeseries
    assert "mpm_center_of_mass_z" in timeseries
    assert "mpm_max_velocity_norm" in timeseries
    assert "coupling_force_norm" in timeseries
    assert "coupling_clipped_particle_count" in timeseries
    assert len(timeseries["time"]) == 3


def test_extract_snapshot_timeseries_tolerates_disabled_optional_groups(tmp_path):
    import numpy as np

    from fsi.postprocess import extract_snapshot_timeseries

    for step in (0, 2, 4):
        np.savez_compressed(
            tmp_path / f"snapshot_{step:06d}.npz",
            step=np.array(step, dtype=np.int64),
            time=np.array(step * 1.0e-3, dtype=np.float64),
            lbm_shape=np.array([8, 8, 8], dtype=np.int32),
            mpm_particle_count=np.array(1, dtype=np.int32),
            mpm_positions=np.array([[4.0 + step * 1.0e-3, 4.0, 4.0]], dtype=np.float32),
            mpm_velocities=np.zeros((1, 3), dtype=np.float32),
        )

    timeseries = extract_snapshot_timeseries(tmp_path)

    assert timeseries["step"].tolist() == [0, 2, 4]
    assert "mpm_center_of_mass_x" in timeseries
    assert "lbm_mean_density" not in timeseries
    assert "coupling_force_norm" not in timeseries
    assert "coupling_clipped_particle_count" not in timeseries


def test_summarize_coupling_diagnostics_extracts_step9_counts(tmp_path):
    from fsi.postprocess import summarize_coupling_diagnostics

    for step in (0, 2, 4):
        _write_full_schema_snapshot(tmp_path / f"snapshot_{step:06d}.npz", step)

    diagnostics = summarize_coupling_diagnostics(tmp_path)

    assert diagnostics["step"].tolist() == [0, 2, 4]
    assert "time" in diagnostics
    assert "coupling_force_norm" in diagnostics
    assert "total_particle_coupling_force_norm" in diagnostics
    assert "total_fluid_coupling_force_norm" in diagnostics
    assert "coupling_unsupported_particle_count" in diagnostics
    assert "coupling_partial_support_particle_count" in diagnostics
    assert "coupling_clipped_particle_count" in diagnostics


def test_timeseries_csv_and_json_writers(tmp_path):
    import json

    import numpy as np

    from fsi.postprocess import write_timeseries_csv, write_timeseries_json

    timeseries = {
        "time": np.array([0.0, 1.0], dtype=np.float64),
        "step": np.array([0, 1], dtype=np.int64),
        "mpm_particle_count": np.array([1, 1], dtype=np.int32),
    }

    csv_path = write_timeseries_csv(timeseries, tmp_path / "timeseries.csv")
    json_path = write_timeseries_json(timeseries, tmp_path / "timeseries.json")

    csv_text = csv_path.read_text(encoding="utf-8")
    json_data = json.loads(json_path.read_text(encoding="utf-8"))

    assert csv_text.splitlines()[0].startswith("step,time")
    assert "mpm_particle_count" in csv_text.splitlines()[0]
    assert json_data["step"] == [0, 1]
    assert json_data["time"] == [0.0, 1.0]


def test_timeseries_csv_rejects_length_mismatch(tmp_path):
    import numpy as np

    from fsi.postprocess import write_timeseries_csv

    with pytest.raises(ValueError):
        write_timeseries_csv(
            {
                "step": np.array([0, 1]),
                "time": np.array([0.0]),
            },
            tmp_path / "bad.csv",
        )


def test_validation_summary_loading_and_flattening(tmp_path):
    import json

    from fsi.postprocess import (
        load_validation_summary,
        validation_summary_table,
        write_validation_summary_csv,
    )

    summary_path = tmp_path / "validation_summary.json"
    summary_path.write_text(
        json.dumps(
            [
                {
                    "case_name": "coupling_force_limit",
                    "passed": True,
                    "metrics": [
                        {
                            "name": "force_balance_norm",
                            "value": 1.0e-8,
                            "lower": None,
                            "upper": 1.0e-6,
                            "passed": True,
                            "units": "",
                            "description": "balanced force",
                        }
                    ],
                    "metadata": {"step": 10},
                }
            ]
        ),
        encoding="utf-8",
    )

    reports = load_validation_summary(summary_path)
    rows = validation_summary_table(reports)
    csv_path = write_validation_summary_csv(rows, tmp_path / "validation_summary.csv")

    assert len(reports) == 1
    assert rows == [
        {
            "case_name": "coupling_force_limit",
            "case_passed": True,
            "metric_name": "force_balance_norm",
            "value": 1.0e-8,
            "lower": None,
            "upper": 1.0e-6,
            "metric_passed": True,
            "units": "",
            "description": "balanced force",
        }
    ]
    assert "case_name,case_passed,metric_name" in csv_path.read_text(encoding="utf-8")


def test_load_validation_summary_rejects_non_list_json(tmp_path):
    import json

    from fsi.postprocess import load_validation_summary

    path = tmp_path / "bad_summary.json"
    path.write_text(json.dumps({"case_name": "not-a-list"}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_validation_summary(path)


def test_plot_timeseries_writes_png(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "particles.png"
    code = (
        "import numpy as np; "
        "from fsi.postprocess import plot_timeseries; "
        f"output_path = {str(path)!r}; "
        "plot_timeseries("
        "{'step': np.array([0, 1, 2]), 'mpm_particle_count': np.array([1, 2, 3])}, "
        "'mpm_particle_count', output_path)"
    )

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
    )

    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0
