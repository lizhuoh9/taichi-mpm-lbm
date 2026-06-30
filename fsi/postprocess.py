from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SnapshotInfo:
    """Metadata for one NPZ simulation snapshot."""

    path: Path
    step: int
    time: float
    particle_count: int
    lbm_shape: tuple[int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "step": self.step,
            "time": self.time,
            "particle_count": self.particle_count,
            "lbm_shape": self.lbm_shape,
        }


def list_npz_snapshots(output_dir: str | Path) -> list[Path]:
    """Return snapshot_*.npz files sorted by numeric step."""

    directory = Path(output_dir)
    if not directory.exists():
        return []
    return sorted(directory.glob("snapshot_*.npz"), key=_snapshot_sort_key)


def load_npz_snapshot(path: str | Path) -> dict[str, np.ndarray]:
    """Load an NPZ snapshot into a normal dictionary of arrays."""

    snapshot_path = Path(path)
    with np.load(snapshot_path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def snapshot_info(path: str | Path) -> SnapshotInfo:
    """Read metadata from one NPZ snapshot."""

    snapshot_path = Path(path)
    data = load_npz_snapshot(snapshot_path)
    lbm_shape = tuple(int(value) for value in np.asarray(data["lbm_shape"]).reshape(-1))
    if len(lbm_shape) != 3:
        raise ValueError(f"lbm_shape must have 3 entries, got {lbm_shape}.")
    return SnapshotInfo(
        path=snapshot_path,
        step=_int_scalar(data["step"]),
        time=_float_scalar(data["time"]),
        particle_count=_int_scalar(data["mpm_particle_count"]),
        lbm_shape=lbm_shape,
    )


def summarize_snapshots(output_dir: str | Path) -> list[SnapshotInfo]:
    """Return metadata for all NPZ snapshots in an output directory."""

    return [snapshot_info(path) for path in list_npz_snapshots(output_dir)]


def extract_snapshot_timeseries(output_dir: str | Path) -> dict[str, np.ndarray]:
    """Extract metadata and available derived diagnostics from NPZ snapshots."""

    rows = [_snapshot_timeseries_row(path) for path in list_npz_snapshots(output_dir)]
    if not rows:
        return {
            "step": np.array([], dtype=np.int64),
            "time": np.array([], dtype=np.float64),
            "mpm_particle_count": np.array([], dtype=np.int64),
        }

    common_keys = set(rows[0])
    for row in rows[1:]:
        common_keys.intersection_update(row)

    ordered_keys = _ordered_keys(common_keys)
    return {key: np.array([row[key] for row in rows]) for key in ordered_keys}


def summarize_coupling_diagnostics(output_dir: str | Path) -> dict[str, np.ndarray]:
    """Extract Step 9 coupling diagnostic time series when present."""

    timeseries = extract_snapshot_timeseries(output_dir)
    keys = ["step", "time"]
    keys.extend(
        key
        for key in (
            "coupling_force_norm",
            "total_particle_coupling_force_norm",
            "total_fluid_coupling_force_norm",
            "coupling_unsupported_particle_count",
            "coupling_partial_support_particle_count",
            "coupling_clipped_particle_count",
        )
        if key in timeseries
    )
    return {key: timeseries[key] for key in keys if key in timeseries}


def write_timeseries_csv(timeseries: dict[str, np.ndarray], path: str | Path) -> Path:
    """Write a compact CSV time-series table without pandas."""

    keys = _validate_timeseries(timeseries)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    length = len(np.asarray(timeseries[keys[0]]))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for index in range(length):
            writer.writerow({key: _jsonify(np.asarray(timeseries[key])[index]) for key in keys})
    return output_path


def write_timeseries_json(timeseries: dict[str, np.ndarray], path: str | Path) -> Path:
    """Write a NumPy-safe JSON time-series file."""

    _validate_timeseries(timeseries)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonify(timeseries), indent=2), encoding="utf-8")
    return output_path


def load_validation_summary(path: str | Path) -> list[dict[str, Any]]:
    """Load validation summary JSON written by the validation benchmark example."""

    summary_path = Path(path)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("validation summary JSON must contain a list of reports.")
    return data


def validation_summary_table(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten validation reports into one row per metric."""

    rows: list[dict[str, Any]] = []
    for report in reports:
        case_name = report.get("case_name", "")
        case_passed = bool(report.get("passed", False))
        for metric in report.get("metrics", []):
            rows.append(
                {
                    "case_name": case_name,
                    "case_passed": case_passed,
                    "metric_name": metric.get("name", ""),
                    "value": metric.get("value"),
                    "lower": metric.get("lower"),
                    "upper": metric.get("upper"),
                    "metric_passed": bool(metric.get("passed", False)),
                    "units": metric.get("units", ""),
                    "description": metric.get("description", ""),
                }
            )
    return rows


def write_validation_summary_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write flattened validation metric rows to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_name",
        "case_passed",
        "metric_name",
        "value",
        "lower",
        "upper",
        "metric_passed",
        "units",
        "description",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonify(row.get(key)) for key in fieldnames})
    return output_path


def plot_timeseries(
    timeseries: dict[str, np.ndarray],
    y_key: str,
    output_path: str | Path,
    *,
    x_key: str = "step",
) -> Path:
    """Write a simple PNG line plot for one time-series key."""

    if x_key not in timeseries:
        raise KeyError(f"Missing x_key in time series: {x_key}.")
    if y_key not in timeseries:
        raise KeyError(f"Missing y_key in time series: {y_key}.")

    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(np.asarray(timeseries[x_key]), np.asarray(timeseries[y_key]), marker="o")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _snapshot_sort_key(path: Path) -> tuple[int, int | str]:
    stem = path.stem
    prefix = "snapshot_"
    if stem.startswith(prefix):
        suffix = stem[len(prefix) :]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, path.name)


def _snapshot_timeseries_row(path: Path) -> dict[str, float | int]:
    data = load_npz_snapshot(path)
    row: dict[str, float | int] = {
        "step": _int_scalar(data["step"]),
        "time": _float_scalar(data["time"]),
        "mpm_particle_count": _int_scalar(data["mpm_particle_count"]),
    }

    if "lbm_density" in data:
        density = np.asarray(data["lbm_density"])
        row["lbm_mean_density"] = float(np.mean(density))
        row["lbm_total_mass_estimate"] = float(np.sum(density))
    if "lbm_velocity" in data:
        row["lbm_max_velocity_norm"] = _max_vector_norm(data["lbm_velocity"])
    if "mpm_positions" in data and np.asarray(data["mpm_positions"]).size > 0:
        center = np.mean(np.asarray(data["mpm_positions"], dtype=np.float64), axis=0)
        row["mpm_center_of_mass_x"] = float(center[0])
        row["mpm_center_of_mass_y"] = float(center[1])
        row["mpm_center_of_mass_z"] = float(center[2])
    if "mpm_velocities" in data:
        row["mpm_max_velocity_norm"] = _max_vector_norm(data["mpm_velocities"])
    if "coupling_force" in data:
        row["coupling_force_norm"] = float(np.linalg.norm(np.asarray(data["coupling_force"])))
    if "total_particle_coupling_force" in data:
        row["total_particle_coupling_force_norm"] = float(
            np.linalg.norm(np.asarray(data["total_particle_coupling_force"]))
        )
    if "total_fluid_coupling_force" in data:
        row["total_fluid_coupling_force_norm"] = float(
            np.linalg.norm(np.asarray(data["total_fluid_coupling_force"]))
        )
    for key in (
        "coupling_unsupported_particle_count",
        "coupling_partial_support_particle_count",
        "coupling_clipped_particle_count",
    ):
        if key in data:
            row[key] = _int_scalar(data[key])
    return row


def _max_vector_norm(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return 0.0
    vectors = array.reshape(-1, array.shape[-1])
    return float(np.linalg.norm(vectors, axis=1).max())


def _ordered_keys(keys: set[str]) -> list[str]:
    preferred = [
        "step",
        "time",
        "mpm_particle_count",
        "lbm_mean_density",
        "lbm_total_mass_estimate",
        "lbm_max_velocity_norm",
        "mpm_center_of_mass_x",
        "mpm_center_of_mass_y",
        "mpm_center_of_mass_z",
        "mpm_max_velocity_norm",
        "coupling_force_norm",
        "total_particle_coupling_force_norm",
        "total_fluid_coupling_force_norm",
        "coupling_unsupported_particle_count",
        "coupling_partial_support_particle_count",
        "coupling_clipped_particle_count",
    ]
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(keys.difference(ordered)))
    return ordered


def _validate_timeseries(timeseries: dict[str, np.ndarray]) -> list[str]:
    if not timeseries:
        raise ValueError("time series must not be empty.")
    keys = _ordered_keys(set(timeseries))
    lengths = {len(np.asarray(timeseries[key])) for key in keys}
    if len(lengths) != 1:
        raise ValueError("all time series arrays must have the same length.")
    return keys


def _int_scalar(value: np.ndarray) -> int:
    return int(np.asarray(value).reshape(()).item())


def _float_scalar(value: np.ndarray) -> float:
    return float(np.asarray(value).reshape(()).item())


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
