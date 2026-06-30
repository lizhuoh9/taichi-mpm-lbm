from __future__ import annotations

from pathlib import Path

from .reference import (
    ReferenceDataset,
    ReferenceMetricTolerance,
    ReferenceReport,
    compare_metrics,
    load_reference_dataset,
    save_reference_dataset,
)
from .validation import ValidationReport
from .validation_cases import (
    run_coupled_drift_case,
    run_coupling_boundary_support_case,
    run_coupling_force_limit_case,
    run_lbm_force_response_case,
    run_lbm_mass_conservation_case,
    run_mpm_gravity_response_case,
)


CREATED_BY = "fsi-lbm-mpm Step 11 reference generator"


def default_reference_data_dir() -> Path:
    """Return the repository's committed reference data directory."""

    return Path(__file__).resolve().parents[1] / "data" / "reference"


def compute_lbm_periodic_mass_reference_metrics() -> dict[str, float]:
    report = run_lbm_mass_conservation_case()
    return _selected_metrics(report, ("relative_mass_error", "max_velocity_norm"))


def compute_lbm_force_response_reference_metrics() -> dict[str, float]:
    report = run_lbm_force_response_case()
    return _selected_metrics(report, ("mean_ux_growth", "max_velocity_norm"))


def compute_mpm_gravity_reference_metrics() -> dict[str, float]:
    report = run_mpm_gravity_response_case()
    return _selected_metrics(
        report,
        ("center_of_mass_y_delta", "positions_finite", "velocities_finite"),
    )


def compute_coupled_drift_reference_metrics() -> dict[str, float]:
    report = run_coupled_drift_case()
    return _selected_metrics(
        report,
        ("enabled_dx", "disabled_dx_abs", "enabled_minus_disabled_dx"),
    )


def compute_coupling_stability_reference_metrics() -> dict[str, float]:
    force_limit = run_coupling_force_limit_case()
    boundary_support = run_coupling_boundary_support_case()
    force_metrics = _selected_metrics(force_limit, ("particle_force_norm", "force_balance_norm"))
    boundary_metrics = _selected_metrics(
        boundary_support,
        ("partial_support_particle_count", "min_particle_valid_weight"),
    )
    return {**force_metrics, **boundary_metrics}


def build_reference_datasets() -> list[ReferenceDataset]:
    """Compute all Step 11 reference datasets from the current implementation."""

    return [
        ReferenceDataset(
            schema_version=1,
            case_name="lbm_periodic_mass_reference",
            description="Small periodic LBM mass conservation reference.",
            metrics=compute_lbm_periodic_mass_reference_metrics(),
            tolerances={
                "relative_mass_error": ReferenceMetricTolerance(abs=1.0e-5, rel=0.0),
                "max_velocity_norm": ReferenceMetricTolerance(abs=1.0e-4, rel=0.0),
            },
            metadata={"source_case": "lbm_periodic_mass_conservation", "steps": 20},
            created_by=CREATED_BY,
        ),
        ReferenceDataset(
            schema_version=1,
            case_name="lbm_force_response_reference",
            description="Small forced periodic LBM velocity-growth reference.",
            metrics=compute_lbm_force_response_reference_metrics(),
            tolerances={
                "mean_ux_growth": ReferenceMetricTolerance(abs=1.0e-8, rel=1.0e-2),
                "max_velocity_norm": ReferenceMetricTolerance(abs=1.0e-8, rel=1.0e-2),
            },
            metadata={"source_case": "lbm_force_response", "steps": 20},
            created_by=CREATED_BY,
        ),
        ReferenceDataset(
            schema_version=1,
            case_name="mpm_gravity_reference",
            description="Weak-gravity MPM response reference.",
            metrics=compute_mpm_gravity_reference_metrics(),
            tolerances={
                "center_of_mass_y_delta": ReferenceMetricTolerance(abs=1.0e-6, rel=1.0e-2),
                "positions_finite": ReferenceMetricTolerance(abs=0.0, rel=0.0),
                "velocities_finite": ReferenceMetricTolerance(abs=0.0, rel=0.0),
            },
            metadata={"source_case": "mpm_gravity_response", "steps": 10},
            created_by=CREATED_BY,
        ),
        ReferenceDataset(
            schema_version=1,
            case_name="coupled_drift_reference",
            description="Enabled-vs-disabled coupling drift reference.",
            metrics=compute_coupled_drift_reference_metrics(),
            tolerances={
                "enabled_dx": ReferenceMetricTolerance(abs=1.0e-6, rel=5.0e-2),
                "disabled_dx_abs": ReferenceMetricTolerance(abs=1.0e-6, rel=0.0),
                "enabled_minus_disabled_dx": ReferenceMetricTolerance(abs=1.0e-6, rel=5.0e-2),
            },
            metadata={"source_case": "coupled_enabled_vs_disabled_drift", "steps": 2},
            created_by=CREATED_BY,
        ),
        ReferenceDataset(
            schema_version=1,
            case_name="coupling_stability_reference",
            description="Step 9 force-limit and boundary-support reference.",
            metrics=compute_coupling_stability_reference_metrics(),
            tolerances={
                "particle_force_norm": ReferenceMetricTolerance(abs=1.0e-7, rel=1.0e-3),
                "force_balance_norm": ReferenceMetricTolerance(abs=1.0e-6, rel=0.0),
                "partial_support_particle_count": ReferenceMetricTolerance(abs=0.0, rel=0.0),
                "min_particle_valid_weight": ReferenceMetricTolerance(abs=1.0e-6, rel=1.0e-3),
            },
            metadata={
                "source_cases": [
                    "coupling_force_limit",
                    "coupling_boundary_support",
                ]
            },
            created_by=CREATED_BY,
        ),
    ]


def generate_reference_datasets(output_dir: str | Path) -> list[Path]:
    """Write freshly computed reference datasets to an explicit output directory."""

    directory = Path(output_dir)
    paths: list[Path] = []
    for dataset in build_reference_datasets():
        paths.append(save_reference_dataset(dataset, directory / _reference_filename(dataset)))
    return paths


def run_reference_validation_suite(
    reference_dir: str | Path | None = None,
) -> list[ReferenceReport]:
    """Compare current small-case metrics against committed reference datasets."""

    directory = default_reference_data_dir() if reference_dir is None else Path(reference_dir)
    current_metrics = _current_metrics_by_case()
    reports: list[ReferenceReport] = []
    for dataset in _load_reference_datasets(directory):
        reports.append(
            compare_metrics(
                dataset.case_name,
                current_metrics[dataset.case_name],
                dataset,
            )
        )
    return reports


def _load_reference_datasets(directory: Path) -> list[ReferenceDataset]:
    return [load_reference_dataset(path) for path in sorted(directory.glob("*.json"))]


def _current_metrics_by_case() -> dict[str, dict[str, float]]:
    return {
        "lbm_periodic_mass_reference": compute_lbm_periodic_mass_reference_metrics(),
        "lbm_force_response_reference": compute_lbm_force_response_reference_metrics(),
        "mpm_gravity_reference": compute_mpm_gravity_reference_metrics(),
        "coupled_drift_reference": compute_coupled_drift_reference_metrics(),
        "coupling_stability_reference": compute_coupling_stability_reference_metrics(),
    }


def _selected_metrics(report: ValidationReport, names: tuple[str, ...]) -> dict[str, float]:
    by_name = {metric.name: float(metric.value) for metric in report.metrics}
    return {name: by_name[name] for name in names}


def _reference_filename(dataset: ReferenceDataset) -> str:
    return f"{dataset.case_name}.json"
