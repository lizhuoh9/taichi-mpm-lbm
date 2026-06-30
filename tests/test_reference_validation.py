import json

import pytest

from fsi.postprocess import validation_summary_table
from fsi.reference import (
    ReferenceDataset,
    ReferenceMetricTolerance,
    compare_metrics,
    load_reference_dataset,
    reference_report_to_validation_report,
    save_reference_dataset,
)
from fsi.reference_cases import (
    default_reference_data_dir,
    generate_reference_datasets,
    run_reference_validation_suite,
)


def _sample_dataset() -> ReferenceDataset:
    return ReferenceDataset(
        schema_version=1,
        case_name="sample_reference",
        description="Sample reference dataset.",
        metrics={"x": 10.0, "zero": 0.0},
        tolerances={
            "x": ReferenceMetricTolerance(abs=0.1, rel=0.01),
            "zero": ReferenceMetricTolerance(abs=1.0e-6, rel=0.0),
        },
        metadata={"steps": 1},
        created_by="test",
    )


def test_reference_dataset_save_load_round_trip(tmp_path):
    path = save_reference_dataset(_sample_dataset(), tmp_path / "sample_reference.json")

    loaded = load_reference_dataset(path)

    assert loaded.case_name == "sample_reference"
    assert loaded.schema_version == 1
    assert loaded.metrics["x"] == pytest.approx(10.0)
    assert loaded.tolerances["x"].abs == pytest.approx(0.1)
    assert loaded.tolerances["x"].rel == pytest.approx(0.01)
    assert loaded.metadata == {"steps": 1}
    assert loaded.created_by == "test"


def test_reference_loader_rejects_invalid_schema_version(tmp_path):
    path = tmp_path / "bad_schema.json"
    data = _sample_dataset().to_dict()
    data["schema_version"] = 2
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_reference_dataset(path)


def test_reference_loader_rejects_missing_tolerance(tmp_path):
    path = tmp_path / "missing_tolerance.json"
    data = _sample_dataset().to_dict()
    del data["tolerances"]["x"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_reference_dataset(path)


def test_compare_metrics_passes_and_fails_by_tolerance():
    reference = _sample_dataset()

    passing = compare_metrics(
        "sample_reference",
        {"x": 10.05, "zero": 5.0e-7},
        reference,
    )
    failing = compare_metrics(
        "sample_reference",
        {"x": 11.0, "zero": 2.0e-6},
        reference,
    )

    assert passing.passed
    assert not failing.passed
    assert failing.comparisons[0].metric_name == "x"
    assert failing.comparisons[0].abs_error == pytest.approx(1.0)
    assert failing.comparisons[0].rel_error == pytest.approx(0.1)


def test_compare_metrics_requires_current_metric():
    with pytest.raises(KeyError):
        compare_metrics("sample_reference", {"x": 10.0}, _sample_dataset())


def test_reference_report_serialization_and_validation_bridge():
    report = compare_metrics(
        "sample_reference",
        {"x": 10.05, "zero": 5.0e-7},
        _sample_dataset(),
    )
    data = report.to_dict()
    validation_report = reference_report_to_validation_report(report)
    rows = validation_summary_table([validation_report.to_dict()])

    assert data["case_name"] == "sample_reference"
    assert data["passed"] is True
    assert len(data["comparisons"]) == 2
    assert validation_report.case_name == "reference_sample_reference"
    assert validation_report.passed
    assert rows[0]["case_name"] == "reference_sample_reference"
    assert rows[0]["metric_name"].endswith("_abs_error")


def test_validation_bridge_uses_effective_abs_tolerance_for_relative_pass():
    reference = ReferenceDataset(
        schema_version=1,
        case_name="relative_reference",
        description="Relative tolerance sample.",
        metrics={"x": 100.0},
        tolerances={"x": ReferenceMetricTolerance(abs=0.1, rel=0.01)},
    )
    report = compare_metrics("relative_reference", {"x": 100.5}, reference)
    validation_report = reference_report_to_validation_report(report)

    assert report.passed
    assert validation_report.metrics[0].value == pytest.approx(0.5)
    assert validation_report.metrics[0].upper == pytest.approx(1.0)
    assert validation_report.metrics[0].passed


def test_committed_reference_files_load_and_cover_tolerances():
    reference_dir = default_reference_data_dir()
    paths = sorted(reference_dir.glob("*.json"))
    names = {path.name for path in paths}

    assert len(paths) >= 5
    assert "immersed_boundary_contact_reference.json" in names
    for path in paths:
        dataset = load_reference_dataset(path)
        assert dataset.schema_version == 1
        assert dataset.metrics
        assert set(dataset.metrics) == set(dataset.tolerances)


def test_generate_reference_datasets_writes_loadable_json(tmp_path):
    paths = generate_reference_datasets(tmp_path)

    assert len(paths) >= 5
    for path in paths:
        dataset = load_reference_dataset(path)
        assert dataset.case_name
        assert dataset.metrics
        assert set(dataset.metrics) == set(dataset.tolerances)


def test_reference_validation_suite_passes_and_converts_to_validation_reports():
    reports = run_reference_validation_suite()
    validation_reports = [reference_report_to_validation_report(report) for report in reports]
    rows = validation_summary_table([report.to_dict() for report in validation_reports])

    assert len(reports) >= 5
    assert all(report.passed for report in reports)
    assert all(report.passed for report in validation_reports)
    assert rows
