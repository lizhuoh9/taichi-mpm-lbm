import math

import pytest

from fsi.validation import (
    ValidationMetric,
    ValidationReport,
    bounded_metric,
    finite_metric,
    relative_error,
)
from fsi.validation_cases import run_validation_suite


def _metric_by_name(report: ValidationReport, name: str) -> ValidationMetric:
    return next(metric for metric in report.metrics if metric.name == name)


@pytest.fixture(scope="module")
def validation_reports() -> dict[str, ValidationReport]:
    return {report.case_name: report for report in run_validation_suite()}


def test_bounded_metric_passes_and_fails():
    passing = bounded_metric("x", 0.5, lower=0.0, upper=1.0)
    too_high = bounded_metric("x", 2.0, lower=0.0, upper=1.0)
    not_finite = bounded_metric("x", math.nan, lower=0.0, upper=1.0)

    assert passing.passed
    assert not too_high.passed
    assert not not_finite.passed


def test_finite_metric_and_relative_error():
    assert finite_metric("finite", 1.0).passed
    assert not finite_metric("nan", math.nan).passed
    assert not finite_metric("too_high", 2.0, upper=1.0).passed

    assert relative_error(11.0, 10.0) == pytest.approx(0.1)
    assert relative_error(0.0, 0.0) == pytest.approx(0.0)


def test_validation_report_to_dict():
    report = ValidationReport(
        case_name="sample",
        metrics=(bounded_metric("x", 0.5, lower=0.0, upper=1.0),),
        metadata={"steps": 1},
    )

    data = report.to_dict()

    assert data["case_name"] == "sample"
    assert data["passed"] is True
    assert data["metadata"] == {"steps": 1}
    assert data["metrics"][0]["name"] == "x"
    assert data["metrics"][0]["passed"] is True


def test_lbm_mass_conservation_case_passes(validation_reports):
    report = validation_reports["lbm_periodic_mass_conservation"]

    assert report.passed
    assert _metric_by_name(report, "relative_mass_error").value <= 1.0e-5


def test_lbm_force_response_case_passes(validation_reports):
    report = validation_reports["lbm_force_response"]

    assert report.passed
    assert _metric_by_name(report, "mean_ux_growth").value > 0.0


def test_mpm_zero_gravity_com_case_passes(validation_reports):
    report = validation_reports["mpm_zero_gravity_com_stability"]

    assert report.passed
    assert _metric_by_name(report, "center_of_mass_drift_norm").value <= 1.0e-3


def test_mpm_gravity_response_case_passes(validation_reports):
    report = validation_reports["mpm_gravity_response"]

    assert report.passed
    assert _metric_by_name(report, "center_of_mass_y_delta").value < 0.0


def test_coupled_drift_case_passes(validation_reports):
    report = validation_reports["coupled_enabled_vs_disabled_drift"]

    assert report.passed
    assert _metric_by_name(report, "enabled_minus_disabled_dx").value > 1.0e-7


def test_coupling_force_balance_case_passes(validation_reports):
    report = validation_reports["coupling_force_balance"]

    assert report.passed
    assert _metric_by_name(report, "particle_force_x").value > 0.0
    assert _metric_by_name(report, "fluid_force_x").value < 0.0


def test_validation_suite_composes_expected_cases(validation_reports):
    reports = list(validation_reports.values())

    assert [report.case_name for report in reports] == [
        "lbm_periodic_mass_conservation",
        "lbm_force_response",
        "mpm_zero_gravity_com_stability",
        "mpm_gravity_response",
        "coupled_enabled_vs_disabled_drift",
        "coupling_force_balance",
        "coupling_force_limit",
        "coupling_boundary_support",
    ]
    assert all(report.passed for report in reports)
