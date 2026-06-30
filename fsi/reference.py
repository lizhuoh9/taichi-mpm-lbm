from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .validation import ValidationMetric, ValidationReport


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReferenceMetricTolerance:
    """Absolute and relative tolerance for one reference metric."""

    abs: float = 0.0
    rel: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"abs": float(self.abs), "rel": float(self.rel)}


@dataclass(frozen=True)
class ReferenceDataset:
    """A small committed reference dataset with metric-level tolerances."""

    schema_version: int
    case_name: str
    description: str
    metrics: dict[str, float]
    tolerances: dict[str, ReferenceMetricTolerance]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": int(self.schema_version),
            "case_name": self.case_name,
            "description": self.description,
            "metrics": {key: float(self.metrics[key]) for key in sorted(self.metrics)},
            "tolerances": {
                key: self.tolerances[key].to_dict() for key in sorted(self.tolerances)
            },
        }
        if self.created_by:
            data["created_by"] = self.created_by
        if self.metadata:
            data["metadata"] = self.metadata
        return data


@dataclass(frozen=True)
class ReferenceComparison:
    """Comparison of one current metric value against one reference value."""

    case_name: str
    metric_name: str
    value: float
    reference: float
    abs_error: float
    rel_error: float
    abs_tolerance: float
    rel_tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "metric_name": self.metric_name,
            "value": self.value,
            "reference": self.reference,
            "abs_error": self.abs_error,
            "rel_error": self.rel_error,
            "abs_tolerance": self.abs_tolerance,
            "rel_tolerance": self.rel_tolerance,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ReferenceReport:
    """Reference comparison report for one case."""

    case_name: str
    comparisons: tuple[ReferenceComparison, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(comparison.passed for comparison in self.comparisons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "passed": self.passed,
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "metadata": self.metadata,
        }


def load_reference_dataset(path: str | Path) -> ReferenceDataset:
    """Load a schema-versioned reference dataset from JSON."""

    reference_path = Path(path)
    raw_data = json.loads(reference_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError("reference dataset JSON must contain an object.")

    schema_version = int(_required(raw_data, "schema_version"))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported reference schema_version: {schema_version}.")

    case_name = str(_required(raw_data, "case_name"))
    description = str(_required(raw_data, "description"))
    metrics_raw = _required(raw_data, "metrics")
    tolerances_raw = _required(raw_data, "tolerances")
    if not isinstance(metrics_raw, dict) or not metrics_raw:
        raise ValueError("reference metrics must be a non-empty object.")
    if not isinstance(tolerances_raw, dict):
        raise ValueError("reference tolerances must be an object.")

    metrics = {str(key): _float_value(value, f"metrics.{key}") for key, value in metrics_raw.items()}
    tolerances: dict[str, ReferenceMetricTolerance] = {}
    for metric_name in metrics:
        if metric_name not in tolerances_raw:
            raise ValueError(f"Missing tolerance for reference metric: {metric_name}.")
        tolerances[metric_name] = _parse_tolerance(metric_name, tolerances_raw[metric_name])

    metadata = raw_data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("reference metadata must be an object when provided.")

    return ReferenceDataset(
        schema_version=schema_version,
        case_name=case_name,
        description=description,
        metrics=metrics,
        tolerances=tolerances,
        metadata=metadata,
        created_by=str(raw_data.get("created_by", "")),
    )


def save_reference_dataset(dataset: ReferenceDataset, path: str | Path) -> Path:
    """Save a reference dataset as deterministic UTF-8 JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def compare_metrics(
    case_name: str,
    current: dict[str, float],
    reference: ReferenceDataset,
) -> ReferenceReport:
    """Compare current metrics against a reference dataset."""

    comparisons: list[ReferenceComparison] = []
    for metric_name, reference_value in reference.metrics.items():
        if metric_name not in current:
            raise KeyError(f"Missing current metric for reference comparison: {metric_name}.")
        value = float(current[metric_name])
        ref = float(reference_value)
        abs_error = abs(value - ref)
        rel_error = abs_error if ref == 0.0 else abs_error / abs(ref)
        tolerance = reference.tolerances[metric_name]
        abs_tolerance = float(tolerance.abs)
        rel_tolerance = float(tolerance.rel)
        passed = abs_error <= abs_tolerance or rel_error <= rel_tolerance
        comparisons.append(
            ReferenceComparison(
                case_name=case_name,
                metric_name=metric_name,
                value=value,
                reference=ref,
                abs_error=abs_error,
                rel_error=rel_error,
                abs_tolerance=abs_tolerance,
                rel_tolerance=rel_tolerance,
                passed=bool(passed),
            )
        )

    return ReferenceReport(
        case_name=case_name,
        comparisons=tuple(comparisons),
        metadata={
            "reference_case_name": reference.case_name,
            "reference_description": reference.description,
            **reference.metadata,
        },
    )


def reference_report_to_validation_report(report: ReferenceReport) -> ValidationReport:
    """Convert a reference report into the existing validation report format."""

    return ValidationReport(
        case_name=f"reference_{report.case_name}",
        metrics=tuple(
            ValidationMetric(
                name=f"{comparison.metric_name}_abs_error",
                value=comparison.abs_error,
                upper=_effective_abs_tolerance(comparison),
                passed=comparison.passed,
                description=f"Reference comparison for {comparison.metric_name}.",
            )
            for comparison in report.comparisons
        ),
        metadata=report.metadata,
    )


def _effective_abs_tolerance(comparison: ReferenceComparison) -> float:
    if comparison.reference == 0.0:
        return max(comparison.abs_tolerance, comparison.rel_tolerance)
    return max(
        comparison.abs_tolerance,
        abs(comparison.reference) * comparison.rel_tolerance,
    )


def _required(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required reference field: {key}.")
    return data[key]


def _parse_tolerance(metric_name: str, value: Any) -> ReferenceMetricTolerance:
    if not isinstance(value, dict):
        raise ValueError(f"Tolerance for {metric_name} must be an object.")
    return ReferenceMetricTolerance(
        abs=_float_value(value.get("abs", 0.0), f"tolerances.{metric_name}.abs"),
        rel=_float_value(value.get("rel", 0.0), f"tolerances.{metric_name}.rel"),
    )


def _float_value(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
