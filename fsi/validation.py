from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class ValidationMetric:
    """One scalar validation metric with optional pass bounds."""

    name: str
    value: float
    lower: float | None = None
    upper: float | None = None
    passed: bool = True
    units: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "lower": self.lower,
            "upper": self.upper,
            "passed": self.passed,
            "units": self.units,
            "description": self.description,
        }


@dataclass(frozen=True)
class ValidationReport:
    """A named validation case and its scalar metrics."""

    case_name: str
    metrics: tuple[ValidationMetric, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(metric.passed for metric in self.metrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "passed": self.passed,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "metadata": self.metadata,
        }


def bounded_metric(
    name: str,
    value: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
    units: str = "",
    description: str = "",
) -> ValidationMetric:
    metric_value = float(value)
    passed = math.isfinite(metric_value)
    if lower is not None:
        passed = passed and metric_value >= float(lower)
    if upper is not None:
        passed = passed and metric_value <= float(upper)
    return ValidationMetric(
        name=name,
        value=metric_value,
        lower=lower,
        upper=upper,
        passed=bool(passed),
        units=units,
        description=description,
    )


def finite_metric(
    name: str,
    value: float,
    *,
    upper: float | None = None,
    units: str = "",
    description: str = "",
) -> ValidationMetric:
    return bounded_metric(
        name,
        value,
        upper=upper,
        units=units,
        description=description,
    )


def relative_error(value: float, reference: float) -> float:
    value_float = float(value)
    reference_float = float(reference)
    if reference_float == 0.0:
        return abs(value_float - reference_float)
    return abs((value_float - reference_float) / reference_float)
