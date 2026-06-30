from __future__ import annotations

import json
from pathlib import Path

import taichi as ti

from fsi.postprocess import validation_summary_table, write_validation_summary_csv
from fsi.reference import reference_report_to_validation_report
from fsi.reference_cases import run_reference_validation_suite


def main() -> None:
    ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False, offline_cache=False)

    reports = run_reference_validation_suite()
    validation_reports = [reference_report_to_validation_report(report) for report in reports]
    passed = all(report.passed for report in reports)

    for report in reports:
        status = "PASS" if report.passed else "FAIL"
        print(f"[{status}] {report.case_name}")
        for comparison in report.comparisons:
            print(
                "  {name}: value={value:.6e}, reference={reference:.6e}, "
                "abs_error={abs_error:.6e}".format(
                    name=comparison.metric_name,
                    value=comparison.value,
                    reference=comparison.reference,
                    abs_error=comparison.abs_error,
                )
            )

    output_dir = Path("outputs/reference_validation_suite")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "reference_validation_summary.json"
    json_path.write_text(
        json.dumps([report.to_dict() for report in validation_reports], indent=2),
        encoding="utf-8",
    )

    rows = validation_summary_table([report.to_dict() for report in validation_reports])
    csv_path = write_validation_summary_csv(rows, output_dir / "reference_validation_summary.csv")

    print(f"json: {json_path}")
    print(f"csv: {csv_path}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
