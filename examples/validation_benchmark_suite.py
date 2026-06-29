from __future__ import annotations

import json
from pathlib import Path

import taichi as ti

from fsi.validation_cases import run_validation_suite


def main() -> None:
    ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False, offline_cache=False)

    reports = run_validation_suite()
    passed = all(report.passed for report in reports)

    for report in reports:
        status = "PASS" if report.passed else "FAIL"
        print(f"[{status}] {report.case_name}")
        for metric in report.metrics:
            print(f"  {metric.name}: {metric.value:.6e}")

    output_dir = Path("outputs/validation_benchmark_suite")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validation_summary.json"
    path.write_text(
        json.dumps([report.to_dict() for report in reports], indent=2),
        encoding="utf-8",
    )
    print(f"wrote {path}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
