from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> None:
    validation_example = Path(__file__).with_name("validation_benchmark_suite.py")
    subprocess.run([sys.executable, str(validation_example)], check=True)

    from fsi.postprocess import (
        load_validation_summary,
        validation_summary_table,
        write_validation_summary_csv,
    )

    output_dir = Path("outputs/validation_benchmark_suite")
    summary_path = output_dir / "validation_summary.json"
    loaded_reports = load_validation_summary(summary_path)
    rows = validation_summary_table(loaded_reports)
    csv_path = write_validation_summary_csv(rows, output_dir / "validation_summary.csv")

    failed_rows = [row for row in rows if not row["metric_passed"]]
    for row in failed_rows:
        print(
            "FAIL {case_name}.{metric_name}: {value}".format(
                case_name=row["case_name"],
                metric_name=row["metric_name"],
                value=row["value"],
            )
        )

    print(f"reports: {len(loaded_reports)}")
    print(f"metrics: {len(rows)}")
    print(f"json: {summary_path}")
    print(f"csv: {csv_path}")

    if failed_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
