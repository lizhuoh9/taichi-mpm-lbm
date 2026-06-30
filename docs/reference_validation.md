# Reference-Data Validation

Step 11 adds reference-data validation for small deterministic cases. The goal
is to compare current metrics against committed JSON baselines so solver,
coupling, output, or post-processing changes can catch regressions early.

## What This Is

- Regression protection for small deterministic cases.
- A versioned JSON baseline format.
- Metric-level absolute and relative tolerances.
- A bridge into the existing `ValidationReport` and Step 10 CSV workflow.
- A local workflow suitable for code review.

## What This Is Not

- Experimental validation.
- Paper-level high-fidelity CFD/FSI certification.
- Agreement with Fluent, OpenFOAM, or physical lab measurements.
- A long benchmark campaign.
- ParaView or external solver automation.

## JSON Schema

Reference files live under `data/reference/` and use schema version 1:

```json
{
  "schema_version": 1,
  "case_name": "lbm_periodic_mass_reference",
  "description": "Small periodic LBM mass conservation reference.",
  "created_by": "fsi-lbm-mpm Step 11 reference generator",
  "metadata": {
    "steps": 20
  },
  "metrics": {
    "relative_mass_error": 1.1920928955078125e-7,
    "max_velocity_norm": 0.0
  },
  "tolerances": {
    "relative_mass_error": {
      "abs": 1e-5,
      "rel": 0.0
    },
    "max_velocity_norm": {
      "abs": 1e-4,
      "rel": 0.0
    }
  }
}
```

Every metric must have an explicit tolerance entry. Missing tolerances are
treated as schema errors rather than exact comparisons.

## Tolerance Policy

For each metric:

```text
abs_error = abs(current_value - reference_value)
rel_error = abs_error / abs(reference_value) if reference_value != 0
rel_error = abs_error if reference_value == 0
passed = abs_error <= abs_tolerance or rel_error <= rel_tolerance
```

Absolute tolerance is the main guard for near-zero values. Relative tolerance is
used for nonzero values that may vary slightly across platforms.

## Running Reference Validation

Run:

```bash
python examples/reference_validation_suite.py
```

The example writes:

```text
outputs/reference_validation_suite/reference_validation_summary.json
outputs/reference_validation_suite/reference_validation_summary.csv
```

The JSON file contains converted `ValidationReport` records. The CSV file is a
flattened metric table produced with the Step 10 post-processing helpers.

## Regenerating References

Refresh committed references intentionally with:

```bash
python examples/generate_reference_data.py
```

This overwrites `data/reference/*.json`. Review the resulting git diff before
committing. Reference updates should be treated as validation contract changes,
not as incidental generated output.

## Adding A Reference Case

1. Add a deterministic metric computation function in `fsi/reference_cases.py`.
2. Add a `ReferenceDataset` entry in `build_reference_datasets()`.
3. Give every metric an explicit tolerance.
4. Regenerate `data/reference/*.json`.
5. Add or update tests in `tests/test_reference_validation.py`.
6. Run `examples/reference_validation_suite.py`.
7. Update this document if the schema or workflow changes.

## Limitations

- Reference validation is diagnostic regression coverage.
- Current reference files are small JSON metric baselines.
- No binary reference datasets are committed.
- No external solver or experimental data is included in Step 11.
- Passing Step 11 does not prove high-fidelity physical validation.
