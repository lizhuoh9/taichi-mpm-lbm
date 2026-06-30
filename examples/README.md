# Examples

This directory contains lightweight examples for the FSI LBM-MPM project.

Current examples cover:

- package import smoke checks
- standalone LBM and local-force runs
- standalone MPM cube runs
- coupled penalty-smoke runs
- NPZ output snapshots
- benchmark-style validation summaries
- coupling stability and boundary-support diagnostics
- post-processing snapshots into CSV/JSON/PNG summaries
- post-processing validation reports
- reference-data validation against committed JSON baselines
- immersed-boundary/contact MVP smoke diagnostics

Step 10 post-processing examples:

- `postprocess_snapshots.py`: generate a tiny coupled run, read NPZ snapshots,
  export CSV/JSON time series, and write a PNG plot.
- `postprocess_validation_summary.py`: run the validation suite and flatten
  validation metrics to CSV.

Step 11 reference-validation examples:

- `reference_validation_suite.py`: compare current metrics against committed
  JSON reference baselines and export JSON/CSV summaries.
- `generate_reference_data.py`: intentionally refresh committed reference JSON
  files after reviewing the resulting diff.

Step 12 immersed-boundary/contact example:

- `immersed_boundary_contact_smoke.py`: run a tiny opt-in IB/contact MVP case
  and print scalar fluid, particle, IB, and contact diagnostics.
