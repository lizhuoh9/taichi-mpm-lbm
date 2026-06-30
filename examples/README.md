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

Step 10 post-processing examples:

- `postprocess_snapshots.py`: generate a tiny coupled run, read NPZ snapshots,
  export CSV/JSON time series, and write a PNG plot.
- `postprocess_validation_summary.py`: run the validation suite and flatten
  validation metrics to CSV.
