# Reference Data

This directory contains small JSON reference datasets for Step 11 reference-data
validation.

These files are committed baselines. They are intended for regression
protection against deterministic small cases, not for experimental or
paper-level high-fidelity validation claims.

Refresh the files intentionally with:

```bash
python examples/generate_reference_data.py
```

Review the generated JSON diff before committing refreshed references.
