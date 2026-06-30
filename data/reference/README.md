# Reference Data

This directory contains small JSON reference datasets for the committed
reference-data validation workflow.

These files are committed baselines. They are intended for regression
protection against deterministic small cases, not for experimental or
paper-level high-fidelity validation claims.

Step 12 adds `immersed_boundary_contact_reference.json` for the opt-in
immersed-boundary/contact MVP diagnostics.

Refresh the files intentionally with:

```bash
python examples/generate_reference_data.py
```

Review the generated JSON diff before committing refreshed references.
