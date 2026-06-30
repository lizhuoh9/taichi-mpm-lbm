# FSI LBM-MPM

A 3D two-way fluid-solid coupling simulator using:

- LBM for fluid
- MPM for deformable solids
- Taichi for high-performance kernels

## Current status

Step 12: immersed-boundary/contact MVP.

Implemented:

- project skeleton and configuration system
- dense standalone 3D D3Q19/MRT LBM solver wrapper
- local per-cell LBM force field
- dense standalone 3D elastic MLS-MPM solver
- explicit penalty-based LBM-MPM two-way coupling
- top-level coupled simulation runner
- lightweight coupled example and validation tests
- NPZ/VTK simulation output snapshots
- benchmark-style validation cases
- coupling stability guards and boundary-support diagnostics
- visualization and post-processing utilities for snapshots and validation summaries
- reference-data validation framework and small committed reference cases
- immersed-boundary/contact MVP with dynamic occupancy diagnostics

Not implemented yet:

- high-fidelity external experimental/reference validation
- advanced rigid-body/contact mechanics
- interactive or production rendering workflow

## External references

This project will reuse or adapt code from:

- `yjhp1016/taichi_LBM3D`
- `yuanming-hu/taichi_mpm`

Third-party source code is stored under `third_party/`.

## Development setup

```bash
python -m pip install -e ".[dev]"
pytest
pytest -m "not slow" -q
python examples/smoke_import.py
python examples/lbm_standalone.py
python examples/lbm_local_force.py
python examples/mpm_standalone_cube.py
python examples/coupled_penalty_smoke.py
python examples/coupled_output_snapshot.py
python examples/validation_benchmark_suite.py
python examples/coupling_stability_boundary.py
python examples/postprocess_snapshots.py
python examples/postprocess_validation_summary.py
python examples/reference_validation_suite.py
python examples/immersed_boundary_contact_smoke.py
```

See [docs/postprocessing.md](docs/postprocessing.md) for NPZ, CSV/JSON, PNG, and VTK
post-processing notes.
See [docs/reference_validation.md](docs/reference_validation.md) for the committed JSON
reference-data validation workflow.
See [docs/immersed_boundary_contact.md](docs/immersed_boundary_contact.md) for the
Step 12 immersed-boundary/contact MVP.

## Roadmap

1. [x] Project skeleton and config system.
2. [x] Wrap the 3D D3Q19 MRT LBM solver.
3. [x] Add local LBM force field.
4. [x] Implement 3D MLS-MPM.
5. [x] Implement LBM-MPM two-way coupling.
6. [x] Add coupled examples and validation tests.

## Next milestones

- Add interactive or production rendering workflow.
- Add larger external reference-data campaigns.
- Explore advanced rigid-body/contact mechanics.
