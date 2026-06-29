# FSI LBM-MPM

A 3D two-way fluid-solid coupling simulator using:

- LBM for fluid
- MPM for deformable solids
- Taichi for high-performance kernels

## Current status

Step 8: benchmark-style validation cases.

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

Not implemented yet:

- improved coupling stability and boundary handling
- advanced visualization/rendering workflow
- high-fidelity validation against external reference data

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
```

## Roadmap

1. [x] Project skeleton and config system.
2. [x] Wrap the 3D D3Q19 MRT LBM solver.
3. [x] Add local LBM force field.
4. [x] Implement 3D MLS-MPM.
5. [x] Implement LBM-MPM two-way coupling.
6. [x] Add coupled examples and validation tests.

## Next milestones

- Improve coupling stability and boundary handling.
- Add richer visualization/post-processing tools.
- Add higher-fidelity reference-data validation cases.
