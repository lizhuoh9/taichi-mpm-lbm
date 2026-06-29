# FSI LBM-MPM

A 3D two-way fluid-solid coupling simulator using:

- LBM for fluid
- MPM for deformable solids
- Taichi for high-performance kernels

## Current status

Step 6: coupled examples and validation tests.

Implemented:

- project skeleton and configuration system
- dense standalone 3D D3Q19/MRT LBM solver wrapper
- local per-cell LBM force field
- dense standalone 3D elastic MLS-MPM solver
- explicit penalty-based LBM-MPM two-way coupling
- top-level coupled simulation runner
- lightweight coupled example and validation tests

Not implemented yet:

- high-fidelity validation benchmarks
- visualization/output pipeline

## External references

This project will reuse or adapt code from:

- `yjhp1016/taichi_LBM3D`
- `yuanming-hu/taichi_mpm`

Third-party source code is stored under `third_party/`.

## Development setup

```bash
python -m pip install -e ".[dev]"
pytest
python examples/smoke_import.py
python examples/lbm_standalone.py
python examples/lbm_local_force.py
python examples/mpm_standalone_cube.py
python examples/coupled_penalty_smoke.py
```

## Roadmap

1. [x] Project skeleton and config system.
2. [x] Wrap the 3D D3Q19 MRT LBM solver.
3. [x] Add local LBM force field.
4. [x] Implement 3D MLS-MPM.
5. [x] Implement LBM-MPM two-way coupling.
6. [x] Add coupled examples and validation tests.

## Next milestones

- Add NPZ/VTK output snapshots.
- Add benchmark-style validation cases.
- Improve coupling stability and boundary handling.
