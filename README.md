# FSI LBM-MPM

A 3D two-way fluid-solid coupling simulator using:

- LBM for fluid
- MPM for deformable solids
- Taichi for high-performance kernels

## Current status

Step 1: project skeleton and configuration system.

No physics solver is implemented yet.

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
```

## Roadmap

1. Project skeleton and config system.
2. Wrap the 3D D3Q19 MRT LBM solver.
3. Add local LBM force field.
4. Implement 3D MLS-MPM.
5. Implement LBM-MPM two-way coupling.
6. Add coupled examples and validation tests.
