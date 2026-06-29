# Step 1 Project Skeleton Goal

## Objective

Build the first milestone of the `fsi-lbm-mpm` Python/Taichi project. This step
creates only the project skeleton, configuration system, dependency metadata,
license notices, basic tests, and a smoke import example for a future 3D two-way
LBM-MPM fluid-solid coupling simulator.

This step must not implement LBM, MPM, or coupling physics. The result should be
a stable engineering foundation that later steps can extend module by module.

## Repository Layout To Deliver

The repository root must contain:

```text
fsi/
  __init__.py
  config.py
  lbm3d.py
  mpm3d.py
  coupling.py
  simulation.py
  io.py
  units.py

examples/
  __init__.py
  README.md
  smoke_import.py

tests/
  __init__.py
  test_import.py
  test_config.py

third_party/
  README.md
  taichi_LBM3D/
  taichi_mpm/

scripts/
  README.md

docs/
  README.md
  goals/
    STEP1_PROJECT_SKELETON_GOAL.md

LICENSES.md
README.md
pyproject.toml
.gitignore
```

If Git submodules can be added for the third-party repositories, use submodules
for:

- `third_party/taichi_LBM3D`: https://github.com/yjhp1016/taichi_LBM3D
- `third_party/taichi_mpm`: https://github.com/yuanming-hu/taichi_mpm

If submodule setup is blocked by the environment, keep placeholder directories
and document the expected repositories in `third_party/README.md`.

## Configuration Contract

Create `fsi/config.py` with immutable dataclasses and explicit validation:

- `BoundaryConfig`
- `LBMConfig`
- `MPMConfig`
- `CouplingConfig`
- `OutputConfig`
- `SimulationConfig`

Each config class must expose `validate() -> None`.

`SimulationConfig` must:

- validate `num_steps > 0`;
- validate `lbm_dt > 0`;
- validate nested LBM, MPM, coupling, and output configs;
- reject mismatched LBM and MPM grid resolutions for the MVP;
- expose `mpm_dt` as `lbm_dt / coupling.mpm_substeps_per_lbm_step`.

The config layer must cover, at minimum:

- LBM grid dimensions, viscosity, reference density, body force, boundary
  configs, and sparse toggle;
- MPM grid dimensions, particle limits, particles per cell, `dx`, `dt`, density,
  Young's modulus, Poisson ratio, gravity, APIC toggle, and material selector;
- coupling enable toggle, penalty coefficient, MPM substeps per LBM step,
  coupling kernel selector, and future diagnostic field toggle;
- simulation backend, random seed, step count, and LBM time step;
- output directory, output interval, output format, and output field toggles.

## Placeholder Solver Contract

Create placeholder classes only:

- `LBMSolver3D` in `fsi/lbm3d.py`
- `MPMSolver3D` in `fsi/mpm3d.py`
- `LBMMpmCoupler` in `fsi/coupling.py`
- `FSISimulation` in `fsi/simulation.py`

Constructors should store the config and call its `validate()` method.

Placeholder runtime methods must raise `NotImplementedError` with clear step
messages. They must not contain kernels, numerical loops, or physics logic.

## Explicit Out Of Scope

Do not implement any of the following in Step 1:

- LBM collision;
- LBM streaming;
- MRT operators;
- MPM P2G;
- MPM G2P;
- MPM constitutive updates;
- LBM-MPM interpolation kernels;
- penalty coupling kernels;
- reaction force scattering;
- runtime-heavy examples;
- `to_numpy` or `from_numpy` workflows;
- `ti.init()` in library modules;
- changes inside third-party source trees.

The only accepted behavior for unimplemented solver actions is a clear
`NotImplementedError`.

## Package And Tooling Contract

Create `pyproject.toml` with:

- project name `fsi-lbm-mpm`;
- Python requirement `>=3.10`;
- dependencies `numpy`, `taichi`, and `pyevtk`;
- dev dependencies `pytest` and `ruff`;
- pytest configured for `tests`;
- project root added to pytest `pythonpath`;
- Ruff configured with line length 100 and target version `py310`.

Extend `.gitignore` so the repository ignores:

- Python caches and build artifacts;
- test, Ruff, MyPy, and virtual environment caches;
- Taichi caches;
- generated simulation outputs;
- VTK/NPZ/NPY output artifacts;
- common OS and IDE local files.

## Documentation Contract

Create:

- root `README.md` describing the project, current Step 1 status, external
  references, setup commands, and roadmap;
- `LICENSES.md` documenting future MIT-licensed reuse or adaptation from
  `yjhp1016/taichi_LBM3D` and `yuanming-hu/taichi_mpm`;
- `third_party/README.md` explaining expected external repositories and the
  rule that third-party code must not be edited directly;
- `examples/README.md`, `scripts/README.md`, and `docs/README.md` with concise
  purpose statements.

Documentation must make clear that no physics solver is implemented yet.

## Tests To Write

Add pytest coverage for:

- importing `fsi`;
- exported config classes are available from the package;
- default `SimulationConfig` validates;
- `mpm_dt` is computed correctly;
- invalid LBM viscosity fails validation;
- invalid MPM Poisson ratio fails validation;
- invalid coupling gamma fails validation;
- mismatched LBM and MPM grids fail validation;
- placeholder solver methods raise `NotImplementedError`;
- library modules do not call `ti.init()`.

## Smoke Example

Create `examples/smoke_import.py` that:

- creates `SimulationConfig`;
- validates it;
- prints a ready message;
- prints grid dimensions;
- prints `lbm_dt`;
- prints computed `mpm_dt`.

It must remain import/config-only and must not initialize Taichi or run physics.

## Acceptance Commands

The final state must pass:

```bash
python -m pip install -e ".[dev]"
pytest
python examples/smoke_import.py
python -m compileall fsi tests examples
```

If the environment already has compatible dependencies, the installed editable
package should be used for validation. If installation or dependency download is
blocked by the sandbox or network, record the exact blocker and still run all
locally possible checks.

## Completion And Push Contract

When implementation and verification are complete:

1. Review the diff for unintended third-party edits or accidental physics
   implementation.
2. Commit with a conventional commit message.
3. Push to `origin/main` for `lizhuoh9/taichi-mpm-lbm`.
4. Report the final commit hash, pushed branch, and verification results.
