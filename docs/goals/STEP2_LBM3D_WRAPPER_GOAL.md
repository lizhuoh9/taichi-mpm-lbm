# Step 2 LBM3D Wrapper Goal

## Objective

Wrap the single-phase dense 3D D3Q19/MRT LBM solver from
`third_party/taichi_LBM3D/Single_phase/lbm_solver_3d.py` into
`fsi.lbm3d.LBMSolver3D`.

Step 2 must produce a runnable standalone LBM solver while keeping the project
scope narrow. The implementation should adapt the dense single-phase LBM
algorithm into a class-oriented API, preserve the original solver's D3Q19
ordering and MRT data, and keep the rest of the project as placeholders.

## In Scope

- Dense single-phase 3D LBM only.
- D3Q19 lattice directions, weights, and opposite-direction table.
- MRT moment matrix, inverse matrix, and relaxation diagonal setup.
- Guo forcing using one global force vector `ext_f[None]`.
- Periodic streaming with solid-cell bounce-back.
- X-left and X-right pressure or velocity boundary handling based on the
  upstream solver.
- Periodic Y and Z boundaries only.
- Solid mask initialization from NumPy arrays and text files.
- Lightweight NumPy query helpers for tests, diagnostics, and examples.
- A standalone LBM example that runs a small periodic simulation.
- Smoke tests for initialization, mass conservation, global forcing, solid mask
  behavior, and scope boundaries.

## Out Of Scope

Do not implement any of the following in Step 2:

- MPM solver behavior.
- LBM-MPM coupling behavior.
- Per-cell or local LBM force fields.
- Sparse LBM storage.
- Multiphase LBM.
- Moving boundaries.
- General six-face non-periodic boundary conditions.
- Runtime-heavy examples or large grid outputs.
- `ti.init()` inside `fsi/` library modules.
- `to_numpy()` or `from_numpy()` calls inside the timestep path.
- Any edit under `third_party/`.

## Configuration Contract

Extend `LBMConfig` only as needed for standalone LBM:

- add `initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)`;
- validate that `initial_velocity` has exactly three components;
- keep `force` as a global 3-vector;
- keep `use_sparse`, but `LBMSolver3D` must raise `NotImplementedError` when it
  is true.

`LBMSolver3D` must validate the supported Step 2 boundary subset:

- X-left and X-right may be `periodic`, `pressure`, or `velocity`;
- X-left and X-right `wall` are not implemented in Step 2;
- Y and Z faces must remain `periodic`;
- solid obstacles and walls should be represented with the solid mask for now.

## Solver API Contract

`fsi/lbm3d.py` must expose `@ti.data_oriented class LBMSolver3D` with at least:

- `__init__(config: LBMConfig)`;
- `initialize(solid_np: np.ndarray | None = None) -> None`;
- `static_init()`;
- `init_fields()`;
- `feq(...)`;
- `multiply_m(...)`;
- `guo_force(...)`;
- `meq_vec(...)`;
- `collide()`;
- `periodic_index(...)`;
- `stream()`;
- `apply_boundary()`;
- `update_macro()`;
- `step()`;
- `density_numpy()`;
- `velocity_numpy()`;
- `solid_numpy()`;
- `distribution_numpy()`;
- `total_mass()`;
- `max_velocity_norm()`;
- `load_solid_from_txt(...)`.

The runtime timestep path is:

```text
collide()
stream()
apply_boundary()
update_macro()
```

`initialize()` is allowed to use `from_numpy()` to load the initial solid mask
and static data. Query helpers are allowed to use `to_numpy()`. The timestep
path must stay Taichi-resident.

## Field Contract

The solver should keep names close to the upstream dense solver:

- `f`, `F`, `rho`, `v`;
- `e`, `e_f`, `w`;
- `solid`;
- `LR`, `S_dig`;
- `ext_f`;
- `M`, `inv_M`;
- boundary velocity 0D vector fields for the supported boundary API.

Static NumPy arrays for `M`, `inv_M`, `LR`, and `S_dig` should be copied into
Taichi fields during construction. The D3Q19 direction ordering must remain
consistent with the upstream opposite-direction table.

## Tests

Update the existing placeholder tests:

- remove the `LBMSolver3D.initialize()` and `LBMSolver3D.step()` expected
  `NotImplementedError` assertions;
- keep MPM, coupling, and simulation placeholder assertions.

Add LBM tests for:

- small-grid initialization without NaN or shape errors;
- mass conservation for a zero-force, fully periodic grid;
- global force increasing mean X velocity;
- solid-mask cells staying at zero velocity;
- invalid solid-mask shape raising `ValueError`;
- unsupported sparse LBM raising `NotImplementedError`;
- unsupported non-periodic Y/Z boundaries raising `NotImplementedError`;
- no `ti.init()` calls in `fsi/`.

Tests must use small grids such as `8x8x8` and CPU Taichi initialization from
test code, not from library modules.

## Example

Add `examples/lbm_standalone.py`:

- call `ti.init()` in the script;
- use a small fully periodic grid, e.g. `32x16x8`;
- apply a small global X force;
- run 100 steps;
- print mass and max velocity every 20 steps;
- avoid writing heavy files.

## Documentation

Update `README.md` so it states:

- Step 1 is complete;
- Step 2 dense standalone 3D D3Q19/MRT LBM wrapper is implemented;
- local per-cell force fields, MPM, and LBM-MPM coupling are still not
  implemented;
- `examples/lbm_standalone.py` is the standalone LBM smoke example.

## Acceptance Commands

The final state must pass:

```bash
python -m pip install -e ".[dev]"
pytest
python examples/smoke_import.py
python examples/lbm_standalone.py
python -m compileall fsi tests examples
ruff check .
```

Additional checks:

```bash
git diff -- third_party
grep -R "ti.init(" fsi
```

`git diff -- third_party` must show no edits to third-party source. The
`ti.init()` scan must have no matches under `fsi/`.

## Completion And Push Contract

After implementation and verification:

1. Review the staged diff and confirm no third-party source edits.
2. Commit with a conventional commit message.
3. Push the completed work to `origin/main` for `lizhuoh9/taichi-mpm-lbm`.
4. Report the final commit hash, pushed branch, and verification results.
