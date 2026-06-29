# Step 3 Local LBM Force Field Goal

## Objective

Replace the LBM solver's global external force `ext_f[None]` with a dense
per-cell force field `force[nx, ny, nz]`, while preserving the existing
`LBMConfig.force` uniform-force behavior.

Step 3 prepares the LBM side for future reaction-force scatter from MPM. It must
not implement MPM, real LBM-MPM coupling, particle-to-grid scatter, or sparse
LBM.

## In Scope

- Dense per-cell force field for the existing single-phase D3Q19/MRT LBM solver.
- Per-cell local Guo forcing in collision.
- Per-cell half-step velocity correction in macro update.
- Compatibility with `LBMConfig(force=(fx, fy, fz))` as the default uniform base
  force after `initialize()`.
- Explicit public force-management API:
  - `clear_force()`
  - `reset_force()`
  - `set_uniform_force(force)`
  - `add_uniform_force(force)`
  - `set_force_from_numpy(force_np)`
  - `force_numpy()`
  - `total_force()`
  - `zero_force_on_solid()`
- Local-force smoke tests and a lightweight local-force example.

## Out Of Scope

Do not implement any of the following in Step 3:

- MPM solver behavior.
- `LBMMpmCoupler` coupling behavior.
- Particle-to-grid force scatter from MPM particles.
- Sparse LBM.
- Moving boundaries.
- VTK or heavy output workflows.
- Any edits under `third_party/`.
- `ti.init()` inside `fsi/` library modules.
- `to_numpy()` or `from_numpy()` inside the LBM timestep path.

## Force Semantics

`LBMSolver3D.force[cell]` is the total local force density used by LBM kernels.

`LBMConfig.force` remains the uniform base force. During `initialize()`, fluid
cells should receive `config.force`, while solid cells should receive zero
force. This preserves Step 2 behavior for existing examples and tests.

`step()` must use the current force field as-is. It must not call
`clear_force()`, `reset_force()`, `set_uniform_force()`, or any other force
mutation automatically. Future simulation drivers and coupling kernels need to
own force-field lifetime explicitly.

Solid-cell force diagnostics should remain zero after force-management
operations, even though collision already skips solid cells.

`set_force_from_numpy()` is a convenience method for tests, debugging, and
external initialization. Performance-critical future coupling should update
`force[cell]` directly from Taichi kernels.

## Solver Changes

Update `fsi/lbm3d.py`:

- add `self.force = ti.Vector.field(3, ti.f32, shape=(nx, ny, nz))`;
- add `self.base_force = ti.Vector.field(3, ti.f32, shape=())`;
- remove numerical use of `ext_f[None]`;
- ideally remove `ext_f` entirely;
- initialize `base_force[None]` from `LBMConfig.force`;
- have `initialize()` call `reset_force()` after the solid mask and equilibrium
  fields are initialized;
- have `guo_force()` accept a local force vector or equivalent cell-local input;
- have `collide()` pass `self.force[i, j, k]` into Guo forcing;
- have `update_macro()` use `self.force[cell]` for the half-step velocity
  correction.

The timestep path remains:

```text
collide()
stream()
apply_boundary()
update_macro()
```

No force API should be called implicitly from that path.

## Force API Contract

Implement:

```python
def clear_force(self) -> None: ...
def reset_force(self) -> None: ...
def set_uniform_force(self, force: tuple[float, float, float]) -> None: ...
def add_uniform_force(self, force: tuple[float, float, float]) -> None: ...
def set_force_from_numpy(self, force_np: np.ndarray) -> None: ...
def force_numpy(self) -> np.ndarray: ...
def total_force(self) -> np.ndarray: ...
def zero_force_on_solid(self) -> None: ...
```

Validation helpers must reject:

- force vectors that do not have exactly three components;
- force NumPy arrays whose shape is not `(nx, ny, nz, 3)`.

`total_force()` should sum force only over fluid cells.

## Tests

Extend `tests/test_lbm3d.py` with coverage for:

- force field initialized from `LBMConfig.force`;
- `clear_force()` zeroing the field;
- `reset_force()` restoring the base force;
- `set_uniform_force()` and `add_uniform_force()`;
- `set_force_from_numpy()` accepting a local force array;
- invalid force shape raising `ValueError`;
- invalid force vector length raising `ValueError`;
- local force producing a stronger response in the forced region than an
  unforced region, or at minimum producing positive mean velocity in the forced
  direction if streaming diffusion makes regional comparison unstable;
- solid-cell force remaining zero after reset, set, add, and NumPy-set
  operations.

Existing Step 2 tests must continue to pass, especially:

- no-force mass conservation;
- `LBMConfig.force` still increasing mean X velocity;
- solid cells remain zero velocity;
- unsupported sparse storage and Y/Z non-periodic boundaries still fail;
- MPM, coupling, and simulation placeholders remain `NotImplementedError`.

## Example

Add `examples/lbm_local_force.py`:

- call `ti.init()` in the script;
- create a small fully periodic LBM grid;
- set `LBMConfig.force` to zero;
- use `set_force_from_numpy()` to apply X-force to a slab/block;
- run 100 steps;
- print forced-region mean `ux`, global mean `ux`, and `max|u|`;
- do not write output files.

Existing `examples/lbm_standalone.py` must continue to run without changes to
its public behavior.

## Documentation

Update `README.md` so it states:

- current status is Step 3 local per-cell LBM force field;
- implemented items include project skeleton, dense D3Q19/MRT LBM, and local
  per-cell LBM force field;
- not implemented items are MPM solver and LBM-MPM coupling;
- development commands include `python examples/lbm_local_force.py`;
- roadmap item 3 is complete.

## Acceptance Commands

The final state must pass:

```bash
python -m pip install -e ".[dev]"
pytest -q
python examples/smoke_import.py
python examples/lbm_standalone.py
python examples/lbm_local_force.py
python -m compileall fsi tests examples
ruff check .
git diff -- third_party
```

Additional checks:

```powershell
Select-String -Path fsi/*.py -Pattern 'ti.init'
Select-String -Path fsi/lbm3d.py -Pattern 'ext_f'
```

The `ti.init` scan must have no output. The `ext_f` scan should ideally have no
output. If an `ext_f` compatibility alias remains, `guo_force()` and
`update_macro()` must not read `ext_f[None]`.

## Completion And Push Contract

After implementation and verification:

1. Review the staged diff and confirm `third_party/` is untouched.
2. Commit with a conventional commit message.
3. Push the completed work to `origin/main` for `lizhuoh9/taichi-mpm-lbm`.
4. Verify the remote branch hash.
5. Report the final commit hash, pushed branch, and verification results.
