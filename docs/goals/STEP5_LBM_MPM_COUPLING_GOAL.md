# Step 5 LBM-MPM Coupling Goal

## Objective

Implement an explicit partitioned two-way coupling MVP between
`fsi.lbm3d.LBMSolver3D` and `fsi.mpm3d.MPMSolver3D` through
`fsi.coupling.LBMMpmCoupler`.

Step 5 must connect the standalone LBM and standalone MPM solvers with a small,
testable, momentum-balanced penalty coupling:

```text
LBM velocity field -> interpolate to MPM particles -> particle coupling force
MPM particle force -> equal opposite reaction scatter -> LBM force field
```

The result should be a real coupling engine, not a full production simulation
runner. `FSISimulation.run()` remains reserved for a later step.

## In Scope

- Replace the `LBMMpmCoupler` placeholder with a Taichi data-oriented coupler.
- Accept a `CouplingConfig`, `LBMSolver3D`, and `MPMSolver3D` in the coupler.
- Validate matching LBM and MPM grid dimensions.
- Validate the configured quadratic kernel and positive step sizes.
- Interpolate LBM grid velocity to MPM particles with a quadratic 3x3x3 kernel.
- Skip out-of-domain and LBM solid cells during interpolation and reaction scatter.
- Normalize valid kernel weights near boundaries and solid cells.
- Compute particle penalty force:

```text
F_p = gamma * m_p * (u_fluid_at_particle - v_particle)
```

- Store particle force in `mpm.particle_force`.
- Scatter equal-and-opposite reaction force density to an LBM-side
  `coupling_force` field.
- Keep coupling reaction force zero on LBM solid cells.
- Optionally build a solid volume fraction field from MPM particle volumes.
- Implement one coupled step as:

```text
clear coupling fields
for each MPM substep:
    clear MPM particle force
    compute coupling forces from current LBM velocity
    accumulate time-weighted reaction force for LBM
    advance MPM by mpm_dt
reset LBM base force
add accumulated coupling force to LBM force field
advance LBM once
```

- Make `CouplingConfig.enabled=False` perform uncoupled stepping:
  LBM and MPM still advance, but coupling forces remain zero.
- Add diagnostics for force balance and exported coupling fields.
- Add focused tests for construction validation, force direction, force balance,
  solid-cell masking, finite stepping, and disabled coupling behavior.
- Update README to mark Step 5 complete and Step 6 still pending.

## Out Of Scope

Do not implement any of the following in Step 5:

- Full `FSISimulation.run()` orchestration.
- Coupled example campaign or full validation benchmark.
- Immersed-boundary no-slip enforcement.
- Solid mask mutation from MPM particles.
- CPIC, cut cells, rigid bodies, fracture, plasticity, contact, or multimaterial
  coupling.
- Sparse LBM or sparse MPM grids.
- VTK or other output writing.
- Edits under `third_party/`.
- `ti.init()` or `taichi.init()` inside `fsi/` library modules.
- NumPy `to_numpy()` or `from_numpy()` inside coupling hot kernels or `step()`
  except for diagnostics and tests.

## Coupler API Contract

`fsi/coupling.py` must expose:

```python
@ti.data_oriented
class LBMMpmCoupler:
    def __init__(
        self,
        config: CouplingConfig,
        lbm: LBMSolver3D,
        mpm: MPMSolver3D,
    ) -> None:
        ...

    def step(self, lbm_dt: float) -> None:
        ...

    def compute_coupling_forces(self, dt_ratio: float = 1.0) -> None:
        ...

    def clear_coupling_fields(self) -> None:
        ...

    def coupling_force_numpy(self) -> np.ndarray:
        ...

    def solid_volume_fraction_numpy(self) -> np.ndarray:
        ...

    def total_particle_coupling_force(self) -> np.ndarray:
        ...

    def total_fluid_coupling_force(self) -> np.ndarray:
        ...
```

Recommended internal fields:

```python
self.coupling_force = ti.Vector.field(3, ti.f32, shape=(nx, ny, nz))
self.solid_volume_fraction = ti.field(ti.f32, shape=(nx, ny, nz))
self.gamma_field = ti.field(ti.f32, shape=())
self.dt_ratio_field = ti.field(ti.f32, shape=())
self.cell_volume_field = ti.field(ti.f32, shape=())
```

`coupling_force` stores the fluid reaction force density to add to
`lbm.force`.

## Validation Contract

`LBMMpmCoupler.__init__()` must validate:

- `config.validate()`;
- `config.kernel == "quadratic"`;
- `lbm.nx == mpm.nx`;
- `lbm.ny == mpm.ny`;
- `lbm.nz == mpm.nz`;
- `mpm.dx > 0`;
- `config.gamma >= 0`;
- `config.mpm_substeps_per_lbm_step > 0`.

`compute_coupling_forces(dt_ratio)` must validate:

- `dt_ratio >= 0`;
- LBM has been initialized;
- MPM has been initialized.

`step(lbm_dt)` must validate:

- `lbm_dt > 0`;
- LBM has been initialized;
- MPM has been initialized.

The coupler should raise clear `RuntimeError` messages before delegating to
solver internals when either solver has not been initialized.

## Numerical Contract

Use the same quadratic 3x3x3 B-spline support as the MPM solver:

```text
grid_pos = x_p / mpm.dx
base = floor(grid_pos - 0.5)
fx = grid_pos - base
```

For each active MPM particle:

1. Compute quadratic weights.
2. Accumulate valid LBM grid velocities over non-solid in-domain nodes.
3. Normalize by the sum of valid weights.
4. Compute `F_p = gamma * m_p * (u_fluid - v_p)`.
5. Store `F_p` in `mpm.particle_force[p]`.
6. Scatter the equal and opposite reaction to `coupling_force`:

```text
coupling_force[node] += -w_normalized * F_p * dt_ratio / cell_volume
```

For Step 5:

```text
cell_volume = mpm.dx ** 3
```

When `dt_ratio = 1.0`, force balance should satisfy:

```text
sum_particle_forces + sum_fluid_coupling_force * cell_volume ~= 0
```

When `dt_ratio = mpm_dt / lbm_dt`, the accumulated fluid reaction is the
time-averaged force density applied to the single LBM step.

## Force Direction Invariant

If LBM velocity is positive x and MPM particle velocity is zero:

```text
u_fluid = (+x)
v_particle = 0
F_particle = gamma * m * (+x - 0)
```

Then:

- total particle coupling force x component must be positive;
- total fluid reaction force x component must be negative;
- the two must balance after multiplying fluid force density by cell volume
  for `dt_ratio = 1.0`.

## Solid Volume Fraction Contract

If `config.build_solid_volume_fraction` is true:

- clear `solid_volume_fraction` before rebuilding;
- scatter particle volumes to valid non-solid LBM cells with normalized weights;
- divide by `cell_volume`;
- clamp each cell fraction to at most `1.0`.

If `build_solid_volume_fraction` is false:

- keep the field cleared during force computation and stepping.

The field is diagnostic only in Step 5. It must not mutate the LBM solid mask.

## LBM Force Application Contract

Prefer applying `coupling_force` inside `LBMMpmCoupler` instead of expanding the
public LBM API.

For enabled coupling:

```text
lbm.reset_force()
lbm.force[cell] += coupling_force[cell] for non-solid cells only
lbm.step()
```

For disabled coupling:

```text
lbm.reset_force()
lbm.step()
```

Do not add coupling force to LBM solid cells.

## MPM Contract

The MPM solver already has:

- `particle_force`;
- `clear_particle_force()`;
- `substep(dt=...)`;
- particle position and velocity diagnostics.

Add this diagnostic helper if useful:

```python
def particle_forces_numpy(self) -> np.ndarray:
    count = self.particle_count()
    return self.particle_force.to_numpy()[:count].copy()
```

No other MPM algorithm change should be required for Step 5.

## Test Plan

Create `tests/test_coupling.py` with focused tests:

1. Construction validates matching grid shape.
2. `gamma = 0` produces zero MPM particle force and zero LBM coupling force.
3. Positive LBM x velocity drives zero-velocity particles in positive x.
4. Equal-and-opposite force balance holds for `dt_ratio = 1.0`.
5. Coupling force is zero on LBM solid cells.
6. `step(lbm_dt)` runs several finite coupled steps.
7. `enabled=False` performs uncoupled stepping with zero coupling force.
8. Invalid `lbm_dt` values fail.
9. Coupler raises clear errors if LBM or MPM is not initialized.

Update `tests/test_config.py`:

- remove the expectation that `LBMMpmCoupler.step()` raises
  `NotImplementedError`;
- keep `FSISimulation.run()` as the remaining placeholder.

Existing LBM and MPM tests must continue to pass.

## Documentation Contract

Update `README.md`:

- current status becomes Step 5 explicit LBM-MPM two-way coupling MVP;
- implemented list includes explicit penalty-based LBM-MPM two-way coupling;
- not implemented list becomes coupled examples and validation tests;
- roadmap item 5 is checked;
- roadmap item 6 remains pending.

Do not add a full coupled example unless Step 6 is explicitly pulled forward.

## Acceptance Commands

Final state must pass:

```powershell
python -m pip install -e ".[dev]"
pytest tests\test_coupling.py -q
pytest -q
python examples\smoke_import.py
python examples\lbm_standalone.py
python examples\lbm_local_force.py
python examples\mpm_standalone_cube.py
python -m compileall fsi tests examples
ruff check .
git diff -- third_party
Select-String -Path fsi\*.py -Pattern 'ti\.init\(|taichi\.init\('
```

Expected:

- all tests pass;
- examples run successfully;
- `compileall` passes;
- `ruff check .` passes;
- `git diff -- third_party` has no output;
- the `ti.init` scan has no output.

## Completion And Push Contract

After implementation and verification:

1. Review staged diff and confirm only Step 5 files are included.
2. Commit with:

```text
feat(coupling): implement lbm-mpm penalty coupling
```

3. Push `main -> origin/main` for `lizhuoh9/taichi-mpm-lbm`.
4. Verify the remote `refs/heads/main` hash matches the local commit.
5. Report the commit hash, pushed branch, README update, and verification
   results.

## Definition Of Done

Step 5 is complete when:

- `LBMMpmCoupler` no longer raises `NotImplementedError`;
- MPM particles receive penalty force from interpolated LBM velocity;
- LBM receives equal-and-opposite reaction force density;
- force direction tests pass;
- force balance tests pass;
- LBM solid cells receive no coupling force;
- disabled coupling remains uncoupled but still steps both solvers;
- several coupled steps remain finite;
- README marks Step 5 complete and Step 6 pending;
- `third_party/` remains untouched;
- no `ti.init()` or `taichi.init()` exists inside `fsi/`;
- the implementation is committed and pushed to `origin/main`.
