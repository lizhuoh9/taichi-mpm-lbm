# Step 9 Coupling Stability And Boundary Handling Goal

## Objective

Improve explicit LBM-MPM penalty-coupling stability and boundary handling while
preserving the Step 5 coupling model.

This step must keep the existing penalty formulation:

```text
F_p = gamma * m_p * (u_fluid_at_particle - v_particle)
```

The goal is not a new immersed-boundary method. The goal is practical stability
guards, boundary-support diagnostics, and regression validation for
near-boundary, near-solid, unsupported, and high-gamma smoke cases.

## Starting Point

Step 8 completed benchmark-style validation cases in commit
`7d291c6ee769d849f6730e99744996feb623ec94`.

The repository currently has:

- project skeleton and configuration system
- dense standalone 3D D3Q19/MRT LBM solver wrapper
- local per-cell LBM force field
- dense standalone 3D elastic MLS-MPM solver
- explicit penalty-based LBM-MPM two-way coupling
- top-level coupled simulation runner
- lightweight coupled examples and validation tests
- NPZ/VTK simulation output snapshots
- benchmark-style validation cases

## In Scope

- Add optional coupling force limiting.
- Add optional relative-velocity limiting before force computation.
- Add optional gamma ramp over early coupled steps.
- Add valid kernel-support diagnostics for coupling interpolation/scatter.
- Track particles with no valid fluid support.
- Track particles with partial valid support near domain boundaries or LBM solid
  cells.
- Keep equal-and-opposite force balance valid after clipping and ramping.
- Add public coupler diagnostic accessors.
- Extend `FSISimulation.diagnostics()` with lightweight coupling diagnostic
  counts.
- Extend NPZ coupling output with particle support and diagnostic count fields.
- Add tests for config validation, force limiting, relative-velocity limiting,
  gamma ramp, boundary support diagnostics, solid-mask support diagnostics,
  unsupported support behavior, and high-gamma finite runs.
- Add Step 9 validation cases using the Step 8 validation report format.
- Add one small stability/boundary example.
- Update README and examples README.

## Out Of Scope

- New coupling formulas.
- Full immersed-boundary no-slip enforcement.
- Writing MPM solid volume fraction back to the LBM solid mask.
- Rigid-body contact, fracture, CPIC, plasticity, or multimaterial contact.
- Sparse grids.
- Large benchmark campaigns.
- Heavy output artifacts committed to git.
- Edits under `third_party/`.
- Calling `ti.init()` inside `fsi/`.

## Configuration Contract

Extend `CouplingConfig` with:

```python
force_limit: float | None = None
relative_velocity_limit: float | None = None
gamma_ramp_steps: int = 0
min_valid_weight: float = 1.0e-6
```

### `force_limit`

If provided, cap the norm of the final particle coupling force:

```text
if ||F_p|| > force_limit:
    F_p = F_p / ||F_p|| * force_limit
```

`None` means no force clipping and must preserve existing Step 5/8 behavior.

### `relative_velocity_limit`

If provided, cap the norm of the relative velocity used by the penalty force:

```text
dv = u_fluid - v_particle
if ||dv|| > relative_velocity_limit:
    dv = dv / ||dv|| * relative_velocity_limit
F_p = gamma_eff * m_p * dv
```

`None` means no relative-velocity clipping.

### `gamma_ramp_steps`

Avoid initial coupling shock by ramping effective gamma over early coupled
steps:

```text
gamma_eff = gamma * min(1.0, (coupled_step + 1) / gamma_ramp_steps)
```

`gamma_ramp_steps = 0` disables ramping and preserves existing behavior.

### `min_valid_weight`

Treat particles with valid fluid support weight at or below this threshold as
unsupported:

```text
if valid_weight_sum <= min_valid_weight:
    particle_force = 0
    particle_coupling_mask = 0
    unsupported_particle_count += 1
```

This prevents tiny support sums from being normalized into large local effects.

### Validation Rules

`CouplingConfig.validate()` must reject:

- negative `gamma`
- nonpositive `mpm_substeps_per_lbm_step`
- unsupported `kernel`
- nonpositive `force_limit` when provided
- nonpositive `relative_velocity_limit` when provided
- negative `gamma_ramp_steps`
- nonpositive `min_valid_weight`
- `min_valid_weight > 1.0`

## Coupler State And Diagnostics

Add scalar fields to `LBMMpmCoupler`:

- `force_limit_field`
- `relative_velocity_limit_field`
- `use_force_limit_field`
- `use_relative_velocity_limit_field`
- `gamma_ramp_steps_field`
- `coupled_step_field`
- `min_valid_weight_field`

Add particle/diagnostic fields:

- `particle_valid_weight`
- `particle_coupling_mask`
- `unsupported_particle_count`
- `partial_support_particle_count`
- `clipped_particle_count`

### Diagnostic Semantics

```text
particle_valid_weight[p] = valid kernel support weight sum before normalization
particle_coupling_mask[p] = 1 if particle received coupling force, else 0
unsupported_particle_count = active particles with valid_weight_sum <= min_valid_weight
partial_support_particle_count = active particles with min_valid_weight < valid_weight_sum < 1 - tolerance
clipped_particle_count = active particles whose dv or force was clipped
```

The diagnostic fields must be cleared whenever coupling fields are cleared.

## Public Coupler API

Add:

```python
def particle_valid_weights_numpy(self) -> np.ndarray:
    ...

def particle_coupling_mask_numpy(self) -> np.ndarray:
    ...

def coupling_diagnostics(self) -> dict[str, int | float]:
    ...

def effective_gamma(self) -> float:
    ...
```

`coupling_diagnostics()` should return at least:

```python
{
    "unsupported_particle_count": int,
    "partial_support_particle_count": int,
    "clipped_particle_count": int,
    "min_particle_valid_weight": float,
    "mean_particle_valid_weight": float,
    "effective_gamma": float,
}
```

Only active particles should contribute to min/mean support diagnostics.

## Kernel Behavior

Update `_compute_coupling_forces_kernel()` so each active particle:

1. Computes valid support only from in-domain, non-solid LBM cells.
2. Stores `particle_valid_weight[p]`.
3. Marks unsupported particles when valid weight is at or below
   `min_valid_weight`.
4. Counts partial support when valid support is below full support but still
   usable.
5. Interpolates fluid velocity using normalized valid weights.
6. Applies optional relative-velocity limiting.
7. Applies gamma ramp.
8. Computes particle force with the Step 5 penalty model.
9. Applies optional force limiting.
10. Stores final force in `mpm.particle_force[p]`.
11. Scatters exactly the opposite final force density to valid non-solid LBM
    cells.

## Force Balance Invariant

All clipping and ramping must be applied before writing both sides of the
coupling:

```text
mpm.particle_force[p] = final_particle_force
coupling_force[node] += -w * final_particle_force * dt_ratio / cell_volume
```

This must preserve:

```text
sum_particle_force + total_fluid_force_density * cell_volume ~= 0
```

## Gamma Ramp Contract

Use `(coupled_step + 1) / gamma_ramp_steps` so the first coupled step has a
nonzero force when ramping is enabled.

Increment `coupled_step_field` once per completed `LBMMpmCoupler.step()` call,
not once per MPM substep. `compute_coupling_forces()` should report the current
effective gamma without advancing the ramp.

## Boundary Handling Contracts

### Domain Boundary Partial Support

A particle near the simulation-domain boundary should:

- have `particle_valid_weight < 1.0`
- be counted as partial support
- remain coupled if support exceeds `min_valid_weight`
- produce finite particle/fluid forces
- preserve force balance

### Solid-Mask Partial Support

A particle whose kernel overlaps LBM solid cells should:

- have partial support
- not scatter force to solid cells
- remain coupled when enough non-solid support exists
- preserve force balance

### Unsupported Support

A particle whose kernel has no sufficient non-solid fluid support should:

- have `particle_coupling_mask == 0`
- get zero particle force
- produce zero coupling force
- increment `unsupported_particle_count`

## Output Integration

When `write_coupling_fields` is true, NPZ output should also write:

- `coupling_particle_valid_weight`
- `coupling_particle_mask`
- `coupling_unsupported_particle_count`
- `coupling_partial_support_particle_count`
- `coupling_clipped_particle_count`

These arrays/counts are diagnostic output and must not change simulation
behavior.

`FSISimulation.diagnostics()` should include:

- `coupling_unsupported_particle_count`
- `coupling_partial_support_particle_count`
- `coupling_clipped_particle_count`
- `coupling_effective_gamma`

## Step 9 Validation Cases

Extend `fsi.validation_cases` with:

```python
def run_coupling_force_limit_case() -> ValidationReport:
    ...

def run_coupling_boundary_support_case() -> ValidationReport:
    ...
```

### `run_coupling_force_limit_case`

Use a high-gamma, high-relative-velocity setup:

- LBM initial velocity `(1.0, 0.0, 0.0)`
- one MPM particle
- `gamma=100.0`
- `force_limit=0.05`

Metrics:

- `particle_force_norm <= 0.05 + tolerance`
- `clipped_particle_count >= 1`
- `force_balance_norm <= tolerance`

### `run_coupling_boundary_support_case`

Use a particle near the domain boundary:

- particle at approximately `(0.6, 4.0, 4.0)`
- `min_valid_weight=1.0e-6`

Metrics:

- `partial_support_particle_count >= 1`
- `min_particle_valid_weight > min_valid_weight`
- `force_balance_norm <= tolerance`
- particle force is finite

`run_validation_suite()` must include both new Step 9 cases.

## Tests

Add `tests/test_coupling_stability.py` covering:

- config validation for `force_limit`
- config validation for `relative_velocity_limit`
- config validation for `gamma_ramp_steps`
- config validation for `min_valid_weight`
- force limit clips particle force
- relative-velocity limit clips before force computation
- clipped force still balances fluid reaction
- gamma ramp reduces the first force relative to no-ramp coupling
- gamma ramp reaches full gamma after enough completed coupled steps
- near-domain-boundary partial support diagnostics
- solid-mask partial support diagnostics and no force on solid cells
- full solid support produces unsupported particle
- high-gamma finite coupled run with force limit and ramp

Update:

- `tests/test_config.py` for new `CouplingConfig` validation rules
- `tests/test_output.py` for NPZ coupling diagnostic keys
- `tests/test_validation_benchmarks.py` expected suite names

## Example

Add:

```text
examples/coupling_stability_boundary.py
```

The example should:

- call `ti.init(...)` itself
- run a small high-gamma/ramped/clipped near-boundary case
- print compact simulation diagnostics
- print unsupported/partial/clipped counts
- avoid heavy output by default

## README Updates

After implementation, update README to:

- mark current status as Step 9 coupling stability and boundary handling
- add coupling stability guards and boundary-support diagnostics to the
  implemented list
- remove improved coupling stability and boundary handling from not
  implemented
- add `python examples/coupling_stability_boundary.py`
- update next milestones toward visualization, reference-data validation, and
  advanced immersed-boundary/contact handling

Update `examples/README.md` to include the new stability/boundary example.

## Acceptance Commands

Use the trusted Taichi environment:

```powershell
D:\working\taichi\env\python.exe -m pip install -e ".[dev]"
D:\working\taichi\env\python.exe -m pytest tests\test_coupling_stability.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_coupling.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_config.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_validation_benchmarks.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_output.py -q
D:\working\taichi\env\python.exe -m pytest -m "not slow" -q
D:\working\taichi\env\python.exe -m pytest -q
D:\working\taichi\env\python.exe examples\smoke_import.py
D:\working\taichi\env\python.exe examples\lbm_standalone.py
D:\working\taichi\env\python.exe examples\lbm_local_force.py
D:\working\taichi\env\python.exe examples\mpm_standalone_cube.py
D:\working\taichi\env\python.exe examples\coupled_penalty_smoke.py
D:\working\taichi\env\python.exe examples\coupled_output_snapshot.py
D:\working\taichi\env\python.exe examples\validation_benchmark_suite.py
D:\working\taichi\env\python.exe examples\coupling_stability_boundary.py
D:\working\taichi\env\python.exe -m compileall fsi tests examples
D:\working\taichi\env\python.exe -m ruff check .
git diff -- third_party
Select-String -Path fsi\*.py -Pattern 'ti\.init\(|taichi\.init\('
git status --short --ignored outputs
```

Expected results:

- coupling stability tests pass
- existing coupling tests pass
- config/output/validation tests pass
- fast suite passes
- full suite passes
- all examples run
- `compileall` passes
- `ruff` passes
- no `third_party` diff
- no `ti.init()` / `taichi.init()` calls under `fsi/`
- generated outputs remain ignored and unstaged

## Definition Of Done

- `docs/goals/STEP9_COUPLING_STABILITY_BOUNDARY_GOAL.md` is committed.
- `CouplingConfig` exposes and validates stability/boundary-support parameters.
- `LBMMpmCoupler` supports `force_limit`.
- `LBMMpmCoupler` supports `relative_velocity_limit`.
- `LBMMpmCoupler` supports `gamma_ramp_steps`.
- `LBMMpmCoupler` supports `min_valid_weight`.
- Coupler records `particle_valid_weight` and `particle_coupling_mask`.
- Coupler records unsupported, partial-support, and clipped particle counts.
- Near-boundary particle case is finite and reports partial support.
- Solid-mask partial support case does not scatter force to solid cells.
- Unsupported support case produces no coupling force.
- Clipped/ramped force preserves equal-and-opposite force balance.
- Step 8 validation suite includes Step 9 stability/boundary cases.
- Coupling stability/boundary example exists.
- README marks Step 9 complete.
- Full acceptance commands pass or any unavoidable local-environment exception
  is documented precisely.
- The final commit is pushed to `origin/main`.
