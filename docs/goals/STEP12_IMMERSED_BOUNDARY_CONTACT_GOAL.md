# Step 12 Immersed-Boundary And Contact MVP Goal

## Objective

Step 12 adds a deliberately small immersed-boundary/contact MVP on top of the
existing penalty-based LBM-MPM coupling stack.

The goal is to let MPM-occupied regions optionally affect the LBM fluid field
through a dynamic solid volume fraction and Brinkman-style drag force, while
also adding lightweight contact diagnostics and optional particle velocity
damping. This is an MVP for regression-safe solver development, not a full
no-slip immersed-boundary method and not a rigid-body contact solver.

The implementation must preserve all existing default behavior. Every new
immersed-boundary/contact feature is opt-in and disabled by default so the Step
11 committed reference baselines continue to protect the current penalty
coupling behavior.

## Source Of Truth

This file is the implementation contract for Step 12. The short Codex goal for
this work should reference this path directly:

```text
docs/goals/STEP12_IMMERSED_BOUNDARY_CONTACT_GOAL.md
```

If there is any conflict between a speculative idea and this contract, follow
this contract.

## Starting Point

Step 11 is complete on `origin/main` at commit:

```text
2fadd5580289b6bd3122d6973b118c0473591153
```

The current code already has:

- `CouplingConfig` with penalty-coupling stability controls.
- `LBMMpmCoupler.solid_volume_fraction`, built from MPM particles during
  coupling diagnostics.
- `LBMMpmCoupler.coupling_force` and particle coupling diagnostics.
- `FSISimulation.diagnostics()`.
- NPZ/VTK snapshot output and post-processing time-series extraction.
- Step 8/9 validation cases.
- Step 11 JSON reference-data validation.

Step 12 should reuse these pieces instead of creating a separate solver path.

## In Scope

- Add opt-in immersed-boundary/contact controls to `CouplingConfig`.
- Reuse the existing particle-scattered `solid_volume_fraction` as the dynamic
  immersed occupancy field.
- Add an optional Brinkman-style immersed-boundary fluid resistance:

```text
F_ib(cell) = -ib_drag * phi(cell) * rho(cell) * u(cell)
```

- Keep dynamic immersed occupancy separate from the static LBM `solid` mask.
- Add immersed-boundary force fields and scalar diagnostics to
  `LBMMpmCoupler`.
- Add optional maximum per-cell immersed-boundary force clipping.
- Add lightweight contact candidate diagnostics for MPM particles near dynamic
  occupied or static solid support regions.
- Add optional bounded contact velocity damping for contact candidates.
- Expose immersed-boundary/contact diagnostics through
  `LBMMpmCoupler.coupling_diagnostics()` and `FSISimulation.diagnostics()`.
- Include immersed-boundary/contact arrays and counters in NPZ output when
  `write_coupling_fields=True`.
- Add post-processing extraction for immersed-boundary/contact time series.
- Add Step 12 validation cases to `fsi.validation_cases`.
- Add one committed Step 12 reference JSON dataset under `data/reference/`.
- Update the Step 11 reference suite so it includes the Step 12 reference case
  without changing the older committed baselines unless their default behavior
  actually changes.
- Add a small smoke example.
- Update README, examples README, post-processing docs, and a dedicated Step 12
  doc.
- Add focused tests and keep the full suite passing.

## Out Of Scope

- Full no-slip immersed-boundary method.
- Cut-cell methods.
- CPIC.
- Rigid-body contact mechanics.
- Particle-particle contact.
- Fracture, plasticity, or multi-material contact.
- Replacing the existing Step 5/9 penalty coupling model.
- Writing dynamic particle occupancy back into the static LBM `solid` mask by
  default.
- High-fidelity external experimental or paper-level validation claims.
- Large generated artifacts or committed `outputs/`.
- Edits under `third_party/`.
- Calling `ti.init()` or `taichi.init()` from any module under `fsi/`.

## Configuration Contract

Extend `CouplingConfig` with these fields:

```python
immersed_boundary_enabled: bool = False
immersed_boundary_drag: float = 0.0
immersed_boundary_fraction_threshold: float = 0.1
immersed_boundary_max_force: float | None = None

contact_enabled: bool = False
contact_velocity_damping: float = 0.0
contact_fraction_threshold: float = 0.5
```

### Field Semantics

`immersed_boundary_enabled` enables dynamic immersed-boundary fluid forcing.
Default is `False`.

`immersed_boundary_drag` is the Brinkman-style resistance coefficient. It must
be non-negative. A zero value produces no immersed-boundary force even when the
feature is enabled.

`immersed_boundary_fraction_threshold` suppresses force application in cells
where dynamic occupancy is below the threshold. It must be in `[0, 1]`.

`immersed_boundary_max_force` optionally limits the norm of each cell's
immersed-boundary force vector before it is added to `lbm.force`. It must be
positive when provided.

`contact_enabled` enables contact candidate diagnostics and optional particle
velocity damping. Default is `False`.

`contact_velocity_damping` is a bounded multiplicative damping amount applied
to contact candidates when contact is enabled:

```text
v_p <- (1 - contact_velocity_damping) * v_p
```

It must be in `[0, 1]`.

`contact_fraction_threshold` is the dynamic/static support threshold used to
classify a particle as a contact candidate. It must be in `[0, 1]`.

### Validation Rules

`CouplingConfig.validate()` must reject:

- negative `immersed_boundary_drag`
- `immersed_boundary_fraction_threshold` outside `[0, 1]`
- non-positive `immersed_boundary_max_force` when provided
- `contact_velocity_damping` outside `[0, 1]`
- `contact_fraction_threshold` outside `[0, 1]`

Existing validation behavior must remain unchanged.

## Coupler Field Contract

Add these fields to `LBMMpmCoupler`:

```python
self.immersed_boundary_force = ti.Vector.field(3, ti.f32, shape=(nx, ny, nz))
self.ib_drag_field = ti.field(ti.f32, shape=())
self.ib_fraction_threshold_field = ti.field(ti.f32, shape=())
self.ib_max_force_field = ti.field(ti.f32, shape=())
self.use_ib_max_force_field = ti.field(ti.i32, shape=())
self.ib_enabled_field = ti.field(ti.i32, shape=())

self.ib_active_cell_count = ti.field(ti.i32, shape=())
self.ib_clipped_cell_count = ti.field(ti.i32, shape=())
self.ib_total_force = ti.Vector.field(3, ti.f32, shape=())

self.contact_enabled_field = ti.field(ti.i32, shape=())
self.contact_velocity_damping_field = ti.field(ti.f32, shape=())
self.contact_fraction_threshold_field = ti.field(ti.f32, shape=())
self.contact_candidate_count = ti.field(ti.i32, shape=())
self.contact_damped_particle_count = ti.field(ti.i32, shape=())
self.particle_contact_mask = ti.field(ti.i32, shape=mpm.max_particles)
```

If implementation can use fewer scalar fields without weakening behavior or
testability, that is acceptable. The public behavior and diagnostics must remain
equivalent.

## Coupler Public API Contract

Add:

```python
def immersed_boundary_force_numpy(self) -> np.ndarray:
    ...

def particle_contact_mask_numpy(self) -> np.ndarray:
    ...

def immersed_boundary_diagnostics(self) -> dict[str, int | float | np.ndarray]:
    ...
```

`particle_contact_mask_numpy()` must return only active particle entries, like
the existing particle mask/weight accessors.

`coupling_diagnostics()` must include at least:

```python
{
    "ib_active_cell_count": int,
    "ib_clipped_cell_count": int,
    "ib_total_force_x": float,
    "ib_total_force_y": float,
    "ib_total_force_z": float,
    "ib_total_force_norm": float,
    "contact_candidate_count": int,
    "contact_damped_particle_count": int,
}
```

Existing keys must remain present and keep their meaning.

## Immersed-Boundary Force Contract

The dynamic immersed-boundary force must:

1. Use the current `solid_volume_fraction` field as `phi`.
2. Use the current LBM density and velocity.
3. Apply only on LBM cells where `lbm.solid[cell] == 0`.
4. Apply only where `phi >= immersed_boundary_fraction_threshold`.
5. Be zero when `immersed_boundary_enabled=False`.
6. Be zero when `immersed_boundary_drag == 0`.
7. Oppose the local fluid velocity.
8. Optionally clip per-cell force norm when
   `immersed_boundary_max_force` is provided.
9. Record active and clipped cell counts.
10. Record the total immersed-boundary force vector.

Recommended Taichi structure:

```python
@ti.kernel
def _compute_immersed_boundary_force_kernel(self):
    for cell in ti.grouped(self.lbm.rho):
        self.immersed_boundary_force[cell] = ti.Vector([0.0, 0.0, 0.0])
        if self.lbm.solid[cell] == 0:
            phi = self.solid_volume_fraction[cell]
            if phi >= threshold:
                force = -drag * phi * self.lbm.rho[cell] * self.lbm.v[cell]
                # optional clipping
                self.immersed_boundary_force[cell] = force
```

Then add the field to `lbm.force` before `lbm.step()`.

## Step Order Contract

`LBMMpmCoupler.step()` should preserve the existing partitioned workflow and add
immersed-boundary/contact processing in this order:

```text
clear coupling, immersed-boundary, and contact fields
for each MPM substep:
    clear particle force
    if coupling enabled:
        clear particle coupling diagnostics
        compute penalty coupling force and dynamic solid volume fraction
    mpm.substep()
    if contact enabled:
        classify contact candidates
        optionally damp contact-candidate velocities
clamp solid volume fraction
lbm.reset_force()
apply penalty coupling force to LBM
if immersed boundary enabled:
    compute immersed-boundary force from current rho, velocity, and phi
    add immersed-boundary force to LBM
lbm.step()
increment coupled step counter
```

The implementation may classify/damp contact once per LBM step instead of once
per MPM substep if tests and docs state that MVP behavior. It must be stable and
deterministic.

## Contact MVP Contract

Contact handling is diagnostic and optional. It must not claim full contact
physics.

For each active particle:

1. Interpolate dynamic `solid_volume_fraction` around the particle using the
   same quadratic 3x3x3 support.
2. Also count static LBM solid support in the same neighborhood.
3. Mark `particle_contact_mask[p] = 1` if dynamic support or static support
   reaches `contact_fraction_threshold`.
4. Increment `contact_candidate_count`.
5. If `contact_velocity_damping > 0`, damp particle velocity and increment
   `contact_damped_particle_count`.

The damping factor must be bounded by `CouplingConfig.validate()`.

## Static Solid Mask Separation

Step 12 must not mutate `lbm.solid` from dynamic MPM occupancy. The static LBM
solid mask remains the static boundary/obstacle representation. The dynamic
fields remain:

```text
solid_volume_fraction
immersed_boundary_force
particle_contact_mask
```

This separation is a required stability and regression boundary.

## Output Contract

When `OutputConfig.write_coupling_fields=True`, NPZ snapshots must include:

```text
immersed_boundary_force
ib_total_force
ib_active_cell_count
ib_clipped_cell_count
particle_contact_mask
contact_candidate_count
contact_damped_particle_count
```

VTK fluid output should include `immersed_boundary_force` when coupling fields
are written. VTK particle output should include `contact_mask` when MPM
particles are written. If VTK particle scalar support is too invasive, NPZ is
the acceptance requirement and docs should avoid promising VTK contact masks.

`outputs/` remains ignored and must not be committed.

## Post-Processing Contract

`extract_snapshot_timeseries()` must extract these optional keys when present:

```text
ib_force_norm
ib_total_force_norm
ib_active_cell_count
ib_clipped_cell_count
contact_candidate_count
contact_damped_particle_count
```

`summarize_coupling_diagnostics()` must include these keys when present.

`docs/postprocessing.md` must document the new optional fields.

## Simulation Diagnostics Contract

`FSISimulation.diagnostics()` must include lightweight Step 12 scalar
diagnostics:

```text
coupling_ib_active_cell_count
coupling_ib_clipped_cell_count
coupling_ib_total_force_norm
coupling_contact_candidate_count
coupling_contact_damped_particle_count
```

Names may be slightly different if they clearly preserve the coupling namespace
and tests encode the contract. Do not return large arrays from
`FSISimulation.diagnostics()`.

## Validation Case Contract

Add to `fsi.validation_cases`:

```python
def run_immersed_boundary_drag_case() -> ValidationReport:
    ...

def run_contact_diagnostics_case() -> ValidationReport:
    ...
```

Add both to `run_validation_suite()`.

### `run_immersed_boundary_drag_case`

Setup:

- small periodic LBM grid
- initial LBM velocity in `+x`
- one MPM particle or small particle group producing nonzero dynamic occupancy
- `immersed_boundary_enabled=True`
- positive `immersed_boundary_drag`
- coupling gamma may be zero or small, but `build_solid_volume_fraction` must
  be active

Metrics:

- `ib_active_cell_count >= 1`
- `ib_total_force_x < 0`
- `ib_total_force_norm > 0`
- max fluid speed after an IB-enabled step is finite
- an equivalent no-IB or pre-step speed comparison proves the drag does not
  accelerate the fluid in `+x`

### `run_contact_diagnostics_case`

Setup:

- small coupled simulation
- one MPM particle in or near static LBM solid support, or a deterministic high
  dynamic occupancy region
- `contact_enabled=True`
- `contact_velocity_damping > 0`

Metrics:

- `contact_candidate_count >= 1`
- `contact_damped_particle_count >= 1`
- `particle_contact_mask_sum >= 1`
- particle speed after contact damping is lower than the initial speed

## Reference Data Contract

Add one committed JSON baseline:

```text
data/reference/immersed_boundary_contact_reference.json
```

Update `fsi.reference_cases` with:

```python
def compute_immersed_boundary_contact_reference_metrics() -> dict[str, float]:
    ...
```

Add the dataset to `build_reference_datasets()` and `_current_metrics_by_case()`.

Metrics should include:

```text
ib_active_cell_count
ib_total_force_x
ib_total_force_norm
contact_candidate_count
contact_damped_particle_count
```

Existing Step 11 JSON files should remain unchanged because default IB/contact
behavior remains disabled. If any old baseline changes, inspect and justify the
diff before committing.

## Test Contract

Add:

```text
tests/test_immersed_boundary_contact.py
```

Required focused tests:

1. `CouplingConfig` rejects invalid immersed-boundary drag.
2. `CouplingConfig` rejects invalid immersed-boundary fraction thresholds.
3. `CouplingConfig` rejects invalid immersed-boundary max force.
4. `CouplingConfig` rejects invalid contact damping.
5. `CouplingConfig` rejects invalid contact fraction thresholds.
6. With dynamic solid fraction present and IB disabled, immersed-boundary force
   remains zero and active cell count is zero.
7. With IB enabled, drag force opposes positive fluid velocity.
8. IB max force clips per-cell force and records clipped cell count.
9. IB force remains zero on static LBM solid cells.
10. A few coupled steps with IB enabled remain finite.
11. Contact candidate detection marks the particle and increments diagnostics.
12. Contact damping reduces particle speed.
13. NPZ output includes IB/contact arrays and counters.
14. Post-processing extracts IB/contact time-series keys.
15. Validation cases pass and expose expected metrics.
16. Reference validation loads and passes the new committed reference dataset.

Update existing tests where the public output/schema contract grows:

- `tests/test_config.py`
- `tests/test_output.py`
- `tests/test_postprocess.py`
- `tests/test_reference_validation.py`
- `tests/test_validation_benchmarks.py`

Keep tests focused. Do not create long-running physical benchmarks.

## Example Contract

Add:

```text
examples/immersed_boundary_contact_smoke.py
```

The example should:

- call `ti.init()` only inside the example entry point
- create a tiny periodic coupled case
- enable immersed-boundary drag
- enable contact diagnostics/damping
- run a small number of steps
- print scalar diagnostics:
  - step
  - max fluid speed
  - max particle speed
  - IB active cell count
  - IB total force norm
  - contact candidate count
- avoid heavy output by default

## Documentation Contract

Add:

```text
docs/immersed_boundary_contact.md
```

It must describe:

- what the MVP does
- what it explicitly does not do
- new `CouplingConfig` fields
- IB force formula and limitations
- contact diagnostics/damping behavior
- NPZ/post-processing diagnostics
- example command
- no high-fidelity validation claim

Update:

- `README.md`
- `examples/README.md`
- `docs/postprocessing.md`
- `docs/reference_validation.md`
- `data/reference/README.md`

README should move current status to:

```text
Step 12: immersed-boundary/contact MVP.
```

README should move "advanced immersed-boundary/contact handling" out of the
"Not implemented yet" list and replace it with a more honest future item such
as "advanced rigid-body/contact mechanics".

## Public API Contract

No new top-level package exports are required unless the implementation adds a
new user-facing helper module. If public exports are added, update
`tests/test_import.py`.

Existing imports must keep working.

## Acceptance Commands

Use the trusted Taichi environment:

```powershell
D:\working\taichi\env\python.exe -m pip install -e ".[dev]"

D:\working\taichi\env\python.exe -m pytest tests\test_immersed_boundary_contact.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_coupling_stability.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_output.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_postprocess.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_reference_validation.py -q
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
D:\working\taichi\env\python.exe examples\postprocess_snapshots.py
D:\working\taichi\env\python.exe examples\postprocess_validation_summary.py
D:\working\taichi\env\python.exe examples\reference_validation_suite.py
D:\working\taichi\env\python.exe examples\immersed_boundary_contact_smoke.py

D:\working\taichi\env\python.exe -m compileall fsi tests examples
ruff check .
git diff --check
git diff -- third_party
Select-String -Path fsi\*.py -Pattern 'ti\.init\(|taichi\.init\('
git status --short --ignored outputs
```

Expected:

- focused Step 12 tests pass
- existing coupling, output, postprocess, and reference tests pass
- fast suite passes
- full suite passes
- all examples pass
- compileall passes
- ruff passes
- `third_party` diff is empty
- no `ti.init()`/`taichi.init()` appears under `fsi/`
- generated outputs remain ignored and unstaged

The pre-push hook also runs `pytest -q`; a successful push is part of the final
verification evidence.

## Implementation Order

1. Add this goal file.
2. Create the short Codex goal referencing this file.
3. Add red focused tests in `tests/test_immersed_boundary_contact.py`.
4. Extend `CouplingConfig` fields and validation.
5. Add immersed-boundary/contact fields to `LBMMpmCoupler`.
6. Add clear kernels for immersed-boundary and contact fields.
7. Implement dynamic IB force computation and application to LBM.
8. Implement contact candidate classification and optional velocity damping.
9. Add public diagnostics and numpy accessors.
10. Extend `FSISimulation.diagnostics()`.
11. Extend NPZ output and VTK fluid output where practical.
12. Extend post-processing time-series extraction.
13. Add validation cases.
14. Add reference-case computation and committed JSON baseline.
15. Add the smoke example.
16. Add docs and README updates.
17. Run focused tests.
18. Run full acceptance commands.
19. Commit with:

```text
feat(coupling): add immersed-boundary contact MVP
```

20. Push `main` to `origin/main`.
21. Confirm local HEAD equals `origin/main`.

## Definition Of Done

Step 12 is complete only when all of these are true:

- `docs/goals/STEP12_IMMERSED_BOUNDARY_CONTACT_GOAL.md` exists.
- `CouplingConfig` has IB/contact controls with validation.
- IB/contact controls default to disabled and do not change Step 11 defaults.
- `LBMMpmCoupler` has an immersed-boundary force field.
- IB drag force is computed from `solid_volume_fraction`, LBM density, and LBM
  velocity.
- IB force opposes fluid velocity.
- IB force is not applied to static LBM solid cells.
- IB max-force clipping is available.
- Contact candidate diagnostics are available.
- Optional contact velocity damping is available.
- Coupler and simulation diagnostics include IB/contact scalar diagnostics.
- NPZ output includes IB/contact arrays and counters.
- Post-processing extracts IB/contact time series.
- Step 12 validation cases exist.
- A committed Step 12 reference JSON exists.
- Reference validation suite passes.
- `examples/immersed_boundary_contact_smoke.py` exists and runs.
- README marks Step 12 complete without overstating physical validation.
- Focused and full pytest suites pass.
- Examples pass.
- `compileall` and `ruff` pass.
- `third_party/` has no diff.
- `fsi/*.py` has no `ti.init()`/`taichi.init()`.
- Generated outputs are ignored and not staged.
- The final commit is pushed to `origin/main`.
