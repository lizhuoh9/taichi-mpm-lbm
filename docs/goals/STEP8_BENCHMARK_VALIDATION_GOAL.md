# Step 8 Benchmark-Style Validation Goal

## Objective

Add lightweight benchmark-style validation cases for the current LBM, MPM,
coupling, simulation, and output pipeline.

This step establishes reproducible regression and behavior validation. It must
not claim high-fidelity physical certification or agreement with external
experimental/reference data.

## Starting Point

Step 7 completed NPZ/VTK output snapshots in commit
`76253f1a02cac2a7814b95d5161f8a2ecb0135d3`.

The repository currently has:

- standalone dense 3D D3Q19/MRT LBM solver wrapper
- local per-cell LBM force field
- standalone dense 3D elastic MLS-MPM solver
- explicit penalty-based LBM-MPM two-way coupling
- top-level `FSISimulation`
- coupled smoke/validation tests
- NPZ/VTK output snapshots

## In Scope

- Add deterministic, small validation cases that run in ordinary automated
  tests.
- Add reusable validation metric/report helpers that are pure Python and do
  not initialize Taichi.
- Validate LBM periodic mass conservation.
- Validate LBM force-response behavior.
- Validate MPM center-of-mass stability under zero gravity.
- Validate MPM downward center-of-mass response under gravity.
- Validate coupled enabled-vs-disabled particle drift.
- Validate coupling force balance in a small deterministic case.
- Add a validation example that runs the suite, prints a compact report, and
  writes JSON under `outputs/`.
- Add tests for helpers and validation cases with loose but meaningful
  tolerances.
- Add a pytest `slow` marker and document a fast development test command
  without excluding important tests from the full `pytest -q` run.
- Harden Step 7 VTK output so it respects `write_lbm_fields`,
  `write_mpm_particles`, and `write_coupling_fields` consistently with NPZ.
- Update README to mark Step 8 complete and align user-facing commands and
  milestones.

## Out Of Scope

- Large benchmark campaigns or long-running validation sweeps.
- External-reference validation against experiments, papers, or Fluent data.
- Claims of high-fidelity physical correctness.
- New coupling formulas or solver-physics changes unless a clear bug blocks
  the Step 8 validation contracts.
- ParaView/rendering workflows beyond writing the existing lightweight VTK
  snapshots.
- Committing generated JSON, NPZ, VTK, cache, or runtime output artifacts.
- Editing `third_party/`.
- Calling `ti.init()` from any module under `fsi/`.

## Public API Targets

Add `fsi/validation.py` with:

- `ValidationMetric`
- `ValidationReport`
- `bounded_metric`
- `finite_metric`
- `relative_error`

`ValidationMetric` should be an immutable dataclass with:

- `name`
- `value`
- optional `lower`
- optional `upper`
- `passed`
- optional `units`
- optional `description`

`ValidationReport` should be an immutable dataclass with:

- `case_name`
- `metrics`
- `metadata`
- `passed` property
- `to_dict()`

Add `fsi/validation_cases.py` with:

- `run_lbm_mass_conservation_case()`
- `run_lbm_force_response_case()`
- `run_mpm_zero_gravity_com_case()`
- `run_mpm_gravity_response_case()`
- `run_coupled_drift_case()`
- `run_coupling_force_balance_case()`
- `run_validation_suite()`

These modules must not call `ti.init()`. Tests use `tests/conftest.py` for
Taichi initialization; examples initialize Taichi explicitly.

## Validation Case Contracts

### LBM Periodic Mass Conservation

Use a small periodic `8 x 8 x 8` LBM grid with no force. Initialize, record
total mass, run a short fixed number of steps, and report:

- `relative_mass_error`, passing at a loose regression tolerance
- `max_velocity_norm`, finite and bounded

### LBM Force Response

Use a small periodic LBM grid with a tiny positive x-force. Initialize, record
mean x velocity, run a short fixed number of steps, and report:

- `mean_ux_growth`, passing when positive
- `max_velocity_norm`, finite and bounded

Do not overfit to exact acceleration.

### MPM Zero-Gravity COM Stability

Use a small `16 x 16 x 16` MPM domain with a central particle box, zero gravity,
and short fixed substeps. Report:

- `center_of_mass_drift_norm`, passing at a loose regression tolerance
- `max_velocity_norm`, finite and bounded

### MPM Gravity Response

Use a small MPM domain with weak downward gravity and short fixed substeps.
Report:

- `center_of_mass_y_delta`, passing when negative
- finite positions/velocities checks

### Coupled Enabled-vs-Disabled Drift

Run two matched `FSISimulation` cases:

- coupling enabled
- coupling disabled

Use a tiny initial LBM flow in +x, no gravity, one MPM particle, and a short
step count. Report:

- enabled x drift
- disabled x drift
- enabled-minus-disabled margin

The enabled drift should exceed the disabled drift by a meaningful tolerance.

### Coupling Force Balance

Use one particle in uniform flow and call
`coupler.compute_coupling_forces(dt_ratio=1.0)`. Report:

- particle force x is positive
- fluid reaction force x is negative
- norm of total particle force plus integrated fluid force is small

Use the coupler cell volume when comparing force density to particle force.

## Step 7 VTK Hardening

`SimulationOutputWriter.write_vtk_snapshot()` must respect output flags:

- write a fluid VTK file only when LBM fields or coupling fields are enabled
- write a particle VTK file only when MPM particle output is enabled
- return an empty list when all output groups are disabled

`_write_fluid_vtk()` must include:

- LBM fields only when `write_lbm_fields` is true
- coupling fields only when `write_coupling_fields` is true

Add tests covering:

- VTK coupling-only output writes only a fluid file
- VTK all-groups-disabled output returns no files

## Tests

Add `tests/test_validation_benchmarks.py` covering:

- `bounded_metric` pass/fail behavior
- `relative_error`
- `ValidationReport.to_dict()`
- LBM mass conservation validation report passes
- LBM force response validation report passes
- MPM zero-gravity COM validation report passes
- MPM gravity validation report passes
- coupled drift validation report passes
- coupling force-balance validation report passes
- composed validation suite returns all expected reports and passes

Extend existing output tests for the VTK flag behavior described above.

Extend import tests to include the new public validation helpers.

## Example

Add `examples/validation_benchmark_suite.py`.

The example must:

- call `ti.init(...)` itself
- run `run_validation_suite()`
- print one PASS/FAIL line per report
- print compact metric values
- write `outputs/validation_benchmark_suite/validation_summary.json`
- exit nonzero if any report fails

The generated JSON must remain ignored by git through the existing `outputs/`
ignore rule.

## README Updates

After implementation, update README to:

- mark current status as Step 8 benchmark-style validation cases
- include benchmark-style validation cases in the implemented list
- keep high-fidelity external validation out of the implemented list
- add `pytest -m "not slow" -q` as a fast development command
- add `python examples/validation_benchmark_suite.py`
- update next milestones toward coupling stability, richer visualization, and
  higher-fidelity reference-data validation

## Acceptance Commands

Use the trusted Taichi environment:

```powershell
D:\working\taichi\env\python.exe -m pip install -e ".[dev]"
D:\working\taichi\env\python.exe -m pytest tests\test_validation_benchmarks.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_output.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_simulation.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_coupled_validation.py -q
D:\working\taichi\env\python.exe -m pytest -m "not slow" -q
D:\working\taichi\env\python.exe -m pytest -q
D:\working\taichi\env\python.exe examples\smoke_import.py
D:\working\taichi\env\python.exe examples\lbm_standalone.py
D:\working\taichi\env\python.exe examples\lbm_local_force.py
D:\working\taichi\env\python.exe examples\mpm_standalone_cube.py
D:\working\taichi\env\python.exe examples\coupled_penalty_smoke.py
D:\working\taichi\env\python.exe examples\coupled_output_snapshot.py
D:\working\taichi\env\python.exe examples\validation_benchmark_suite.py
D:\working\taichi\env\python.exe -m compileall fsi tests examples
ruff check .
git diff -- third_party
Select-String -Path fsi\*.py -Pattern 'ti\.init\(|taichi\.init\('
git status --short --ignored outputs
```

Expected results:

- focused validation tests pass
- output hardening tests pass
- simulation/coupled regressions pass
- fast suite passes
- full suite passes
- all examples run
- validation summary is written under ignored `outputs/`
- `compileall` passes
- `ruff` passes
- no `third_party` diff
- no `ti.init()` / `taichi.init()` calls under `fsi/`
- generated outputs are ignored and not staged

## Definition Of Done

- `docs/goals/STEP8_BENCHMARK_VALIDATION_GOAL.md` is committed.
- Validation metric/report helpers exist and are pure Python.
- Deterministic validation cases exist and are covered by tests.
- Validation benchmark example exists and writes JSON under `outputs/`.
- Step 7 VTK output flags are hardened and covered by tests.
- README marks Step 8 complete.
- Full acceptance commands pass or any unavoidable local-environment exception
  is documented precisely.
- The final commit is pushed to `origin/main`.
