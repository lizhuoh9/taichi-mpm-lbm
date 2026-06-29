# Step 7 Output Snapshots Goal

## Objective

Implement lightweight simulation output snapshots for coupled LBM-MPM runs.

Step 7 builds on Step 6:

```text
FSISimulation.run()
  -> optional output snapshots
      -> NPZ fields for analysis
      -> VTK smoke files for visualization tools
```

The goal is to make simulation state saveable and reproducible for later
analysis, validation, and visualization work. This step must not claim
high-fidelity benchmark validation.

## In Scope

- Add a dedicated output writer module.
- Use the existing `OutputConfig`.
- Write NPZ snapshots with metadata, LBM fields, MPM particle fields, and
  coupling fields.
- Support `OutputConfig.write_lbm_fields`, `write_mpm_particles`, and
  `write_coupling_fields`.
- Support `output_format == "npz"`.
- Support `output_format == "vtk"` with smoke-level VTK files.
- Support `output_format == "both"` by writing NPZ and VTK files.
- Integrate output writing into `FSISimulation` with explicit opt-in controls.
- Keep `FSISimulation.run()` default behavior unchanged: no files written
  unless requested.
- Add a `write_snapshot()` method on `FSISimulation`.
- Add interval-based output writing through `run(write_output=True)`.
- Add a small output-focused example.
- Add tests using `tmp_path`; tests must not write to the repository root.
- Update README to mark Step 7 complete and keep next milestones honest.

## Out Of Scope

Do not implement any of the following in Step 7:

- High-fidelity benchmark validation.
- Rendering images or videos.
- Long-running visualization campaigns.
- Heavy output artifacts committed to git.
- Changes to LBM, MPM, or coupling numerical formulas.
- LBM solid-mask mutation from MPM particles.
- Advanced visualization or post-processing tooling.
- Edits under `third_party/`.
- `ti.init()` or `taichi.init()` inside `fsi/` library modules.

## Output Module Contract

Add:

```text
fsi/output.py
```

It must expose:

```python
class SimulationOutputWriter:
    def __init__(self, config: OutputConfig) -> None:
        ...

    def ensure_output_dir(self) -> None:
        ...

    def should_write(self, step: int) -> bool:
        ...

    def write_snapshot(self, sim: FSISimulation) -> list[Path]:
        ...

    def write_npz_snapshot(self, sim: FSISimulation) -> Path:
        ...

    def write_vtk_snapshot(self, sim: FSISimulation) -> list[Path]:
        ...
```

Use `TYPE_CHECKING` to avoid a runtime import cycle:

```python
if TYPE_CHECKING:
    from .simulation import FSISimulation
```

The writer must not call `ti.init()`.

## File Naming Contract

Use zero-padded step numbers:

```text
snapshot_000000.npz
fluid_000000.vti
particles_000000.vtu
```

When using pyevtk functions, return the actual path created by the function.
Do not assume pyevtk file suffix behavior without normalizing the returned
result.

## NPZ Snapshot Contract

NPZ output is the primary Step 7 output and must be strictly tested.

Every NPZ snapshot must include metadata:

```python
"step"                # int64 scalar
"time"                # float64 scalar
"lbm_shape"           # int32 array [nx, ny, nz]
"mpm_particle_count"  # int32 scalar
```

When `OutputConfig.write_lbm_fields` is true, include:

```python
"lbm_density"   # shape: (nx, ny, nz)
"lbm_velocity"  # shape: (nx, ny, nz, 3)
"lbm_force"     # shape: (nx, ny, nz, 3)
"lbm_solid"     # shape: (nx, ny, nz)
```

When `OutputConfig.write_mpm_particles` is true, include:

```python
"mpm_positions"              # shape: (n, 3)
"mpm_velocities"             # shape: (n, 3)
"mpm_particle_forces"        # shape: (n, 3)
"mpm_deformation_gradients"  # shape: (n, 3, 3)
"mpm_active"                 # shape: (n,)
```

When `OutputConfig.write_coupling_fields` is true, include:

```python
"coupling_force"                  # shape: (nx, ny, nz, 3)
"solid_volume_fraction"           # shape: (nx, ny, nz)
"total_particle_coupling_force"   # shape: (3,)
"total_fluid_coupling_force"      # shape: (3,)
```

Use existing solver and coupler diagnostics. Do not use NumPy transfer inside
Taichi kernels.

## VTK Snapshot Contract

VTK output is smoke-level in Step 7.

When `output_format == "vtk"` or `"both"`, write:

- a fluid grid file containing scalar/vector fields that pyevtk can write;
- a particle point file containing particle positions and velocity/force data.

Suggested pyevtk functions:

- `imageToVTK()` for uniform grid fields;
- `pointsToVTK()` for MPM particle points.

The VTK tests only need to assert that returned files exist and have expected
VTK suffixes. Do not add strict contents tests for VTK in Step 7.

## FSISimulation Integration Contract

Extend `FSISimulation`:

```python
self.output_writer = SimulationOutputWriter(config.output)
```

Add:

```python
def write_snapshot(self) -> list[Path]:
    self._validate_initialized()
    return self.output_writer.write_snapshot(self)
```

Extend `run()`:

```python
def run(
    self,
    steps: int | None = None,
    write_output: bool = False,
    write_initial: bool = False,
) -> list[dict[str, object]]:
    ...
```

Behavior:

- `run()` default remains no output writing.
- If `write_output and write_initial`, write step 0 before stepping.
- After each `step()`, write a snapshot if
  `output_writer.should_write(step_index)` is true.
- `run()` still returns diagnostics history.

## Public API Contract

Update `fsi.__init__` to expose `SimulationOutputWriter` through lazy public
exports, consistent with the Step 6 lazy solver/coupler/simulation exports.

Existing config and solver exports must remain.

## Example Contract

Add:

```text
examples/coupled_output_snapshot.py
```

The example must:

- call `ti.init()` in the script;
- use a small coupled periodic case;
- configure `OutputConfig(output_dir=Path("outputs/coupled_output_snapshot"))`;
- use `output_interval=5`;
- use `output_format="npz"` by default;
- run `sim.run(steps=10, write_output=True, write_initial=True)`;
- print written snapshot file names;
- not commit generated output files.

Expected generated files:

```text
snapshot_000000.npz
snapshot_000005.npz
snapshot_000010.npz
```

The repository `.gitignore` must keep `outputs/` ignored.

## Test Plan

Add `tests/test_output.py`.

Required tests:

1. `SimulationOutputWriter.should_write()` respects `output_interval`.
2. `sim.write_snapshot()` writes `snapshot_000001.npz` after one step.
3. NPZ includes metadata and expected LBM/MPM/coupling keys and shapes.
4. Output field flags include and exclude the expected NPZ groups.
5. `run(steps=5, write_output=True)` writes interval snapshots only.
6. `run(..., write_output=True, write_initial=True)` writes step 0 too.
7. `run()` without output flags writes no files.
8. `write_snapshot()` before initialization raises `RuntimeError`.
9. `output_format="vtk"` writes existing VTK smoke files.
10. `output_format="both"` writes NPZ and VTK files.

Update `tests/test_simulation.py` as needed for the extended `run()` signature.

Update `tests/test_config.py` with explicit `OutputConfig` validation tests:

- invalid interval fails;
- invalid output format fails.

Existing tests must continue to pass.

## README Contract

Update README:

- current status becomes Step 7 NPZ/VTK output snapshots;
- implemented list includes NPZ/VTK simulation output snapshots;
- not implemented list becomes:
  - high-fidelity validation benchmarks;
  - advanced visualization/rendering workflow;
  - improved coupling stability and boundary handling;
- development commands include:

```bash
python examples/coupled_output_snapshot.py
```

- next milestones remove output snapshots and keep:
  - benchmark-style validation cases;
  - coupling stability and boundary improvements;
  - richer visualization/post-processing tools.

## Git Hygiene Contract

Confirm `.gitignore` covers generated outputs. At minimum:

```gitignore
outputs/
```

Generated `outputs/` files must not be staged or committed.

Tests must write only under `tmp_path`.

## Acceptance Commands

Final state must pass:

```powershell
python -m pip install -e ".[dev]"
pytest tests\test_output.py -q
pytest tests\test_simulation.py -q
pytest tests\test_coupled_validation.py -q
pytest -q
python examples\smoke_import.py
python examples\lbm_standalone.py
python examples\lbm_local_force.py
python examples\mpm_standalone_cube.py
python examples\coupled_penalty_smoke.py
python examples\coupled_output_snapshot.py
python -m compileall fsi tests examples
ruff check .
git diff -- third_party
Select-String -Path fsi\*.py -Pattern 'ti\.init\(|taichi\.init\('
git status --short
```

Expected:

- all tests pass;
- all examples run;
- output example creates snapshots under ignored `outputs/`;
- generated output files are not staged;
- `compileall` passes;
- `ruff check .` passes;
- `third_party` diff has no output;
- `fsi/*.py` has no `ti.init()` / `taichi.init()`.

## Implementation Order

1. Add this goal file.
2. Inspect local pyevtk API from the installed dependency.
3. Add red tests in `tests/test_output.py`.
4. Add output config tests.
5. Implement `fsi/output.py`.
6. Integrate writer into `FSISimulation`.
7. Add `FSISimulation.write_snapshot()`.
8. Extend `FSISimulation.run()` with output flags.
9. Add lazy public export for `SimulationOutputWriter`.
10. Add `examples/coupled_output_snapshot.py`.
11. Update README.
12. Run focused output tests.
13. Run full acceptance commands.
14. Commit:

```text
feat(output): add simulation snapshot writers
```

15. Push `main -> origin/main`.

## Definition Of Done

Step 7 is complete when:

- `docs/goals/STEP7_OUTPUT_SNAPSHOTS_GOAL.md` exists.
- `fsi/output.py` exists.
- `FSISimulation` owns an `output_writer`.
- `FSISimulation.write_snapshot()` writes current-step snapshots.
- `FSISimulation.run(write_output=True)` writes snapshots by interval.
- `run()` default writes no files.
- NPZ snapshots contain metadata, LBM fields, MPM particle fields, and coupling
  fields when enabled.
- Output field flags are respected.
- `output_format="npz"` works.
- `output_format="vtk"` works at smoke level.
- `output_format="both"` works.
- A coupled output example runs.
- README marks Step 7 complete.
- `pytest -q` passes.
- examples pass.
- compileall and ruff pass.
- `third_party/` remains untouched.
- `fsi/*.py` contains no `ti.init()` / `taichi.init()`.
- generated outputs are ignored and not committed.
- completed work is committed and pushed to `origin/main`.
