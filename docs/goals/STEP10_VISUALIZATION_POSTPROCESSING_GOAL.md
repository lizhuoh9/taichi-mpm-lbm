# Step 10 Visualization And Post-Processing Goal

## Objective

Add lightweight visualization and post-processing tools for existing simulation
outputs and validation summaries.

Step 10 should turn Step 7 NPZ/VTK snapshots, Step 8 validation JSON, and Step 9
coupling diagnostics into readable, reproducible, comparable analysis artifacts:

- snapshot listings
- snapshot metadata summaries
- diagnostic time series
- CSV summaries
- JSON summaries
- basic PNG plots
- validation metric tables
- VTK/ParaView usage documentation

Step 10 must not change solver physics, coupling formulas, Taichi kernels, or
third-party code. It is a post-processing workflow step.

## Source State

The starting point is `origin/main` at Step 9:

```text
2d6ddc3a93ec2cdcf0eb6744f7da22c5c2d871b3
feat(coupling): add stability guards and diagnostics
```

The repository already has:

- `SimulationOutputWriter.write_npz_snapshot()`
- `SimulationOutputWriter.write_vtk_snapshot()`
- NPZ metadata keys:
  - `step`
  - `time`
  - `lbm_shape`
  - `mpm_particle_count`
- optional NPZ LBM keys:
  - `lbm_density`
  - `lbm_velocity`
  - `lbm_force`
  - `lbm_solid`
- optional NPZ MPM keys:
  - `mpm_positions`
  - `mpm_velocities`
  - `mpm_particle_forces`
  - `mpm_deformation_gradients`
  - `mpm_active`
- optional NPZ coupling keys:
  - `coupling_force`
  - `solid_volume_fraction`
  - `total_particle_coupling_force`
  - `total_fluid_coupling_force`
  - `coupling_particle_valid_weight`
  - `coupling_particle_mask`
  - `coupling_unsupported_particle_count`
  - `coupling_partial_support_particle_count`
  - `coupling_clipped_particle_count`
- validation report objects from `fsi.validation`
- `examples/validation_benchmark_suite.py`, which writes
  `outputs/validation_benchmark_suite/validation_summary.json`

## In Scope

Implement a pure-Python post-processing layer:

- Add `fsi/postprocess.py`.
- Add a frozen `SnapshotInfo` dataclass.
- List `snapshot_*.npz` files sorted by numeric step.
- Load NPZ snapshots into normal `dict[str, np.ndarray]` objects.
- Read metadata from one NPZ snapshot.
- Summarize all snapshots in an output directory.
- Extract robust diagnostic time series from all available snapshots.
- Extract Step 9 coupling diagnostic time series when present.
- Write time series to CSV without pandas.
- Write time series to JSON with NumPy-safe conversion.
- Load validation summary JSON.
- Flatten validation reports into metric table rows.
- Generate a basic PNG line plot from a time series.
- Import plotting dependencies lazily inside plotting functions.
- Add tests using `tmp_path`.
- Add examples that generate small outputs under ignored `outputs/`.
- Add documentation for NPZ, CSV/JSON, PNG, and VTK/ParaView usage.
- Update README to mark Step 10 complete.
- Update `examples/README.md`.
- Add selected postprocess helpers to lazy public exports in `fsi.__init__`.

## Out Of Scope

Do not implement:

- interactive GUI
- heavy rendering pipeline
- video generation
- animation pipeline
- ParaView automation
- high-fidelity external-reference validation
- new benchmark physics
- new solver formulas
- new coupling formulas
- MPM, LBM, or coupling kernel redesign
- changes under `third_party/`
- generated outputs committed to git

## Non-Negotiable Constraints

- `fsi/postprocess.py` must be pure Python plus NumPy and standard library.
- `fsi/postprocess.py` must not call `ti.init()`.
- `fsi/postprocess.py` must not import Taichi.
- `fsi/postprocess.py` must not construct solvers.
- `fsi/postprocess.py` must not depend on pandas.
- Matplotlib, if used, must be imported lazily inside plotting functions.
- Normal `import fsi` must stay lightweight and reliable.
- Snapshot time-series extraction must tolerate missing optional field groups.
- Tests must not require ParaView.
- Tests must not commit or depend on committed generated outputs.
- Generated files must stay under ignored `outputs/` or pytest `tmp_path`.

## Dependency Decision

Add `matplotlib>=3.7.0` to the `dev` optional dependency group because Step 10
explicitly includes basic PNG plot generation.

`matplotlib` must remain a development/analysis dependency, not a core import
dependency:

```python
def plot_timeseries(...):
    import matplotlib.pyplot as plt
```

This preserves cheap imports for core solver users and allows `pip install -e
".[dev]"` to install plotting support for examples and tests.

## Public Module

Add:

```text
fsi/postprocess.py
```

The module should expose these public APIs:

```python
@dataclass(frozen=True)
class SnapshotInfo:
    path: Path
    step: int
    time: float
    particle_count: int
    lbm_shape: tuple[int, int, int]

    def to_dict(self) -> dict[str, Any]: ...


def list_npz_snapshots(output_dir: str | Path) -> list[Path]: ...
def load_npz_snapshot(path: str | Path) -> dict[str, np.ndarray]: ...
def snapshot_info(path: str | Path) -> SnapshotInfo: ...
def summarize_snapshots(output_dir: str | Path) -> list[SnapshotInfo]: ...
def extract_snapshot_timeseries(output_dir: str | Path) -> dict[str, np.ndarray]: ...
def summarize_coupling_diagnostics(output_dir: str | Path) -> dict[str, np.ndarray]: ...
def write_timeseries_csv(timeseries: dict[str, np.ndarray], path: str | Path) -> Path: ...
def write_timeseries_json(timeseries: dict[str, np.ndarray], path: str | Path) -> Path: ...
def load_validation_summary(path: str | Path) -> list[dict[str, Any]]: ...
def validation_summary_table(reports: list[dict[str, Any]]) -> list[dict[str, Any]]: ...
def write_validation_summary_csv(rows: list[dict[str, Any]], path: str | Path) -> Path: ...
def plot_timeseries(
    timeseries: dict[str, np.ndarray],
    y_key: str,
    output_path: str | Path,
    *,
    x_key: str = "step",
) -> Path: ...
```

Adding `write_validation_summary_csv()` is allowed because the examples need a
small, dependency-free way to write flattened validation rows.

## Snapshot Ordering Contract

`list_npz_snapshots()` should:

- accept `str | Path`
- return `Path` objects
- match files named `snapshot_*.npz`
- sort files by numeric step when possible
- fall back to lexicographic ordering only for malformed names
- return an empty list for a missing or empty directory

Expected sorted example:

```text
snapshot_000000.npz
snapshot_000002.npz
snapshot_000010.npz
```

## Snapshot Loading Contract

`load_npz_snapshot()` should:

- accept `str | Path`
- raise `FileNotFoundError` for missing files
- use `np.load(..., allow_pickle=False)`
- return a plain dictionary
- eagerly copy arrays out of the `NpzFile` context so the result remains valid
  after the file is closed

## Snapshot Metadata Contract

`snapshot_info()` should read these required keys:

- `step`
- `time`
- `lbm_shape`
- `mpm_particle_count`

It should return:

```python
SnapshotInfo(
    path=Path(...),
    step=int(...),
    time=float(...),
    particle_count=int(...),
    lbm_shape=(nx, ny, nz),
)
```

If a required metadata key is missing, raising `KeyError` is acceptable.

## Time Series Contract

`extract_snapshot_timeseries()` should always include available metadata:

- `step`
- `time`
- `mpm_particle_count`

It should include derived LBM fields when source arrays exist:

- `lbm_mean_density`
- `lbm_total_mass_estimate`
- `lbm_max_velocity_norm`

It should include derived MPM fields when source arrays exist:

- `mpm_center_of_mass_x`
- `mpm_center_of_mass_y`
- `mpm_center_of_mass_z`
- `mpm_max_velocity_norm`

It should include derived coupling fields when source arrays exist:

- `coupling_force_norm`
- `total_particle_coupling_force_norm`
- `total_fluid_coupling_force_norm`
- `coupling_unsupported_particle_count`
- `coupling_partial_support_particle_count`
- `coupling_clipped_particle_count`

The function should not fail just because a field group is absent. For example,
snapshots created with:

```python
OutputConfig(write_lbm_fields=False, write_coupling_fields=False)
```

should still produce a usable time series containing metadata and any MPM
derived fields that are present.

## Coupling Diagnostics Contract

`summarize_coupling_diagnostics()` should extract the Step 9 coupling diagnostic
keys from `extract_snapshot_timeseries()`.

At minimum, when present, include:

- `step`
- `time`
- `coupling_force_norm`
- `total_particle_coupling_force_norm`
- `total_fluid_coupling_force_norm`
- `coupling_unsupported_particle_count`
- `coupling_partial_support_particle_count`
- `coupling_clipped_particle_count`

If no coupling diagnostic keys are present, return at least:

- `step`
- `time`

This keeps downstream code robust when `write_coupling_fields=False`.

## CSV Contract

`write_timeseries_csv()` should:

- require a non-empty time series
- validate all arrays have the same first dimension length
- write UTF-8 text
- use stdlib `csv`
- avoid pandas
- write one row per time sample
- include all keys as columns
- keep stable column ordering with `step` and `time` first when present
- return the output path

For scalar NumPy values, convert to native Python scalars before writing.

## JSON Contract

`write_timeseries_json()` should:

- write UTF-8 JSON
- use `indent=2`
- convert NumPy arrays to lists
- convert NumPy scalar types to native scalar values
- return the output path

## Validation Summary Contract

`load_validation_summary()` should:

- accept the JSON path from `examples/validation_benchmark_suite.py`
- return `list[dict[str, Any]]`
- validate that the JSON top-level value is a list
- raise `ValueError` for non-list top-level JSON

`validation_summary_table()` should flatten reports like:

```python
{
    "case_name": "coupling_force_limit",
    "passed": True,
    "metrics": [
        {
            "name": "force_balance_norm",
            "value": 1.2e-8,
            "lower": None,
            "upper": 1e-6,
            "passed": True,
            "units": "",
            "description": "...",
        }
    ],
    "metadata": {...},
}
```

into rows like:

```python
{
    "case_name": "coupling_force_limit",
    "case_passed": True,
    "metric_name": "force_balance_norm",
    "value": 1.2e-8,
    "lower": None,
    "upper": 1e-6,
    "metric_passed": True,
    "units": "",
    "description": "...",
}
```

`write_validation_summary_csv()` should write these rows with a stable header.

## Plotting Contract

`plot_timeseries()` should:

- import matplotlib lazily inside the function
- accept `timeseries`, `y_key`, `output_path`, and optional `x_key`
- validate `x_key` and `y_key` exist
- write a PNG file
- create parent directories if needed
- close the matplotlib figure after saving
- return the output path

Tests should verify that the PNG exists and is non-empty.

## Public Exports

Update `fsi/__init__.py` lazy exports for selected public APIs:

- `SnapshotInfo`
- `list_npz_snapshots`
- `extract_snapshot_timeseries`
- `load_validation_summary`
- `validation_summary_table`

Optional additional exports are acceptable if they stay focused on user-facing
post-processing.

Do not eagerly import `fsi.postprocess` at package import time.

## Examples

Add:

```text
examples/postprocess_snapshots.py
examples/postprocess_validation_summary.py
```

### `examples/postprocess_snapshots.py`

The example should:

- call `ti.init(...)` in the example, not in `fsi/`
- create a tiny coupled simulation
- write NPZ snapshots to `outputs/postprocess_snapshots`
- use `output_interval=2`
- write initial output and four simulation steps
- produce:
  - `snapshot_000000.npz`
  - `snapshot_000002.npz`
  - `snapshot_000004.npz`
  - `timeseries.csv`
  - `timeseries.json`
  - at least one PNG plot
- print a short summary
- exit nonzero if expected output generation fails

### `examples/postprocess_validation_summary.py`

The example should:

- generate a fresh validation summary through the existing validation suite
  entry point
- allow the validation suite entry point to own `ti.init(...)`
- write `outputs/validation_benchmark_suite/validation_summary.json`
- read that JSON through `load_validation_summary()`
- flatten metrics with `validation_summary_table()`
- write `validation_summary.csv`
- print failed metrics if any
- exit nonzero if any validation report fails

Running the existing validation example in a subprocess is acceptable. This
keeps Step 10 post-processing pure and avoids mixing a second orchestration
layer into the same Taichi process.

## Documentation

Add:

```text
docs/postprocessing.md
```

It should document:

- NPZ snapshot purpose
- VTK snapshot purpose
- how to list snapshots
- how to extract time series
- how to write CSV/JSON
- how to generate a PNG plot
- how to post-process validation summaries
- how to open VTK files in ParaView
- limitations of this workflow

The limitations section must say:

- post-processing is diagnostic, not high-fidelity validation
- VTK support is file output only, not ParaView automation
- no interactive rendering or video generation is included

## README Updates

Update README current status to:

```text
Step 10: visualization and post-processing workflow.
```

Add implemented item:

```text
visualization and post-processing utilities for snapshots and validation summaries
```

Update not-implemented items to leave out generic post-processing but still list:

- high-fidelity validation against external reference data
- advanced immersed-boundary/contact handling
- interactive or production rendering workflow

Add development commands:

```bash
python examples/postprocess_snapshots.py
python examples/postprocess_validation_summary.py
```

Add a link to:

```text
docs/postprocessing.md
```

Update next milestones to:

- add higher-fidelity reference-data validation cases
- explore advanced immersed-boundary/contact handling
- add interactive or production rendering workflow

## Examples README Updates

Update `examples/README.md` to include:

- post-processing snapshots into CSV/JSON/PNG summaries
- post-processing validation reports
- `postprocess_snapshots.py`
- `postprocess_validation_summary.py`

## Tests

Add:

```text
tests/test_postprocess.py
```

Required test coverage:

1. `list_npz_snapshots()` sorts by numeric step.
2. `load_npz_snapshot()` returns a normal dictionary with arrays.
3. `snapshot_info()` reads metadata correctly.
4. `summarize_snapshots()` returns `SnapshotInfo` objects.
5. `extract_snapshot_timeseries()` extracts metadata and derived fields from
   multiple real snapshots.
6. `extract_snapshot_timeseries()` remains robust when optional field groups are
   disabled.
7. `summarize_coupling_diagnostics()` extracts Step 9 coupling count series when
   present.
8. `write_timeseries_csv()` writes a readable CSV with stable key ordering.
9. `write_timeseries_json()` writes JSON-safe values.
10. `load_validation_summary()` loads list JSON and rejects non-list JSON.
11. `validation_summary_table()` flattens validation metrics.
12. `write_validation_summary_csv()` writes flattened metric rows.
13. `plot_timeseries()` writes a non-empty PNG.

Update:

```text
tests/test_import.py
```

to cover selected public postprocess exports.

## Acceptance Commands

Use the trusted local Taichi environment:

```powershell
D:\working\taichi\env\python.exe -m pip install -e ".[dev]"

D:\working\taichi\env\python.exe -m pytest tests\test_postprocess.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_output.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_validation_benchmarks.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_import.py -q
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

D:\working\taichi\env\python.exe -m compileall fsi tests examples
ruff check .

git diff -- third_party
Select-String -Path fsi\*.py -Pattern 'ti\.init\(|taichi\.init\('
git status --short --ignored outputs
```

Expected outcomes:

- postprocess tests pass
- output and validation regression tests pass
- import test passes
- fast suite passes
- full suite passes
- all examples run
- CSV/JSON/PNG artifacts are generated under ignored `outputs/`
- compileall passes
- ruff passes
- `third_party` diff is empty
- `fsi/*.py` has no `ti.init()` or `taichi.init()` calls
- generated outputs remain ignored and unstaged

## Commit And Push

After implementation and verification:

```text
feat(postprocess): add snapshot and validation summary tools
```

Push:

```text
main -> origin/main
```

If the pre-push hook reruns pytest and hits Windows temp/cache cleanup problems,
use a one-off local temp/cache environment for that push attempt rather than
committing generated temp files.

## Definition Of Done

Step 10 is complete only when:

- `docs/goals/STEP10_VISUALIZATION_POSTPROCESSING_GOAL.md` exists.
- `fsi/postprocess.py` exists.
- NPZ snapshots can be listed.
- NPZ snapshots can be loaded.
- snapshot metadata can be inspected.
- snapshot time series can be extracted.
- Step 9 coupling diagnostics can be summarized.
- CSV summaries can be written.
- JSON summaries can be written.
- `validation_summary.json` can be loaded.
- validation metrics can be flattened.
- validation metric CSV can be written.
- basic PNG plots can be generated.
- `examples/postprocess_snapshots.py` exists and runs.
- `examples/postprocess_validation_summary.py` exists and runs.
- `docs/postprocessing.md` exists.
- README marks Step 10 complete.
- `examples/README.md` includes the new examples.
- `tests/test_postprocess.py` covers the new module.
- public exports are covered by `tests/test_import.py`.
- full acceptance commands pass or any unavoidable local exception is recorded
  with exact command and error.
- no solver/coupling numerical formulas are changed.
- no `third_party` files are changed.
- no `ti.init()` or `taichi.init()` calls are introduced under `fsi/`.
- generated outputs are ignored and not staged.
- the final commit is pushed to `origin/main`.
