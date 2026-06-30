# Step 11 Reference-Data Validation Goal

## Objective

Add a reference-data validation framework for selected LBM, MPM, coupling,
output, and post-processing behaviors.

Step 11 should introduce small, committed, reviewable JSON reference datasets
and comparison utilities that can detect regressions against deterministic
baselines. This is validation infrastructure. It must not claim experimental,
paper-level, or high-fidelity external physical validation.

## Source State

The starting point is `origin/main` at Step 10:

```text
d97bac102f85dd167a9a3d77938328fab791cd4f
feat(postprocess): add snapshot and validation summary tools
```

The repository already has:

- benchmark-style validation reports in `fsi.validation`
- validation case runners in `fsi.validation_cases`
- post-processing helpers in `fsi.postprocess`
- `examples/validation_benchmark_suite.py`
- `examples/postprocess_validation_summary.py`
- documentation for post-processing in `docs/postprocessing.md`

Step 11 should build on these surfaces rather than replacing them.

## In Scope

- Define a small JSON reference-data schema.
- Add pure-Python reference-data loading and comparison helpers.
- Add deterministic reference-case metric generation functions.
- Add committed lightweight JSON reference fixtures.
- Add validation cases comparing current metrics to reference data.
- Convert reference comparison reports into existing `ValidationReport` objects.
- Add a reference validation example that writes JSON and CSV summaries.
- Add an optional generator example that refreshes reference JSON files.
- Add tests for schema loading, save/load, comparison, tolerance handling,
  reference fixtures, reference suite integration, and ValidationReport bridging.
- Add docs for reference-data validation.
- Update README and examples README.
- Keep generated run outputs ignored.
- Keep `ti.init()` out of `fsi/`.
- Keep `third_party/` unchanged.

## Out Of Scope

Do not implement:

- large binary datasets
- long CFD/FSI benchmark campaigns
- experimental validation claims
- paper-level high-fidelity validation claims
- Fluent/OpenFOAM/ParaView automation
- new solver formulas
- new coupling formulas
- advanced immersed-boundary/contact implementation
- generated `outputs/` committed to git
- changes under `third_party/`
- `ti.init()` calls inside `fsi/`

## Non-Negotiable Constraints

- `fsi/reference.py` must be pure Python and standard library only.
- `fsi/reference.py` must not import Taichi.
- `fsi/reference.py` must not construct solvers.
- `fsi/reference_cases.py` may import solver/validation code but must not call
  `ti.init()`.
- Examples that run Taichi must call `ti.init(...)` themselves.
- Tests must not regenerate committed reference fixtures in place.
- Normal validation must compare against committed references, not generate its
  own expected values dynamically.
- Reference fixtures must be small JSON files suitable for review.
- Tolerances must be explicit per metric.
- Generated validation outputs must stay under ignored `outputs/`.

## Reference Dataset Directory

Add:

```text
data/reference/
```

Required files:

```text
data/reference/README.md
data/reference/lbm_periodic_mass_reference.json
data/reference/lbm_force_response_reference.json
data/reference/mpm_gravity_reference.json
data/reference/coupled_drift_reference.json
data/reference/coupling_stability_reference.json
```

JSON is required instead of NPZ/NPY because this repository already ignores
`*.npz` and `*.npy`, and reference baselines should be readable in code review.

## Reference JSON Schema

Each reference JSON file should use schema version 1:

```json
{
  "schema_version": 1,
  "case_name": "lbm_periodic_mass_reference",
  "description": "Small periodic LBM mass conservation reference.",
  "created_by": "fsi-lbm-mpm Step 11 reference generator",
  "metadata": {
    "steps": 20,
    "grid": [8, 8, 8]
  },
  "metrics": {
    "relative_mass_error": 1.1920928955078125e-7,
    "max_velocity_norm": 0.0
  },
  "tolerances": {
    "relative_mass_error": {
      "abs": 1.0e-5,
      "rel": 0.0
    },
    "max_velocity_norm": {
      "abs": 1.0e-4,
      "rel": 0.0
    }
  }
}
```

Required top-level fields:

- `schema_version`
- `case_name`
- `description`
- `metrics`
- `tolerances`

Optional top-level fields:

- `created_by`
- `metadata`

Every metric must have an explicit tolerance entry.

## Tolerance Rule

For each metric:

```text
abs_error = abs(current_value - reference_value)
rel_error = abs_error / abs(reference_value) when reference_value != 0
rel_error = abs_error when reference_value == 0
passed = abs_error <= abs_tolerance OR rel_error <= rel_tolerance
```

Absolute tolerance should dominate near-zero quantities. Relative tolerance is
allowed for quantities with stable nonzero scale.

Initial tolerance recommendations:

- `relative_mass_error`: `abs <= 1e-5`
- `max_velocity_norm`: `abs <= 1e-4` for near-zero smoke values
- `mean_ux_growth`: `rel <= 1e-2` or `abs <= 1e-8`
- `center_of_mass_y_delta`: `rel <= 1e-2` or `abs <= 1e-6`
- `enabled_dx`: `rel <= 5e-2` or `abs <= 1e-6`
- `force_balance_norm`: `abs <= 1e-6`
- `min_particle_valid_weight`: `rel <= 1e-3` or `abs <= 1e-6`

Tolerances should be loose enough for Taichi CPU/platform variation, but tight
enough to catch obvious regressions.

## `fsi/reference.py`

Add:

```text
fsi/reference.py
```

This module should expose:

```python
@dataclass(frozen=True)
class ReferenceMetricTolerance:
    abs: float = 0.0
    rel: float = 0.0


@dataclass(frozen=True)
class ReferenceDataset:
    schema_version: int
    case_name: str
    description: str
    metrics: dict[str, float]
    tolerances: dict[str, ReferenceMetricTolerance]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_by: str = ""

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ReferenceComparison:
    case_name: str
    metric_name: str
    value: float
    reference: float
    abs_error: float
    rel_error: float
    abs_tolerance: float
    rel_tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ReferenceReport:
    case_name: str
    comparisons: tuple[ReferenceComparison, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool: ...
    def to_dict(self) -> dict[str, Any]: ...


def load_reference_dataset(path: str | Path) -> ReferenceDataset: ...
def save_reference_dataset(dataset: ReferenceDataset, path: str | Path) -> Path: ...
def compare_metrics(
    case_name: str,
    current: dict[str, float],
    reference: ReferenceDataset,
) -> ReferenceReport: ...
def reference_report_to_validation_report(report: ReferenceReport) -> ValidationReport: ...
```

### Loader Contract

`load_reference_dataset()` should:

- read UTF-8 JSON
- require top-level JSON object
- require `schema_version == 1`
- require `case_name`
- require `description`
- require non-empty `metrics`
- require a tolerance for every metric
- reject missing or nonnumeric metric values
- parse tolerance entries into `ReferenceMetricTolerance`
- tolerate missing `metadata` by using `{}`

### Save Contract

`save_reference_dataset()` should:

- write UTF-8 JSON
- use `indent=2`
- write deterministic key order where practical
- create parent directories
- return the path

### Comparison Contract

`compare_metrics()` should:

- require every reference metric to exist in `current`
- raise `KeyError` for missing current metrics
- compute abs and relative errors
- apply the tolerance rule
- return a `ReferenceReport`

### Validation Bridge Contract

`reference_report_to_validation_report()` should produce a `ValidationReport`
whose case name is:

```text
reference_<reference_case_name>
```

Each reference comparison should become a metric named:

```text
<metric_name>_abs_error
```

The metric upper bound should be the absolute tolerance and the `passed` flag
should mirror the reference comparison.

## `fsi/reference_cases.py`

Add:

```text
fsi/reference_cases.py
```

This module may use solver and validation code, but must not call `ti.init()`.

Required public API:

```python
def default_reference_data_dir() -> Path: ...
def compute_lbm_periodic_mass_reference_metrics() -> dict[str, float]: ...
def compute_lbm_force_response_reference_metrics() -> dict[str, float]: ...
def compute_mpm_gravity_reference_metrics() -> dict[str, float]: ...
def compute_coupled_drift_reference_metrics() -> dict[str, float]: ...
def compute_coupling_stability_reference_metrics() -> dict[str, float]: ...
def build_reference_datasets() -> list[ReferenceDataset]: ...
def generate_reference_datasets(output_dir: str | Path) -> list[Path]: ...
def run_reference_validation_suite(
    reference_dir: str | Path | None = None,
) -> list[ReferenceReport]: ...
```

### Reference Cases

Implement these committed reference cases:

1. `lbm_periodic_mass_reference`
   - based on the small periodic LBM mass conservation case
   - metrics:
     - `relative_mass_error`
     - `max_velocity_norm`

2. `lbm_force_response_reference`
   - based on the small forced periodic LBM case
   - metrics:
     - `mean_ux_growth`
     - `max_velocity_norm`

3. `mpm_gravity_reference`
   - based on the weak-gravity MPM response case
   - metrics:
     - `center_of_mass_y_delta`
     - `positions_finite`
     - `velocities_finite`

4. `coupled_drift_reference`
   - based on enabled vs disabled coupling drift
   - metrics:
     - `enabled_dx`
     - `disabled_dx_abs`
     - `enabled_minus_disabled_dx`

5. `coupling_stability_reference`
   - based on recent Step 9 force-limit and boundary-support validation
   - metrics:
     - `particle_force_norm`
     - `force_balance_norm`
     - `partial_support_particle_count`
     - `min_particle_valid_weight`

The first four cases may reuse existing Step 8 validation case functions. The
Step 9 stability reference may combine selected metrics from
`run_coupling_force_limit_case()` and `run_coupling_boundary_support_case()`.

### Generator Contract

`generate_reference_datasets(output_dir)` should:

- build the current reference datasets
- write them into the requested directory
- return the written paths
- never implicitly write into `data/reference` unless the caller explicitly
  passes that directory

### Validation Suite Contract

`run_reference_validation_suite(reference_dir=None)` should:

- use `data/reference` by default
- load committed references
- recompute current metrics
- compare current metrics to references
- return `list[ReferenceReport]`
- not write files

## Examples

Add:

```text
examples/reference_validation_suite.py
examples/generate_reference_data.py
```

### `examples/reference_validation_suite.py`

The example should:

- call `ti.init(...)`
- run `run_reference_validation_suite()`
- convert reports with `reference_report_to_validation_report()`
- write JSON to `outputs/reference_validation_suite/reference_validation_summary.json`
- flatten metrics to CSV using Step 10 postprocess helpers
- write `outputs/reference_validation_suite/reference_validation_summary.csv`
- print `[PASS]` or `[FAIL]` per reference case
- exit nonzero if any reference report fails

### `examples/generate_reference_data.py`

The generator example should:

- call `ti.init(...)`
- write reference JSON into `data/reference`
- print a warning that it overwrites committed reference JSON files
- print each written path
- require users to review git diff before committing refreshed references

This generator is allowed because Step 11 needs controlled fixture refreshes.
It should not run as part of normal tests unless explicitly called.

## Documentation

Add:

```text
docs/reference_validation.md
```

It should document:

- purpose of reference-data validation
- what this workflow is
- what this workflow is not
- JSON schema
- tolerance policy
- how to run reference validation
- how to regenerate references
- how to review reference fixture diffs
- how to add a new reference case

The documentation must avoid claiming experimental or high-fidelity physical
validation.

## README Updates

Update README current status to:

```text
Step 11: reference-data validation framework.
```

Add implemented item:

```text
reference-data validation framework and small committed reference cases
```

Update not-implemented items:

- high-fidelity external experimental/reference validation
- advanced immersed-boundary/contact handling
- interactive or production rendering workflow

Add development command:

```bash
python examples/reference_validation_suite.py
```

Add docs link:

```text
docs/reference_validation.md
```

Update next milestones:

- explore advanced immersed-boundary/contact handling
- add interactive or production rendering workflow
- add larger external reference-data campaigns

## Examples README Updates

Update `examples/README.md` to include:

- reference-data validation against committed JSON baselines
- `reference_validation_suite.py`
- `generate_reference_data.py`

## Public Exports

Update `fsi/__init__.py` lazy exports for selected reference APIs:

- `ReferenceDataset`
- `ReferenceReport`
- `load_reference_dataset`
- `run_reference_validation_suite`

Do not eagerly import `fsi.reference` or `fsi.reference_cases` at package import
time.

## Tests

Add:

```text
tests/test_reference_validation.py
```

Required test coverage:

1. load/save reference dataset round trip
2. invalid schema version rejected
3. missing tolerance rejected
4. comparison pass and fail behavior
5. missing current metric raises `KeyError`
6. reference report serialization
7. committed reference files load and have tolerance coverage
8. generated reference data can be written to `tmp_path`
9. reference validation suite returns passing reports
10. reference reports convert to existing `ValidationReport`
11. converted validation reports can be flattened by Step 10

Update:

```text
tests/test_import.py
```

to verify the new public export names are in `fsi.__all__`.

## Acceptance Commands

Use the trusted local Taichi environment:

```powershell
D:\working\taichi\env\python.exe -m pip install -e ".[dev]"

D:\working\taichi\env\python.exe -m pytest tests\test_reference_validation.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_validation_benchmarks.py -q
D:\working\taichi\env\python.exe -m pytest tests\test_postprocess.py -q
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
D:\working\taichi\env\python.exe examples\reference_validation_suite.py

D:\working\taichi\env\python.exe -m compileall fsi tests examples
ruff check .

git diff -- third_party
Select-String -Path fsi\*.py -Pattern 'ti\.init\(|taichi\.init\('
git status --short --ignored outputs
```

Expected outcomes:

- reference validation tests pass
- committed reference files load
- reference validation suite passes
- postprocess and validation regressions still pass
- fast suite passes
- full suite passes
- examples run
- reference validation JSON/CSV outputs are generated under ignored `outputs/`
- compileall passes
- ruff passes
- `third_party` diff is empty
- `fsi/*.py` has no `ti.init()` or `taichi.init()`
- generated outputs remain ignored and unstaged

If the pre-push hook uses the wrong Python, push with the trusted Taichi
environment first in `PATH` and one-off pytest temp/cache directories rather
than bypassing the hook.

## Commit And Push

After implementation and verification:

```text
feat(validation): add reference-data validation framework
```

Push:

```text
main -> origin/main
```

## Definition Of Done

Step 11 is complete only when:

- `docs/goals/STEP11_REFERENCE_DATA_VALIDATION_GOAL.md` exists.
- `docs/reference_validation.md` exists.
- `data/reference/README.md` exists.
- committed reference JSON fixtures exist under `data/reference/`.
- `fsi/reference.py` exists.
- `fsi/reference_cases.py` exists.
- reference datasets can be loaded and saved.
- missing tolerances are rejected.
- current metrics can be compared to reference metrics.
- `ReferenceReport` can be produced and serialized.
- `ReferenceReport` can be converted to existing `ValidationReport`.
- reference validation suite runs and passes against committed fixtures.
- `examples/reference_validation_suite.py` exists and runs.
- `examples/generate_reference_data.py` exists and runs.
- docs and README describe the workflow without overstating physical validation.
- `tests/test_reference_validation.py` covers the new module and fixtures.
- `tests/test_import.py` covers the public export names.
- acceptance commands pass, or any unavoidable local environment exception is
  recorded with exact command and error.
- no solver/coupling numerical formulas are changed.
- no `third_party` files are changed.
- no `ti.init()` or `taichi.init()` calls are introduced under `fsi/`.
- generated outputs are ignored and not staged.
- the final commit is pushed to `origin/main`.
