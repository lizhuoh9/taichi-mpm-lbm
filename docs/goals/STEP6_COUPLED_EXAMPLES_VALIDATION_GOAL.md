# Step 6 Coupled Examples And Validation Goal

## Objective

Implement a runnable top-level coupled simulation wrapper and add lightweight
coupled examples and validation tests.

Step 6 turns the Step 5 coupling engine into a usable workflow:

```text
SimulationConfig
  -> FSISimulation
      -> LBMSolver3D
      -> MPMSolver3D
      -> LBMMpmCoupler
```

The goal is not to claim high-fidelity FSI validation. The goal is to provide a
small, explicit, tested simulation shell that initializes the solvers, advances
coupled steps, returns diagnostics, and runs a CPU smoke example.

## In Scope

- Implement `fsi.simulation.FSISimulation`.
- Keep `FSISimulation` free of `ti.init()`; callers and examples initialize
  Taichi.
- Instantiate `LBMSolver3D`, `MPMSolver3D`, and `LBMMpmCoupler` from
  `SimulationConfig`.
- Provide explicit initialization methods for LBM and MPM.
- Provide a box-based convenience initializer for examples.
- Provide `step()`, `run()`, and `diagnostics()` methods.
- Return lightweight diagnostics, not full field arrays, from `step()` and
  `run()`.
- Export the public solver/coupler/simulation classes from `fsi.__init__`.
- Add a small `examples/coupled_penalty_smoke.py` CPU example.
- Add focused tests for simulation initialization, stepping, run history,
  validation errors, enabled coupling motion, disabled coupling behavior,
  finite coupled fields, and bounded solid volume fraction.
- Update README to mark Step 6 complete and list next milestones.
- Keep existing Step 5 coupling tests passing.

## Out Of Scope

Do not implement any of the following in Step 6:

- High-fidelity physical benchmark claims.
- Long-running validation campaigns.
- VTK output or heavy visualization.
- Default output-file writing in normal runs.
- Sparse grids.
- LBM solid-mask mutation from MPM particles.
- Immersed-boundary no-slip enforcement.
- CPIC, rigid contact, fracture, plasticity, or multimaterial contact.
- Changes to Step 5 coupling formulas unless tests expose a clear bug.
- Edits under `third_party/`.
- `ti.init()` or `taichi.init()` inside `fsi/` library modules.

## Simulation API Contract

`fsi/simulation.py` must expose:

```python
class FSISimulation:
    def __init__(self, config: SimulationConfig) -> None:
        ...

    def initialize_lbm(self, solid_np: np.ndarray | None = None) -> None:
        ...

    def initialize_mpm_box(
        self,
        lower: tuple[float, float, float],
        upper: tuple[float, float, float],
        spacing: float | tuple[float, float, float] | None = None,
        initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> int:
        ...

    def initialize_mpm_from_numpy(
        self,
        positions: np.ndarray,
        velocities: np.ndarray | None = None,
        particle_mass: float | None = None,
        particle_volume: float | None = None,
    ) -> int:
        ...

    def initialize(
        self,
        solid_np: np.ndarray | None = None,
        mpm_box_lower: tuple[float, float, float] | None = None,
        mpm_box_upper: tuple[float, float, float] | None = None,
        mpm_box_spacing: float | tuple[float, float, float] | None = None,
        mpm_initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> int:
        ...

    def step(self) -> dict[str, object]:
        ...

    def run(self, steps: int | None = None) -> list[dict[str, object]]:
        ...

    def diagnostics(self) -> dict[str, object]:
        ...
```

The constructor must:

- validate `SimulationConfig`;
- instantiate `LBMSolver3D(config.lbm)`;
- instantiate `MPMSolver3D(config.mpm)`;
- instantiate `LBMMpmCoupler(config.coupling, self.lbm, self.mpm)`;
- initialize `step_index = 0`;
- initialize `time = 0.0`;
- initialize internal LBM/MPM state flags.

The constructor must not initialize particle positions, create a default solid
mask, or call `ti.init()`.

## Initialization Contract

`initialize_lbm(solid_np=None)` must delegate to:

```python
self.lbm.initialize(solid_np=solid_np)
```

and mark LBM initialized.

`initialize_mpm_box(...)` must delegate to:

```python
self.mpm.initialize_particles_box(...)
```

and mark MPM initialized.

`initialize_mpm_from_numpy(...)` must delegate to:

```python
self.mpm.initialize_particles_from_numpy(...)
```

and mark MPM initialized.

`initialize(...)` is a convenience method for examples and smoke tests. It must:

- initialize LBM;
- require both `mpm_box_lower` and `mpm_box_upper`;
- raise `ValueError` if either bound is missing;
- initialize MPM with the supplied box;
- return the initialized particle count.

Do not add particle arrays or box defaults to `SimulationConfig`. The explicit
initialization methods are the right place for user-provided geometry.

## Stepping Contract

`step()` must:

- require LBM and MPM initialization;
- call `self.coupler.step(self.config.lbm_dt)`;
- increment `step_index` by 1;
- increment `time` by `config.lbm_dt`;
- return `diagnostics()`.

`run(steps=None)` must:

- require initialization;
- use `config.num_steps` when `steps` is omitted;
- accept a positive override `steps`;
- reject non-positive `steps` with `ValueError`;
- return a list of per-step diagnostics.

`run()` must not write files by default.

## Diagnostics Contract

`diagnostics()` must return a small dictionary:

```python
{
    "step": int,
    "time": float,
    "lbm_total_mass": float,
    "lbm_max_velocity_norm": float,
    "mpm_particle_count": int,
    "mpm_center_of_mass": np.ndarray,
    "mpm_max_velocity_norm": float,
    "total_particle_coupling_force": np.ndarray,
    "total_fluid_coupling_force": np.ndarray,
}
```

Diagnostics may use existing NumPy helper methods. Diagnostics must not be used
inside Taichi hot kernels.

## Public API Contract

Update `fsi/__init__.py` to export:

- `LBMSolver3D`;
- `MPMSolver3D`;
- `LBMMpmCoupler`;
- `FSISimulation`.

Existing config exports must remain.

## Example Contract

Add:

```text
examples/coupled_penalty_smoke.py
```

The example must:

- call `ti.init()` in the script;
- use a small periodic CPU case;
- create a `SimulationConfig`;
- initialize a small MPM particle box;
- run a bounded number of coupled steps;
- print particle count, time, LBM mass, max fluid speed, MPM center of mass,
  and max particle speed;
- not write output files;
- keep runtime appropriate for local smoke testing.

The example is a smoke/stability demonstration, not a benchmark.

## Test Plan

Add `tests/test_simulation.py` for wrapper behavior:

1. `FSISimulation` constructs solvers and coupler from config.
2. `initialize()` initializes both solvers and returns particle count.
3. `step()` advances `step_index` and `time`, and returns diagnostics.
4. `run()` uses `config.num_steps`.
5. `run(steps=...)` accepts an override.
6. `run()` before initialization raises `RuntimeError`.
7. `initialize()` without MPM box bounds raises `ValueError`.
8. `run(steps<=0)` raises `ValueError`.

Add `tests/test_coupled_validation.py` for light coupled validation:

1. Enabled coupling moves particles in the positive flow direction.
2. Disabled coupling does not push particles without gravity.
3. LBM fields, MPM fields, and coupling fields remain finite over several
   steps.
4. `solid_volume_fraction` stays in `[0, 1]`.
5. LBM total mass remains finite and reasonably stable for the small periodic
   smoke case.

Update `tests/test_config.py`:

- remove the expectation that `FSISimulation.run()` raises
  `NotImplementedError`;
- replace it with a construction/config smoke assertion.

Existing tests must continue to pass.

## README Contract

Update README:

- current status becomes Step 6 coupled examples and validation tests;
- implemented list includes:
  - top-level coupled simulation runner;
  - lightweight coupled example and validation tests;
- not implemented list becomes:
  - high-fidelity validation benchmarks;
  - visualization/output pipeline;
- development commands include:

```bash
python examples/coupled_penalty_smoke.py
```

- roadmap item 6 is checked;
- add a short "Next milestones" section for output snapshots, benchmark-style
  validation, and coupling stability/boundary improvements.

## Acceptance Commands

Final state must pass:

```powershell
python -m pip install -e ".[dev]"
pytest tests\test_simulation.py -q
pytest tests\test_coupled_validation.py -q
pytest tests\test_coupling.py -q
pytest -q
python examples\smoke_import.py
python examples\lbm_standalone.py
python examples\lbm_local_force.py
python examples\mpm_standalone_cube.py
python examples\coupled_penalty_smoke.py
python -m compileall fsi tests examples
ruff check .
git diff -- third_party
Select-String -Path fsi\*.py -Pattern 'ti\.init\(|taichi\.init\('
```

Expected:

- all tests pass;
- all examples run successfully;
- `compileall` passes;
- `ruff check .` passes;
- `git diff -- third_party` has no output;
- the `ti.init` scan has no output.

## Implementation Order

1. Add this goal file.
2. Add red tests in `tests/test_simulation.py`.
3. Add red tests in `tests/test_coupled_validation.py`.
4. Update `tests/test_config.py` expected placeholder behavior.
5. Implement `FSISimulation`.
6. Export solver/coupler/simulation classes from `fsi.__init__`.
7. Add `examples/coupled_penalty_smoke.py`.
8. Update README.
9. Run focused tests.
10. Run full acceptance commands.
11. Commit:

```text
feat(simulation): add coupled runner and validation examples
```

12. Push `main -> origin/main`.

## Definition Of Done

Step 6 is complete when:

- `FSISimulation.run()` no longer raises `NotImplementedError`;
- `FSISimulation` can initialize LBM, MPM, and coupler;
- `FSISimulation.step()` advances one coupled LBM step;
- `FSISimulation.run()` advances multiple steps and returns diagnostics history;
- a coupled example runs independently;
- enabled coupling moves particles with fluid flow in a smoke case;
- disabled coupling does not move particles without gravity;
- LBM, MPM, and coupling diagnostic fields remain finite;
- README marks Step 6 complete;
- `third_party/` remains untouched;
- `fsi/*.py` contains no `ti.init()` or `taichi.init()`;
- full tests and acceptance commands pass;
- the completed work is committed and pushed to `origin/main`.
