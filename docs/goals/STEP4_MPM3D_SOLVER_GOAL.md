# Step 4 MPM3D Solver Goal

## Objective

Implement a standalone dense 3D elastic MLS-MPM solver in
`fsi.mpm3d.MPMSolver3D`.

Step 4 gives the project an independently runnable MPM side. It must not
implement LBM-MPM coupling, force exchange, particle-to-grid coupling from LBM,
or any real behavior in `LBMMpmCoupler`.

## In Scope

- Dense 3D MPM grid.
- Elastic material only.
- MLS-MPM / APIC particle-grid transfer.
- Fixed-corotated elastic stress.
- Box particle initialization in grid/lattice coordinates.
- NumPy particle initialization for tests and future import paths.
- Gravity and simple domain boundary handling.
- Lightweight diagnostics and a standalone example.
- Smoke tests for initialization, stability, gravity response, and domain bounds.

## Out Of Scope

Do not implement any of the following in Step 4:

- LBM-MPM coupling.
- `LBMMpmCoupler` behavior.
- MPM force scatter to LBM.
- MPM reading LBM velocity.
- Sparse grid.
- Plasticity, snow, sand, fluid, or multimaterial contact.
- CPIC, cutting, rigid bodies, or fracture.
- Heavy visualization or VTK output.
- Edits under `third_party/`.
- `ti.init()` inside `fsi/` library modules.
- `to_numpy()` or `from_numpy()` inside `substep()`, `p2g()`, `grid_op()`, or
  `g2p()`.

## Configuration Contract

Extend `MPMConfig` only as needed:

- `boundary_width: int = 3`
- `boundary_damping: float = 0.0`

Validation must enforce:

- `boundary_width >= 1`;
- `boundary_damping` in `[0, 1]`.

Existing `MPMConfig` fields remain the core controls:

- grid dimensions;
- `max_particles`;
- `particles_per_cell`;
- `dx`;
- `dt`;
- density;
- Young's modulus;
- Poisson ratio;
- gravity;
- APIC toggle;
- material selector.

## Solver API

`fsi/mpm3d.py` must expose `@ti.data_oriented class MPMSolver3D` with:

- `__init__(config: MPMConfig)`;
- `initialize_particles_box(lower, upper, spacing=None, initial_velocity=(0, 0, 0)) -> int`;
- `initialize_particles_from_numpy(positions, velocities=None, particle_mass=None, particle_volume=None) -> int`;
- `substep(dt=None) -> None`;
- `clear_grid()`;
- `p2g()`;
- `grid_op()`;
- `g2p()`;
- `clear_particle_force()`;
- diagnostics:
  - `particle_count()`;
  - `positions_numpy()`;
  - `velocities_numpy()`;
  - `deformation_gradients_numpy()`;
  - `active_numpy()`;
  - `center_of_mass()`;
  - `max_velocity_norm()`.

`substep()` must require initialized particles and must default to `config.dt`
when `dt` is omitted. It must reject non-positive `dt`.

The timestep order is:

```text
clear_grid()
p2g()
grid_op()
g2p()
```

No LBM solver or coupler should be called from MPM.

## Field Contract

Use fixed-size dense particle fields:

- `num_particles: ti.field(ti.i32, shape=())`;
- `x: ti.Vector.field(3, ti.f32, shape=max_particles)`;
- `v: ti.Vector.field(3, ti.f32, shape=max_particles)`;
- `C: ti.Matrix.field(3, 3, ti.f32, shape=max_particles)`;
- `F: ti.Matrix.field(3, 3, ti.f32, shape=max_particles)`;
- `mass: ti.field(ti.f32, shape=max_particles)`;
- `volume: ti.field(ti.f32, shape=max_particles)`;
- `active: ti.field(ti.i32, shape=max_particles)`;
- `particle_force: ti.Vector.field(3, ti.f32, shape=max_particles)`;
- `grid_v: ti.Vector.field(3, ti.f32, shape=(nx, ny, nz))`;
- `grid_m: ti.field(ti.f32, shape=(nx, ny, nz))`.

`particle_force` is only a generic external particle-force entry point for later
steps. Step 4 must not connect it to LBM.

## Numerical Contract

Implement a small, stable MLS-MPM/APIC solver:

- coordinates are grid/lattice coordinates, not normalized `[0, 1]`;
- quadratic 3x3x3 B-spline weights;
- fixed-corotated elastic stress;
- grid mass normalization;
- gravity in `grid_op()`;
- simple separating/sticky domain boundary;
- `g2p()` updates particle velocity, affine/APIC `C`, position, and deformation
  gradient;
- particle positions are clamped inside the simulation domain.

The first implementation should prioritize stable standalone smoke behavior over
high-fidelity material calibration.

## Tests

Add `tests/test_mpm3d.py` covering:

- box particle initialization;
- too many particles raising `ValueError`;
- `substep()` before initialization raising `RuntimeError`;
- zero-gravity center of mass remaining approximately stable;
- gravity moving center of mass downward;
- multiple substeps producing finite positions, velocities, and deformation
  gradients;
- particles remaining inside the domain.

Update `tests/test_config.py`:

- remove MPM placeholder `NotImplementedError` assertions;
- keep `LBMMpmCoupler` and `FSISimulation` placeholder assertions.

Existing LBM tests must continue to pass.

## Example

Add `examples/mpm_standalone_cube.py`:

- call `ti.init()` in the script;
- create a `32x32x32` MPM config;
- initialize a cube of particles;
- run 100 substeps;
- print particle count, center of mass, and max velocity every 20 steps;
- do not write output files.

## Documentation

Update `README.md`:

- current status becomes Step 4 standalone 3D MLS-MPM solver;
- implemented list includes dense standalone 3D elastic MLS-MPM;
- not implemented list keeps LBM-MPM coupling;
- development commands include `python examples/mpm_standalone_cube.py`;
- roadmap item 4 is checked.

## Acceptance Commands

The final state must pass:

```bash
python -m pip install -e ".[dev]"
pytest -q
python examples/smoke_import.py
python examples/lbm_standalone.py
python examples/lbm_local_force.py
python examples/mpm_standalone_cube.py
python -m compileall fsi tests examples
ruff check .
git diff -- third_party
```

Additional check:

```powershell
Select-String -Path fsi/*.py -Pattern 'ti.init'
```

The `ti.init` scan must have no output.

## Completion And Push Contract

After implementation and verification:

1. Review the diff and confirm `third_party/` is untouched.
2. Commit with a conventional commit message.
3. Push the completed work to `origin/main` for `lizhuoh9/taichi-mpm-lbm`.
4. Verify the remote branch hash.
5. Report the final commit hash, pushed branch, and verification results.
