# Immersed-Boundary / Contact MVP

Step 12 adds an opt-in immersed-boundary/contact MVP to the existing
penalty-based LBM-MPM coupling.

This feature is intentionally small. It provides dynamic occupancy diagnostics,
Brinkman-style fluid resistance, and lightweight particle contact diagnostics.
It does not implement a full no-slip immersed-boundary method or rigid-body
contact mechanics.

## What It Does

- Reuses `LBMMpmCoupler.solid_volume_fraction` as a dynamic MPM occupancy field.
- Computes an optional immersed-boundary drag force on fluid cells:

```text
F_ib = -immersed_boundary_drag * solid_volume_fraction * rho * velocity
```

- Keeps dynamic occupancy separate from the static LBM `solid` mask.
- Records immersed-boundary active cell counts, clipped cell counts, and total
  force.
- Detects simple contact candidates from dynamic occupancy or static solid
  support near MPM particles.
- Optionally damps velocities of contact-candidate particles.
- Writes IB/contact diagnostics to NPZ snapshots and post-processing time
  series.

## What It Does Not Do

- It is not a strict no-slip immersed-boundary method.
- It is not cut-cell, CPIC, or a rigid-body solver.
- It does not implement particle-particle contact.
- It does not model fracture, plasticity, or multi-material contact.
- It does not mutate the static LBM `solid` mask from MPM particle occupancy.
- It does not provide external experimental or paper-level validation.

## Configuration

All new controls live in `CouplingConfig` and default to disabled:

```python
CouplingConfig(
    immersed_boundary_enabled=False,
    immersed_boundary_drag=0.0,
    immersed_boundary_fraction_threshold=0.1,
    immersed_boundary_max_force=None,
    contact_enabled=False,
    contact_velocity_damping=0.0,
    contact_fraction_threshold=0.5,
)
```

`immersed_boundary_enabled` turns on dynamic IB drag. `immersed_boundary_drag`
is the resistance coefficient. `immersed_boundary_fraction_threshold` ignores
tiny occupancy values. `immersed_boundary_max_force` optionally clips each
cell's IB force norm.

`contact_enabled` turns on contact candidate diagnostics. If
`contact_velocity_damping` is positive, candidate particle velocities are
multiplied by `1 - contact_velocity_damping`. `contact_fraction_threshold`
controls the dynamic/static support threshold for marking a candidate.

## Diagnostics

`LBMMpmCoupler.coupling_diagnostics()` includes:

- `ib_active_cell_count`
- `ib_clipped_cell_count`
- `ib_total_force_x`
- `ib_total_force_y`
- `ib_total_force_z`
- `ib_total_force_norm`
- `contact_candidate_count`
- `contact_damped_particle_count`

`FSISimulation.diagnostics()` exposes scalar coupling-prefixed versions:

- `coupling_ib_active_cell_count`
- `coupling_ib_clipped_cell_count`
- `coupling_ib_total_force_norm`
- `coupling_contact_candidate_count`
- `coupling_contact_damped_particle_count`

## Snapshot Fields

When `OutputConfig.write_coupling_fields=True`, NPZ snapshots include:

- `immersed_boundary_force`
- `ib_total_force`
- `ib_active_cell_count`
- `ib_clipped_cell_count`
- `particle_contact_mask`
- `contact_candidate_count`
- `contact_damped_particle_count`

`extract_snapshot_timeseries()` can derive:

- `ib_force_norm`
- `ib_total_force_norm`
- `ib_active_cell_count`
- `ib_clipped_cell_count`
- `contact_candidate_count`
- `contact_damped_particle_count`

## Example

Run:

```bash
python examples/immersed_boundary_contact_smoke.py
```

The example uses a tiny grid, enables IB drag and contact diagnostics, and
prints scalar diagnostics for a few steps. It does not write heavy output by
default.

## Validation

Step 12 adds two lightweight validation cases:

- `immersed_boundary_drag`
- `contact_diagnostics`

The committed reference-data suite includes
`data/reference/immersed_boundary_contact_reference.json`.

These checks are regression guards for the MVP behavior. Passing them does not
prove high-fidelity physical validation.
