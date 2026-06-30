from __future__ import annotations

import numpy as np

from .config import BoundaryConfig, CouplingConfig, LBMConfig, MPMConfig, SimulationConfig
from .lbm3d import LBMSolver3D
from .mpm3d import MPMSolver3D
from .simulation import FSISimulation
from .validation import ValidationReport, bounded_metric, finite_metric, relative_error


def _periodic_lbm_config(**overrides) -> LBMConfig:
    params = {
        "nx": 8,
        "ny": 8,
        "nz": 8,
        "viscosity": 0.1,
        "rho0": 1.0,
        "force": (0.0, 0.0, 0.0),
        "initial_velocity": (0.0, 0.0, 0.0),
        "x_left": BoundaryConfig("periodic"),
        "x_right": BoundaryConfig("periodic"),
        "y_left": BoundaryConfig("periodic"),
        "y_right": BoundaryConfig("periodic"),
        "z_left": BoundaryConfig("periodic"),
        "z_right": BoundaryConfig("periodic"),
    }
    params.update(overrides)
    return LBMConfig(**params)


def _mpm_config(**overrides) -> MPMConfig:
    params = {
        "nx": 16,
        "ny": 16,
        "nz": 16,
        "max_particles": 10_000,
        "dx": 1.0,
        "dt": 1.0e-3,
        "density": 1.0,
        "youngs_modulus": 1_000.0,
        "poisson_ratio": 0.25,
        "gravity": (0.0, 0.0, 0.0),
        "boundary_width": 2,
    }
    params.update(overrides)
    return MPMConfig(**params)


def _coupled_config(*, enabled: bool = True) -> SimulationConfig:
    return SimulationConfig(
        num_steps=2,
        lbm_dt=0.1,
        lbm=_periodic_lbm_config(initial_velocity=(1.0e-3, 0.0, 0.0)),
        mpm=MPMConfig(
            nx=8,
            ny=8,
            nz=8,
            max_particles=128,
            dx=1.0,
            dt=1.0e-3,
            density=1.0,
            youngs_modulus=100.0,
            poisson_ratio=0.25,
            gravity=(0.0, 0.0, 0.0),
            boundary_width=1,
        ),
        coupling=CouplingConfig(
            enabled=enabled,
            gamma=5.0,
            mpm_substeps_per_lbm_step=1,
        ),
    )


def _initialized_one_particle_simulation(config: SimulationConfig) -> FSISimulation:
    sim = FSISimulation(config)
    sim.initialize_mpm_from_numpy(
        positions=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
        velocities=np.zeros((1, 3), dtype=np.float32),
        particle_mass=1.0,
        particle_volume=1.0,
    )
    sim.initialize_lbm()
    return sim


def run_lbm_mass_conservation_case() -> ValidationReport:
    steps = 20
    solver = LBMSolver3D(_periodic_lbm_config(force=(0.0, 0.0, 0.0)))
    solver.initialize()

    mass0 = solver.total_mass()
    for _ in range(steps):
        solver.step()
    mass1 = solver.total_mass()

    return ValidationReport(
        case_name="lbm_periodic_mass_conservation",
        metrics=(
            bounded_metric(
                "relative_mass_error",
                relative_error(mass1, mass0),
                upper=1.0e-5,
                description="Relative total fluid mass drift on a periodic grid.",
            ),
            finite_metric(
                "max_velocity_norm",
                solver.max_velocity_norm(),
                upper=1.0e-4,
                description="Maximum velocity norm remains finite and near zero.",
            ),
        ),
        metadata={"steps": steps, "grid": [solver.nx, solver.ny, solver.nz]},
    )


def run_lbm_force_response_case() -> ValidationReport:
    steps = 20
    solver = LBMSolver3D(_periodic_lbm_config(force=(1.0e-6, 0.0, 0.0)))
    solver.initialize()

    ux0 = float(solver.velocity_numpy()[..., 0].mean())
    for _ in range(steps):
        solver.step()
    ux1 = float(solver.velocity_numpy()[..., 0].mean())

    return ValidationReport(
        case_name="lbm_force_response",
        metrics=(
            bounded_metric(
                "mean_ux_growth",
                ux1 - ux0,
                lower=1.0e-12,
                description="Mean x velocity grows under positive x force.",
            ),
            finite_metric(
                "max_velocity_norm",
                solver.max_velocity_norm(),
                upper=1.0e-2,
                description="Forced velocity remains bounded in the smoke case.",
            ),
        ),
        metadata={"steps": steps, "grid": [solver.nx, solver.ny, solver.nz]},
    )


def run_mpm_zero_gravity_com_case() -> ValidationReport:
    steps = 10
    solver = MPMSolver3D(_mpm_config(gravity=(0.0, 0.0, 0.0), dt=1.0e-3))
    solver.initialize_particles_box((6.0, 6.0, 6.0), (10.0, 10.0, 10.0), spacing=1.0)

    com0 = solver.center_of_mass()
    for _ in range(steps):
        solver.substep()
    com1 = solver.center_of_mass()

    return ValidationReport(
        case_name="mpm_zero_gravity_com_stability",
        metrics=(
            bounded_metric(
                "center_of_mass_drift_norm",
                float(np.linalg.norm(com1 - com0)),
                upper=1.0e-3,
                units="grid",
                description="COM drift stays small without gravity or initial velocity.",
            ),
            finite_metric(
                "max_velocity_norm",
                solver.max_velocity_norm(),
                upper=1.0e-3,
                description="MPM velocities remain bounded without gravity.",
            ),
        ),
        metadata={"steps": steps, "particle_count": solver.particle_count()},
    )


def run_mpm_gravity_response_case() -> ValidationReport:
    steps = 10
    solver = MPMSolver3D(_mpm_config(gravity=(0.0, -1.0e-3, 0.0), dt=1.0e-2))
    solver.initialize_particles_box((6.0, 8.0, 6.0), (10.0, 12.0, 10.0), spacing=1.0)

    com0 = solver.center_of_mass()
    for _ in range(steps):
        solver.substep()
    com1 = solver.center_of_mass()
    positions = solver.positions_numpy()
    velocities = solver.velocities_numpy()

    return ValidationReport(
        case_name="mpm_gravity_response",
        metrics=(
            bounded_metric(
                "center_of_mass_y_delta",
                float(com1[1] - com0[1]),
                upper=-1.0e-12,
                units="grid",
                description="COM moves downward under weak negative y gravity.",
            ),
            bounded_metric(
                "positions_finite",
                1.0 if np.isfinite(positions).all() else 0.0,
                lower=1.0,
                upper=1.0,
                description="All particle positions remain finite.",
            ),
            bounded_metric(
                "velocities_finite",
                1.0 if np.isfinite(velocities).all() else 0.0,
                lower=1.0,
                upper=1.0,
                description="All particle velocities remain finite.",
            ),
        ),
        metadata={"steps": steps, "particle_count": solver.particle_count()},
    )


def run_coupled_drift_case() -> ValidationReport:
    enabled = _initialized_one_particle_simulation(_coupled_config(enabled=True))
    disabled = _initialized_one_particle_simulation(_coupled_config(enabled=False))

    enabled_initial_x = float(enabled.mpm.center_of_mass()[0])
    disabled_initial_x = float(disabled.mpm.center_of_mass()[0])

    enabled.run(steps=2)
    disabled.run(steps=2)

    enabled_dx = float(enabled.mpm.center_of_mass()[0] - enabled_initial_x)
    disabled_dx = float(disabled.mpm.center_of_mass()[0] - disabled_initial_x)

    return ValidationReport(
        case_name="coupled_enabled_vs_disabled_drift",
        metrics=(
            bounded_metric(
                "enabled_dx",
                enabled_dx,
                lower=1.0e-7,
                units="grid",
                description="Enabled coupling advects the particle in +x.",
            ),
            bounded_metric(
                "disabled_dx_abs",
                abs(disabled_dx),
                upper=1.0e-6,
                units="grid",
                description="Disabled coupling leaves the particle nearly fixed.",
            ),
            bounded_metric(
                "enabled_minus_disabled_dx",
                enabled_dx - disabled_dx,
                lower=1.0e-7,
                units="grid",
                description="Enabled drift exceeds the disabled baseline.",
            ),
        ),
        metadata={"steps": 2, "gamma": 5.0, "initial_lbm_ux": 1.0e-3},
    )


def run_coupling_force_balance_case() -> ValidationReport:
    sim = _initialized_one_particle_simulation(_coupled_config(enabled=True))
    sim.coupler.compute_coupling_forces(dt_ratio=1.0)

    particle_force = sim.coupler.total_particle_coupling_force()
    fluid_force_density = sim.coupler.total_fluid_coupling_force()
    integrated_fluid_force = fluid_force_density * sim.coupler.cell_volume
    balance_norm = float(np.linalg.norm(particle_force + integrated_fluid_force))

    return ValidationReport(
        case_name="coupling_force_balance",
        metrics=(
            bounded_metric(
                "particle_force_x",
                float(particle_force[0]),
                lower=1.0e-12,
                description="Particle receives positive x force from uniform flow.",
            ),
            bounded_metric(
                "fluid_force_x",
                float(integrated_fluid_force[0]),
                upper=-1.0e-12,
                description="Fluid receives opposite x reaction force.",
            ),
            bounded_metric(
                "force_balance_norm",
                balance_norm,
                upper=1.0e-6,
                description="Particle force plus integrated fluid reaction is balanced.",
            ),
        ),
        metadata={"dt_ratio": 1.0, "cell_volume": sim.coupler.cell_volume},
    )


def run_coupling_force_limit_case() -> ValidationReport:
    config = SimulationConfig(
        num_steps=1,
        lbm_dt=0.1,
        lbm=_periodic_lbm_config(initial_velocity=(1.0, 0.0, 0.0)),
        mpm=MPMConfig(
            nx=8,
            ny=8,
            nz=8,
            max_particles=128,
            dx=1.0,
            dt=1.0e-3,
            density=1.0,
            youngs_modulus=100.0,
            poisson_ratio=0.25,
            gravity=(0.0, 0.0, 0.0),
            boundary_width=1,
        ),
        coupling=CouplingConfig(
            gamma=100.0,
            force_limit=0.05,
            mpm_substeps_per_lbm_step=1,
        ),
    )
    sim = _initialized_one_particle_simulation(config)
    sim.coupler.compute_coupling_forces(dt_ratio=1.0)

    particle_force = sim.coupler.total_particle_coupling_force()
    fluid_force_density = sim.coupler.total_fluid_coupling_force()
    integrated_fluid_force = fluid_force_density * sim.coupler.cell_volume
    diagnostics = sim.coupler.coupling_diagnostics()

    return ValidationReport(
        case_name="coupling_force_limit",
        metrics=(
            bounded_metric(
                "particle_force_norm",
                float(np.linalg.norm(particle_force)),
                upper=0.05 + 1.0e-7,
                description="High-gamma particle coupling force is clipped.",
            ),
            bounded_metric(
                "clipped_particle_count",
                float(diagnostics["clipped_particle_count"]),
                lower=1.0,
                description="At least one particle records force or velocity clipping.",
            ),
            bounded_metric(
                "force_balance_norm",
                float(np.linalg.norm(particle_force + integrated_fluid_force)),
                upper=1.0e-6,
                description="Clipped particle force still balances fluid reaction.",
            ),
        ),
        metadata={"gamma": 100.0, "force_limit": 0.05},
    )


def run_coupling_boundary_support_case() -> ValidationReport:
    config = SimulationConfig(
        num_steps=1,
        lbm_dt=0.1,
        lbm=_periodic_lbm_config(initial_velocity=(1.0, 0.0, 0.0)),
        mpm=MPMConfig(
            nx=8,
            ny=8,
            nz=8,
            max_particles=128,
            dx=1.0,
            dt=1.0e-3,
            density=1.0,
            youngs_modulus=100.0,
            poisson_ratio=0.25,
            gravity=(0.0, 0.0, 0.0),
            boundary_width=1,
        ),
        coupling=CouplingConfig(
            gamma=1.0,
            min_valid_weight=1.0e-6,
            mpm_substeps_per_lbm_step=1,
        ),
    )
    sim = FSISimulation(config)
    sim.initialize_mpm_from_numpy(
        positions=np.array([[-0.75, 4.0, 4.0]], dtype=np.float32),
        velocities=np.zeros((1, 3), dtype=np.float32),
        particle_mass=1.0,
        particle_volume=1.0,
    )
    sim.initialize_lbm()
    sim.coupler.compute_coupling_forces(dt_ratio=1.0)

    particle_force = sim.coupler.total_particle_coupling_force()
    fluid_force_density = sim.coupler.total_fluid_coupling_force()
    integrated_fluid_force = fluid_force_density * sim.coupler.cell_volume
    diagnostics = sim.coupler.coupling_diagnostics()

    return ValidationReport(
        case_name="coupling_boundary_support",
        metrics=(
            bounded_metric(
                "partial_support_particle_count",
                float(diagnostics["partial_support_particle_count"]),
                lower=1.0,
                description="Near-boundary particle records partial support.",
            ),
            bounded_metric(
                "min_particle_valid_weight",
                float(diagnostics["min_particle_valid_weight"]),
                lower=1.0e-6,
                upper=1.0 - 1.0e-5,
                description="Valid support is finite and below full support.",
            ),
            bounded_metric(
                "force_balance_norm",
                float(np.linalg.norm(particle_force + integrated_fluid_force)),
                upper=1.0e-6,
                description="Partial-support coupling still balances fluid reaction.",
            ),
            bounded_metric(
                "particle_force_finite",
                1.0 if np.isfinite(particle_force).all() else 0.0,
                lower=1.0,
                upper=1.0,
                description="Boundary particle force remains finite.",
            ),
        ),
        metadata={"particle_position": [-0.75, 4.0, 4.0], "min_valid_weight": 1.0e-6},
    )


def run_immersed_boundary_drag_case() -> ValidationReport:
    config = SimulationConfig(
        num_steps=1,
        lbm_dt=1.0e-3,
        lbm=_periodic_lbm_config(initial_velocity=(1.0e-3, 0.0, 0.0)),
        mpm=MPMConfig(
            nx=8,
            ny=8,
            nz=8,
            max_particles=128,
            dx=1.0,
            dt=1.0e-3,
            density=1.0,
            youngs_modulus=100.0,
            poisson_ratio=0.25,
            gravity=(0.0, 0.0, 0.0),
            boundary_width=1,
        ),
        coupling=CouplingConfig(
            gamma=0.0,
            mpm_substeps_per_lbm_step=1,
            immersed_boundary_enabled=True,
            immersed_boundary_drag=0.1,
            immersed_boundary_fraction_threshold=0.01,
            immersed_boundary_max_force=1.0e-3,
        ),
    )
    sim = FSISimulation(config)
    sim.initialize_lbm()
    sim.initialize_mpm_from_numpy(
        positions=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
        velocities=np.zeros((1, 3), dtype=np.float32),
        particle_mass=1.0,
        particle_volume=4.0,
    )

    initial_speed = sim.lbm.max_velocity_norm()
    sim.step()
    diagnostics = sim.coupler.coupling_diagnostics()
    velocity = sim.lbm.velocity_numpy()

    return ValidationReport(
        case_name="immersed_boundary_drag",
        metrics=(
            bounded_metric(
                "ib_active_cell_count",
                float(diagnostics["ib_active_cell_count"]),
                lower=1.0,
                description="At least one dynamic occupied fluid cell receives IB drag.",
            ),
            bounded_metric(
                "ib_total_force_x",
                float(diagnostics["ib_total_force_x"]),
                upper=-1.0e-12,
                description="IB drag opposes positive x fluid velocity.",
            ),
            bounded_metric(
                "ib_total_force_norm",
                float(diagnostics["ib_total_force_norm"]),
                lower=1.0e-12,
                description="IB drag total force is nonzero.",
            ),
            bounded_metric(
                "max_velocity_not_increased",
                float(sim.lbm.max_velocity_norm() - initial_speed),
                upper=1.0e-8,
                description="The IB smoke case does not increase max fluid speed.",
            ),
            bounded_metric(
                "velocity_finite",
                1.0 if np.isfinite(velocity).all() else 0.0,
                lower=1.0,
                upper=1.0,
                description="LBM velocity remains finite with IB drag enabled.",
            ),
        ),
        metadata={"drag": 0.1, "particle_volume": 4.0},
    )


def run_contact_diagnostics_case() -> ValidationReport:
    config = SimulationConfig(
        num_steps=1,
        lbm_dt=1.0e-3,
        lbm=_periodic_lbm_config(initial_velocity=(0.0, 0.0, 0.0)),
        mpm=MPMConfig(
            nx=8,
            ny=8,
            nz=8,
            max_particles=128,
            dx=1.0,
            dt=1.0e-3,
            density=1.0,
            youngs_modulus=100.0,
            poisson_ratio=0.25,
            gravity=(0.0, 0.0, 0.0),
            boundary_width=1,
        ),
        coupling=CouplingConfig(
            gamma=0.0,
            mpm_substeps_per_lbm_step=1,
            contact_enabled=True,
            contact_velocity_damping=0.25,
            contact_fraction_threshold=0.01,
        ),
    )
    solid = np.zeros((8, 8, 8), dtype=np.int32)
    solid[3:6, 3:6, 3:6] = 1

    sim = FSISimulation(config)
    sim.initialize_lbm(solid_np=solid)
    sim.initialize_mpm_from_numpy(
        positions=np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
        velocities=np.array([[0.1, 0.0, 0.0]], dtype=np.float32),
        particle_mass=1.0,
        particle_volume=1.0,
    )

    initial_speed = sim.mpm.max_velocity_norm()
    sim.step()
    final_speed = sim.mpm.max_velocity_norm()
    diagnostics = sim.coupler.coupling_diagnostics()
    contact_mask_sum = float(sim.coupler.particle_contact_mask_numpy().sum())

    return ValidationReport(
        case_name="contact_diagnostics",
        metrics=(
            bounded_metric(
                "contact_candidate_count",
                float(diagnostics["contact_candidate_count"]),
                lower=1.0,
                description="Static solid support marks at least one contact candidate.",
            ),
            bounded_metric(
                "contact_damped_particle_count",
                float(diagnostics["contact_damped_particle_count"]),
                lower=1.0,
                description="Contact damping is applied to at least one particle.",
            ),
            bounded_metric(
                "particle_contact_mask_sum",
                contact_mask_sum,
                lower=1.0,
                description="The active particle contact mask is set.",
            ),
            bounded_metric(
                "particle_speed_delta",
                float(final_speed - initial_speed),
                upper=-1.0e-6,
                description="Contact damping reduces particle speed.",
            ),
        ),
        metadata={"damping": 0.25, "initial_speed": initial_speed},
    )


def run_validation_suite() -> list[ValidationReport]:
    return [
        run_lbm_mass_conservation_case(),
        run_lbm_force_response_case(),
        run_mpm_zero_gravity_com_case(),
        run_mpm_gravity_response_case(),
        run_coupled_drift_case(),
        run_coupling_force_balance_case(),
        run_coupling_force_limit_case(),
        run_coupling_boundary_support_case(),
        run_immersed_boundary_drag_case(),
        run_contact_diagnostics_case(),
    ]
