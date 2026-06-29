from __future__ import annotations

from .config import MPMConfig


class MPMSolver3D:
    """3D MLS-MPM solver placeholder.

    Step 4 will implement a Python-Taichi 3D MLS-MPM solver inspired by taichi_mpm.
    """

    def __init__(self, config: MPMConfig):
        self.config = config
        self.config.validate()

    def initialize_particles_box(self) -> None:
        raise NotImplementedError("MPM particle initialization will be implemented later.")

    def substep(self, dt: float) -> None:
        raise NotImplementedError("MPMSolver3D.substep will be implemented later.")
