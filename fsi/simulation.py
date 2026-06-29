from __future__ import annotations

from .config import SimulationConfig


class FSISimulation:
    """Top-level coupled simulation placeholder."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.config.validate()

    def run(self) -> None:
        raise NotImplementedError("Simulation loop will be implemented later.")
