from __future__ import annotations

from .config import CouplingConfig


class LBMMpmCoupler:
    """LBM-MPM coupling placeholder.

    Later steps will implement velocity interpolation, penalty coupling, and
    reaction force scattering.
    """

    def __init__(self, config: CouplingConfig):
        self.config = config
        self.config.validate()

    def step(self, dt: float) -> None:
        raise NotImplementedError("LBM-MPM coupling will be implemented later.")
