from __future__ import annotations

from .config import LBMConfig


class LBMSolver3D:
    """3D LBM solver placeholder.

    Step 2 will adapt the D3Q19 MRT solver from ``third_party/taichi_LBM3D``.
    """

    def __init__(self, config: LBMConfig):
        self.config = config
        self.config.validate()

    def initialize(self) -> None:
        raise NotImplementedError("LBMSolver3D.initialize will be implemented in step 2.")

    def step(self) -> None:
        raise NotImplementedError("LBMSolver3D.step will be implemented in step 2.")
