from fsi.config import SimulationConfig


def main() -> None:
    cfg = SimulationConfig()
    cfg.validate()
    print("FSI LBM-MPM project skeleton is ready.")
    print(f"Grid: {cfg.lbm.nx} x {cfg.lbm.ny} x {cfg.lbm.nz}")
    print(f"LBM dt: {cfg.lbm_dt}")
    print(f"MPM dt: {cfg.mpm_dt}")


if __name__ == "__main__":
    main()
