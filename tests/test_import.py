def test_import_fsi_package():
    import fsi

    assert hasattr(fsi, "SimulationConfig")
    assert hasattr(fsi, "LBMConfig")
    assert hasattr(fsi, "MPMConfig")
    assert hasattr(fsi, "CouplingConfig")
