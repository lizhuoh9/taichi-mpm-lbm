def test_import_fsi_package():
    import fsi

    assert hasattr(fsi, "SimulationConfig")
    assert hasattr(fsi, "LBMConfig")
    assert hasattr(fsi, "MPMConfig")
    assert hasattr(fsi, "CouplingConfig")
    assert hasattr(fsi, "LBMSolver3D")
    assert hasattr(fsi, "MPMSolver3D")
    assert hasattr(fsi, "LBMMpmCoupler")
    assert hasattr(fsi, "FSISimulation")
    assert hasattr(fsi, "SimulationOutputWriter")
    assert hasattr(fsi, "ValidationMetric")
    assert hasattr(fsi, "ValidationReport")
    assert hasattr(fsi, "run_validation_suite")
    assert "SnapshotInfo" in fsi.__all__
    assert "list_npz_snapshots" in fsi.__all__
    assert "extract_snapshot_timeseries" in fsi.__all__
    assert "load_validation_summary" in fsi.__all__
    assert "validation_summary_table" in fsi.__all__
