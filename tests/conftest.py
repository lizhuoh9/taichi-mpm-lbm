import taichi as ti


def pytest_configure(config):
    ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False, offline_cache=False)
