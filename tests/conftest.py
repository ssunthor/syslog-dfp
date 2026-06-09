import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires RAPIDS cuDF/CuPy (NVIDIA GPU)")
    config.addinivalue_line("markers", "morpheus: requires the full Morpheus container stack")
