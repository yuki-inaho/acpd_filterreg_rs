from __future__ import annotations
import pytest
from driver_adapter import Driver


def pytest_addoption(parser):
    parser.addoption("--native-engine", choices=["cpp", "rust", "driver"], default="cpp")
    parser.addoption("--driver", default="cpp/build/core/acpd_test_driver")


@pytest.fixture
def engine(request, monkeypatch):
    choice = request.config.getoption("--native-engine")
    if choice == "driver":
        from acpd_filterreg import _api
        bridge = Driver(request.config.getoption("--driver"))
        monkeypatch.setattr(_api, "_get_engine", lambda _: bridge)
        return "cpp"
    return choice


@pytest.fixture
def driver(request):
    return Driver(request.config.getoption('--driver'))
