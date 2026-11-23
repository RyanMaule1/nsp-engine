from pathlib import Path

import pytest

from nsp_golden import events

HERE = Path(__file__).resolve().parent.parent


def pytest_addoption(parser):
    g = parser.getgroup("nsp")
    g.addoption("--data", default=str(HERE.parent / "data"), help="extracted data.zip directory")
    g.addoption("--golden", default=str(HERE / "golden" / "374ch"), help="nsp-golden output dir")
    g.addoption("--engine-out", default=None,
                help="C++ engine output dir (events.nspe, filtered.f32); enables test_engine.py")
    g.addoption("--fixed", action="store_true", help="engine ran the fixed-point path (PR-10 rules)")
    g.addoption("--lsb-tol", type=float, default=2.0, help="T-3 fixed-path bound, ADC units")


@pytest.fixture(scope="session")
def data_dir(request):
    d = Path(request.config.getoption("--data"))
    if not (d / "rawDataSample.bin").exists():
        pytest.skip(f"no dataset at {d}")
    return d


@pytest.fixture(scope="session")
def golden_dir(request):
    d = Path(request.config.getoption("--golden"))
    if not (d / "events.nspe").exists():
        pytest.skip(f"no golden files at {d}: run nsp-golden first")
    return d


@pytest.fixture(scope="session")
def golden_events(golden_dir):
    return events.read(golden_dir / "events.nspe")
