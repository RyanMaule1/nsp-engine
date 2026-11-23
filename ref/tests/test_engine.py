"""C++ engine vs golden. Enabled with --engine-out DIR, where DIR holds:

    events.nspe    spike events, spec 7.2
    filtered.f32   filtered signal, float32 LE, sample-major, same channels/order as golden
                   (for the fixed path: convert to ADC units before writing)

    pytest tests/test_engine.py --engine-out ../build/out [--fixed] [--golden golden/1024ch]
"""

from pathlib import Path

import pytest

from nsp_golden import K, compare, events


@pytest.fixture(scope="module")
def engine_dir(request):
    d = request.config.getoption("--engine-out")
    if d is None:
        pytest.skip("pass --engine-out to test the C++ engine")
    return Path(d)


@pytest.fixture(scope="module")
def engine_events(engine_dir):
    p = engine_dir / "events.nspe"
    if not p.exists():
        pytest.skip(f"{p} missing")
    return events.read(p)


def test_t12_event_format(engine_events, golden_events):
    n_samples = int(golden_events.records["sample_index"].max()) + 1_000_000
    assert events.validate(engine_events, channels=golden_events.channel_count, k=K,
                           n_samples=n_samples) == []


def test_t5_spike_agreement(engine_events, golden_events, request):
    """Float path: >= 99.9 % both ways within +-1 sample. Fixed path (PR-10): >= 99.5 %."""
    floor = 0.995 if request.config.getoption("--fixed") else 0.999
    r = compare.event_agreement(engine_events, golden_events, delta=1)
    print(r)
    assert r["recall"] >= floor and r["precision"] >= floor, r


def test_t3_filter_equivalence(engine_dir, golden_dir, golden_events, request):
    p = engine_dir / "filtered.f32"
    if not p.exists():
        pytest.skip(f"{p} missing")
    nch = golden_events.channel_count
    r = compare.filter_error(compare.raw_f32(p, nch),
                             compare.raw_f32(golden_dir / "filtered" / "traces_cached_seg0.raw", nch))
    print(r)
    if request.config.getoption("--fixed"):
        assert r["max_abs"] <= request.config.getoption("--lsb-tol"), r
    else:
        assert r["peak_rel_max"] <= 1e-5, r
