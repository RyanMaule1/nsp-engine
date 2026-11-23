"""Self-tests: is the golden reference itself right? Runs on the real 60 s sample."""

import hashlib
import json

import numpy as np
import pytest
from scipy import signal

from nsp_golden import FS, K, NOISE_REFRESH, NOISE_WINDOW, REFRACTORY, TILE_OFFSET, compare, events, recording
from nsp_golden.detect import detect


@pytest.fixture(scope="module")
def native(data_dir):
    return recording.load(data_dir)


@pytest.fixture(scope="module")
def golden_y(golden_dir):
    man = json.loads((golden_dir / "manifest.json").read_text())
    return compare.raw_f32(golden_dir / "filtered" / "traces_cached_seg0.raw", man["channels"])


def test_channel_map_and_probe(native):
    assert native.get_num_channels() == 374
    assert native.get_num_samples() == 60 * FS
    missing = sorted(set(range(384)) - {int(c) for c in native.channel_ids})
    assert missing == [36, 75, 112, 151, 188, 227, 264, 303, 340, 379]
    assert native.get_probe().get_contact_count() == 374


def test_filter_design_is_4th_order_butterworth():
    sos = recording.sos()
    assert sos.shape == (2, 6)
    _, h = signal.sosfreqz(sos, worN=[300, 1342, 6000], fs=FS)
    db = 20 * np.log10(np.abs(h))
    assert abs(db[0] + 3.01) < 0.05 and abs(db[1]) < 0.05 and abs(db[2] + 3.01) < 0.05


def test_si_chunked_filter_equals_one_continuous_sosfilt(native, golden_y):
    """SpikeInterface filters in 1 s chunks with a 100 ms margin; that must equal a
    single zero-state causal pass (up to float32 storage)."""
    cols = [0, 100, 373]
    raw = native.get_traces(channel_ids=native.channel_ids[cols]).astype(np.float64)
    ref = signal.sosfilt(recording.sos(), raw, axis=0)
    got = np.asarray(golden_y[:, cols], np.float64)
    rel = np.abs(got - ref).max(0) / np.abs(ref).max(0)
    assert rel.max() < 1e-6, rel


def brute_force(y, k=K):
    """Per-sample reading of the rules in detect.py, sharing no code with it."""
    out = []
    for c in range(y.shape[1]):
        last = -10 ** 9
        thr = None
        for n in range(len(y)):
            if n % NOISE_REFRESH == 0 and n >= NOISE_WINDOW:
                a = np.sort(np.abs(y[n - NOISE_WINDOW:n, c].astype(np.float64)))
                thr = -k * a[(NOISE_WINDOW - 1) // 2] / 0.6745
            if thr is not None and y[n, c] < thr <= y[n - 1, c] and n - last >= REFRACTORY:
                last = n
                out.append((n, c, int(np.rint(y[n, c])), int(np.rint(thr))))
    return sorted(out)


def test_detector_equals_brute_force(golden_y):
    y = np.asarray(golden_y[:10 * FS, [0, 57, 200]])
    n, ch, amp, thr, _, _ = detect(y)
    assert len(n) > 500
    assert list(zip(n.tolist(), ch.tolist(), amp.tolist(), thr.tolist())) == brute_force(y)


def test_golden_events_conform(golden_events, golden_y):
    assert events.validate(golden_events, channels=golden_y.shape[1], k=K,
                           n_samples=len(golden_y)) == []
    assert golden_events.records["sample_index"].min() >= NOISE_WINDOW


def test_golden_checksums(golden_dir):
    for line in (golden_dir / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        h = hashlib.sha256()
        with open(golden_dir / name, "rb") as f:
            while b := f.read(1 << 24):
                h.update(b)
        assert h.hexdigest() == digest, name


def test_crosschecks_against_independent_detectors(golden_dir):
    cc = json.loads((golden_dir / "manifest.json").read_text())["crosscheck"]
    assert cc["si_by_channel"]["si_peaks_with_crossing"] > 0.85
    assert cc["si_by_channel"]["golden_with_si_peak"] > 0.75
    if "kilosort_good_units" in cc:
        assert cc["kilosort_good_units"]["median_recall"] > 0.85


def test_tiling_channel_model(native):
    rec = recording.tile(native, 1024)
    assert rec.get_num_channels() == 1024
    a = native.get_traces(start_frame=TILE_OFFSET, end_frame=TILE_OFFSET + 30, channel_ids=native.channel_ids[:1])
    b = rec.get_traces(start_frame=0, end_frame=30, channel_ids=[rec.channel_ids[374]])
    assert np.array_equal(a, b)
    n = native.get_num_samples()
    c = rec.get_traces(start_frame=n - 30, end_frame=n, channel_ids=[rec.channel_ids[2 * 374]])
    w = native.get_traces(start_frame=2 * TILE_OFFSET - 30, end_frame=2 * TILE_OFFSET,
                          channel_ids=native.channel_ids[:1])                       # wrapped
    assert np.array_equal(c, w)


def test_event_file_roundtrip_with_snippets(tmp_path):
    n = np.array([40000, 40000, 50000])
    ch = np.array([1, 3, 1])
    snip = np.zeros((3, 32), np.int16)
    snip[:, 8] = -90
    has = np.array([True, False, True])
    events.write(tmp_path / "e.nspe", 4, 4.5, n, ch, [-90] * 3, [-50] * 3, (snip, has))
    assert (tmp_path / "e.nspe").stat().st_size == 16 + 3 * 16 + 2 * 64
    ev = events.read(tmp_path / "e.nspe")
    assert ev.records["sample_index"].tolist() == n.tolist() and sorted(ev.snippets) == [0, 2]
    assert events.validate(ev, channels=4, k=4.5) == []
    hdr = (tmp_path / "e.nspe").read_bytes()[:16]
    assert hdr == b"NSPE\x01\x00\x04\x00" + (30_000_000).to_bytes(4, "little") + (450).to_bytes(4, "little")


def test_comparison_tools_catch_differences(golden_events):
    def subset(mask=None, shift=0):
        r = golden_events.records.copy()
        if mask is not None:
            r = r[mask]
        r["sample_index"] += shift
        return events.Events(golden_events.channel_count, golden_events.k_x100, r, {})
    same = compare.event_agreement(subset(), golden_events)
    assert same["recall"] == 1 and same["precision"] == 1 and same["exact_time"] == same["golden"]
    half = compare.event_agreement(subset(np.arange(len(golden_events.records)) % 2 == 0), golden_events)
    assert abs(half["recall"] - 0.5) < 0.01 and half["precision"] == 1
    late = compare.event_agreement(subset(shift=1), golden_events, delta=1)
    assert late["recall"] == 1 and late["exact_time"] == 0
    assert compare.event_agreement(subset(shift=2), golden_events, delta=1)["recall"] < 0.01
