"""Spike-event stream, spec 7.2: writer and independent parser (T-12)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spikeinterface.core as si

from . import FS, REFRACTORY

HDR = struct.Struct("<4sBBHII")          # magic, version, reserved, channels, rate mHz, k*100
REC = np.dtype([("sample_index", "<u8"), ("channel", "<u2"), ("flags", "<u2"),
                ("amplitude", "<i2"), ("threshold", "<i2")])
REC_SNIP = np.dtype(REC.descr + [("snippet", "<i2", (32,))])
assert HDR.size == 16 and REC.itemsize == 16 and REC_SNIP.itemsize == 80


@dataclass
class Events:
    channel_count: int
    k_x100: int
    records: np.ndarray                  # REC
    snippets: dict                       # record index -> int16[32]
    sample_rate_mhz: int = FS * 1000

    def to_sorting(self, fs: float = FS) -> si.NumpySorting:
        """One 'unit' per channel, for SpikeInterface's spike-train matching."""
        return si.NumpySorting.from_samples_and_labels(
            [self.records["sample_index"].astype(np.int64)], [self.records["channel"].astype(np.int64)],
            fs, unit_ids=np.arange(self.channel_count))


def write(path, channel_count, k, sample, channel, amplitude, threshold, snippets=None):
    """snippets: (int16[n, 32], bool[n] present) or None."""
    n = len(sample)
    has = snippets[1] if snippets is not None else np.zeros(n, bool)
    with open(path, "wb") as f:
        f.write(HDR.pack(b"NSPE", 1, 0, channel_count, FS * 1000, int(round(k * 100))))
        rec = np.zeros(n, REC)
        rec["sample_index"], rec["channel"] = sample, channel
        rec["amplitude"], rec["threshold"], rec["flags"] = amplitude, threshold, has
        if not has.any():
            f.write(rec.tobytes())
            return
        for i in range(n):                   # variable-length records
            f.write(rec[i:i + 1].tobytes())
            if has[i]:
                f.write(snippets[0][i].astype("<i2").tobytes())


def read(path) -> Events:
    b = Path(path).read_bytes()
    if len(b) < HDR.size:
        raise ValueError("shorter than the 16-byte header")
    magic, ver, rsv, ch, rate, kx = HDR.unpack_from(b)
    if magic != b"NSPE" or ver != 1:
        raise ValueError(f"not an NSPE v1 file (magic {magic!r}, version {ver})")
    body = memoryview(b)[HDR.size:]
    if len(body) % REC.itemsize == 0:
        r = np.frombuffer(body, REC)
        if not (r["flags"] & 1).any():
            return Events(ch, kx, r.copy(), {}, rate)
    recs, snips, off = [], {}, 0
    while off < len(body):
        if len(body) - off < REC.itemsize:
            raise ValueError(f"truncated record at body byte {off}")
        r = np.frombuffer(body[off:off + 16], REC)[0]
        off += 16
        if r["flags"] & 1:
            if len(body) - off < 64:
                raise ValueError(f"truncated snippet at body byte {off}")
            snips[len(recs)] = np.frombuffer(body[off:off + 64], "<i2").copy()
            off += 64
        recs.append(r)
    return Events(ch, kx, np.array(recs, REC), snips, rate)


def validate(ev: Events, channels: int | None = None, k: float | None = None,
             n_samples: int | None = None) -> list[str]:
    """T-12: every field of every record. Returns a list of problems."""
    p = []
    r = ev.records
    if ev.sample_rate_mhz != FS * 1000:
        p.append(f"sample_rate_mhz {ev.sample_rate_mhz} != {FS * 1000}")
    if channels is not None and ev.channel_count != channels:
        p.append(f"channel_count {ev.channel_count} != {channels}")
    if k is not None and ev.k_x100 != int(round(k * 100)):
        p.append(f"k_threshold_x100 {ev.k_x100} != {int(round(k * 100))}")
    if not len(r):
        return p
    si_, ch = r["sample_index"].astype(np.int64), r["channel"].astype(np.int64)
    checks = {
        "channel >= channel_count": ch >= ev.channel_count,
        "undefined flag bits": (r["flags"] & 0xFFFE) != 0,
        "threshold not negative": r["threshold"] >= 0,
        "amplitude above threshold": r["amplitude"] > r["threshold"],
    }
    if n_samples is not None:
        checks["sample_index past end of recording"] = si_ >= n_samples
    bad_order = np.zeros(len(r), bool)
    bad_order[1:] = (np.diff(si_) < 0) | ((np.diff(si_) == 0) & (np.diff(ch) <= 0))
    checks["not sorted by (sample_index, channel)"] = bad_order
    o = np.lexsort((si_, ch))
    refr = np.zeros(len(r), bool)
    refr[o[1:]] = (np.diff(ch[o]) == 0) & (np.diff(si_[o]) < REFRACTORY)
    checks["refractory violated"] = refr
    for name, mask in checks.items():
        if mask.any():
            p.append(f"{name}: {int(mask.sum())} records, first at index {int(np.argmax(mask))}")
    bad_snip = [i for i, s in ev.snippets.items() if s[8] != r["amplitude"][i]]
    if bad_snip:
        p.append(f"snippet[8] != amplitude: {len(bad_snip)} records, first {bad_snip[0]}")
    return p
