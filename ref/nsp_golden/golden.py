"""``nsp-golden``: freeze the golden files for a channel count.

    nsp-golden --data ../data --out golden/374ch
    nsp-golden --data ../data --out golden/1024ch --channels 1024

Output directory:
    filtered/traces_cached_seg0.raw   float32 sample-major filtered signal (T-3), plus
                                      SpikeInterface metadata (si.load(out/'filtered'))
    events.nspe                       golden events, spec 7.2 (T-5)
    noise.npy                         float64 [n_blocks, ch] noise per 100 ms block (NaN = none)
    si_peaks.npy                      SpikeInterface by_channel peaks (cross-check)
    manifest.json, SHA256SUMS
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import spikeinterface
import spikeinterface.core as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from . import FS, K, NOISE_REFRESH, events, recording
from .compare import kilosort_recall, raw_f32, si_peak_agreement
from .detect import detect

GROUP = 16          # channels per detection pass (bounds memory to ~1 GB)


def sha256(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def run_detection(y: np.ndarray, k: float, snippets: bool):
    parts, noise = [], []
    for c0 in range(0, y.shape[1], GROUP):
        n, ch, amp, thr, snip, nz = detect(np.asarray(y[:, c0:c0 + GROUP]), k, snippets=snippets)
        parts.append((n, ch + c0, amp, thr, snip))
        noise.append(nz)
    n = np.concatenate([p[0] for p in parts])
    ch = np.concatenate([p[1] for p in parts])
    o = np.lexsort((ch, n))
    cat = lambda i: np.concatenate([p[i] for p in parts])[o]          # noqa: E731
    snip = None
    if snippets:
        snip = (np.concatenate([p[4][0] for p in parts])[o], np.concatenate([p[4][1] for p in parts])[o])
    return n[o], ch[o], cat(2), cat(3), snip, np.hstack(noise)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nsp-golden", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dir with rawDataSample.bin, channel_map.npy, ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", type=int, help="default: native (374 mapped sites)")
    ap.add_argument("--k", type=float, default=K)
    ap.add_argument("--snippets", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--skip-crosscheck", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, file=sys.stderr, flush=True)                # noqa: E731
    jobs = dict(n_jobs=a.n_jobs, chunk_duration="1s", progress_bar=False)

    native = recording.load(a.data)
    rec = recording.tile(native, a.channels or native.get_num_channels())
    nch, ns = rec.get_num_channels(), rec.get_num_samples()
    log(f"{nch} channels x {ns} samples ({ns / FS:.1f} s)")

    t = time.time()
    filt = recording.bandpass(rec).save(folder=out / "filtered", format="binary", overwrite=True, **jobs)
    log(f"filtered in {time.time() - t:.0f} s")
    y = raw_f32(out / "filtered" / "traces_cached_seg0.raw", nch)

    t = time.time()
    n, ch, amp, thr, snip, noise = run_detection(y, a.k, a.snippets)
    log(f"{len(n)} events in {time.time() - t:.0f} s")
    events.write(out / "events.nspe", nch, a.k, n, ch, amp, thr, snip)
    np.save(out / "noise.npy", noise)
    golden = events.read(out / "events.nspe")

    cross = {}
    if not a.skip_crosscheck:
        t = time.time()
        global_noise = si.get_noise_levels(filt, return_in_uV=False, method="mad", progress_bar=False)
        peaks = detect_peaks(filt, method="by_channel",
                             method_kwargs=dict(peak_sign="neg", detect_threshold=a.k,
                                                exclude_sweep_ms=1.0, noise_levels=global_noise),
                             job_kwargs=jobs)
        np.save(out / "si_peaks.npy", peaks)
        cross["si_by_channel"] = si_peak_agreement(golden, peaks)
        if nch == native.get_num_channels():
            cross["kilosort_good_units"] = kilosort_recall(golden, a.data, ns)
        log(f"cross-checks in {time.time() - t:.0f} s: {json.dumps(cross)}")

    data = Path(a.data)
    inputs = {p: sha256(data / p) for p in ("rawDataSample.bin", "channel_map.npy",
                                           "channel_positions.npy")}
    manifest = {
        "spikeinterface": spikeinterface.__version__, "numpy": np.__version__,
        "inputs_sha256": inputs,
        "channels": nch, "samples": ns, "sampling_frequency": FS,
        "channel_ids": [str(c) for c in rec.channel_ids],
        "uv_per_lsb": recording.UV_PER_LSB,
        "filter_sos": recording.sos().tolist(),
        "k": a.k, "noise_refresh": NOISE_REFRESH,
        "events": len(golden.records), "snippets": a.snippets,
        "crosscheck": cross,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    files = [out / "events.nspe", out / "noise.npy", out / "filtered" / "traces_cached_seg0.raw"]
    (out / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(out)}\n" for p in files))
    print(json.dumps({"events": manifest["events"], "crosscheck": cross}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
