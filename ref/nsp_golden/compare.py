"""Comparisons: engine vs golden (T-3, T-5/PR-10) and golden vs independent
detectors (SpikeInterface by_channel, Kilosort units). Matching uses
SpikeInterface's spike-train comparison tools."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import spikeinterface.core as si
from spikeinterface.comparison.comparisontools import make_match_count_matrix

from . import FS
from .events import Events


def raw_f32(path: str | Path, n_channels: int) -> np.ndarray:
    """A headerless float32 little-endian sample-major file (what SpikeInterface's
    binary folder and the engine's filter dump both contain)."""
    return np.memmap(path, "<f4", "r").reshape(-1, n_channels)


def filter_error(test: np.ndarray, golden: np.ndarray, chunk: int = 300_000) -> dict:
    """T-3: per-channel max|test-golden| / max|golden| over the common length."""
    n = min(len(test), len(golden))
    err = np.zeros(golden.shape[1])
    ref = np.zeros(golden.shape[1])
    for s in range(0, n, chunk):
        g = np.asarray(golden[s:s + chunk], np.float64)
        t = np.asarray(test[s:s + chunk], np.float64)[:len(g)]
        err = np.maximum(err, np.abs(t - g).max(0))
        ref = np.maximum(ref, np.abs(g).max(0))
    rel = err / np.maximum(ref, 1e-30)
    return {"samples": n, "peak_rel_max": float(rel.max()), "max_abs": float(err.max()),
            "worst_channel": int(rel.argmax())}


def _per_channel_matches(a: si.BaseSorting, b: si.BaseSorting, delta: int) -> np.ndarray:
    m = make_match_count_matrix(a, b, delta_frames=delta)
    return np.diag(m.loc[a.unit_ids, a.unit_ids].to_numpy())


def event_agreement(test: Events, golden: Events, delta: int = 1) -> dict:
    """T-5 / PR-10: events matched on the same channel within +-delta samples."""
    if test.channel_count != golden.channel_count:
        raise ValueError("channel counts differ")
    g, t = golden.to_sorting(), test.to_sorting()
    matched = int(_per_channel_matches(g, t, delta).sum())
    ng, nt = len(golden.records), len(test.records)
    exact = len(np.intersect1d(_keys(golden), _keys(test)))
    return {"golden": ng, "test": nt, "matched": matched, "exact_time": exact,
            "recall": matched / max(ng, 1), "precision": matched / max(nt, 1), "delta": delta}


def _keys(ev: Events) -> np.ndarray:
    return ev.records["channel"].astype(np.int64) * (1 << 40) + ev.records["sample_index"].astype(np.int64)


def si_peak_agreement(golden: Events, peaks: np.ndarray, delta: int = 15) -> dict:
    """Golden crossings vs SpikeInterface by_channel peaks (troughs follow crossings)."""
    p = si.NumpySorting.from_samples_and_labels([peaks["sample_index"]], [peaks["channel_index"]],
                                                FS, unit_ids=np.arange(golden.channel_count))
    g = golden.to_sorting()
    # shift crossings by +delta so the +-delta window covers [crossing, crossing + 2*delta]
    shifted = si.NumpySorting.from_samples_and_labels(
        [golden.records["sample_index"].astype(np.int64) + delta],
        [golden.records["channel"].astype(np.int64)], FS, unit_ids=g.unit_ids)
    m = int(_per_channel_matches(shifted, p, delta).sum())
    return {"golden": len(golden.records), "si_peaks": len(peaks), "matched": m,
            "golden_with_si_peak": m / max(len(golden.records), 1),
            "si_peaks_with_crossing": m / max(len(peaks), 1), "window": [0, 2 * delta]}


def kilosort_recall(golden: Events, data_dir: str | Path, n_samples: int,
                    delta: int = 15, min_spikes: int = 50) -> dict:
    """Cross-check (S3: not a pass criterion): for each Kilosort 'good' unit,
    the fraction of its spikes with a golden event within +-delta samples on
    the best-matching channel."""
    import pandas as pd
    import spikeinterface.extractors as se
    ks = se.read_kilosort(data_dir).frame_slice(0, n_samples)
    # this dataset's cluster_groups.csv is tab-separated, which read_kilosort ignores
    labels = pd.read_csv(Path(data_dir) / "cluster_groups.csv", sep="\t")
    good_ids = set(labels.loc[labels["group"] == "good", "cluster_id"])
    orig = ks.get_property("original_cluster_id")
    good = ks.unit_ids[np.isin(orig, list(good_ids))]
    counts = ks.count_num_spikes_per_unit()
    good = [u for u in good if counts[u] >= min_spikes]
    ks = ks.select_units(good)
    m = make_match_count_matrix(ks, golden.to_sorting(), delta_frames=delta).to_numpy()
    n = np.array([counts[u] for u in good])
    recall = m.max(1) / n
    return {"good_units": len(good), "spikes": int(n.sum()),
            "median_recall": float(np.median(recall)) if len(recall) else None,
            "recall_quartiles": np.percentile(recall, [25, 50, 75]).round(3).tolist() if len(recall) else None,
            "delta_samples": delta}
