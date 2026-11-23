"""Load the UCL Neuropixels sample as a SpikeInterface recording (S3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import probeinterface as pi
import spikeinterface.core as si
import spikeinterface.preprocessing as spre

from . import BAND, FILTER_ORDER, FS, TILE_OFFSET

RAW = "rawDataSample.bin"
N_ROWS = 385                      # 384 neural + sync, interleaved by sample
# Phase3 AP: +-0.6 V over 10 bits at gain 500 (from the .ap.meta / imro name)
UV_PER_LSB = 1.2e6 / 1024 / 500   # 2.34375


def load(data_dir: str | Path) -> si.BaseRecording:
    """Neural channels only, in probe order (channel_map.npy), with probe geometry.

    The UCL map lists 374 sites; the 10 reference sites (rows 36, 75, ..., 379)
    and the sync row (384) are dropped. Channel ids are the file rows.
    """
    d = Path(data_dir)
    raw = si.read_binary(d / RAW, sampling_frequency=FS, dtype="int16", num_channels=N_ROWS,
                         gain_to_uV=UV_PER_LSB, offset_to_uV=0.0)
    rows = np.load(d / "channel_map.npy").ravel()
    rec = raw.select_channels(raw.channel_ids[rows])
    probe = pi.Probe(ndim=2)
    probe.set_contacts(np.load(d / "channel_positions.npy"), shapes="square",
                       shape_params={"width": 12})
    probe.set_device_channel_indices(np.arange(len(rows)))
    rec.set_probe(probe, in_place=True)
    return rec


def sync(data_dir: str | Path) -> np.ndarray:
    raw = si.read_binary(Path(data_dir) / RAW, sampling_frequency=FS, dtype="int16",
                         num_channels=N_ROWS)
    return raw.get_traces(channel_ids=[raw.channel_ids[-1]])[:, 0]


def tile(rec: si.BaseRecording, n_channels: int) -> si.BaseRecording:
    """Channel model (S3): channel c = native channel c mod native, delayed by
    floor(c / native) * 10 s. Reads past the end wrap to the start."""
    native = rec.get_num_channels()
    if n_channels <= native:
        return rec.select_channels(rec.channel_ids[:n_channels])
    n = rec.get_num_samples()
    copies = []
    for k in range(-(-n_channels // native)):
        off = (k * TILE_OFFSET) % n
        r = rec if off == 0 else si.concatenate_recordings(
            [rec.frame_slice(off, n), rec.frame_slice(0, off)])
        copies.append(r.rename_channels([f"{c}_t{k}" for c in rec.channel_ids]))
    agg = si.aggregate_channels(copies)
    return agg.select_channels(agg.channel_ids[:n_channels])


def bandpass(rec: si.BaseRecording) -> si.BaseRecording:
    """FR-FLT-1 reference: causal 4th-order Butterworth, 2 biquads, float64 math.

    SpikeInterface filters each chunk with a margin. 100 ms of margin makes the
    chunked result equal to one continuous zero-state ``sosfilt`` to float64
    precision: the slowest pole has radius 0.957, and 0.957^3000 is about 1e-57.
    (The default 'auto' margin of 10 ms leaves errors of about 1e-6, which is
    too close to T-3's 1e-5 tolerance.)
    """
    return spre.bandpass_filter(rec, freq_min=BAND[0], freq_max=BAND[1], filter_order=FILTER_ORDER,
                                ftype="butter", filter_mode="sos", direction="forward",
                                margin_ms=100.0, dtype="float32")


def sos() -> np.ndarray:
    from scipy.signal import butter
    return butter(FILTER_ORDER, BAND, btype="bandpass", fs=FS, output="sos")
